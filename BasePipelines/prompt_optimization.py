import json
import os
import logging
import pandas as pd
from datasets import load_dataset
from Generation.generation import sync_text_based, sync_image_based, sync_hybrid
from dspy import Module
from dspy.teleprompt import MIPROv2
from Generation.prompts import TEXT_PROMPT, IMAGE_PROMPT, HYBRID_PROMPT, O1_IMAGE_PROMPT, O1_HYBRID_PROMPT, O1_TEXT_PROMPT
from sklearn.model_selection import train_test_split
import dspy
from dspy import Signature, Predict, InputField, OutputField
from Evaluation.Generation.evaluation import compute_token_f1

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

model_type = "google-gemini-2.0-flash-thinking-exp"
log_model_type = "gemini-2.0-flash-thinking-exp"

with open(f".keys/openai_api_key.txt", "r") as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()
    api_key = file.read().strip()

lm = dspy.LM('openai/gpt-4o', api_key=api_key)
dspy.configure(lm=lm)

class MultiModalGeneration(Module):
    def __init__(self, input_type, prompt_template):
        super().__init__()
        self.input_type = input_type
        self.prompt_template = prompt_template

        self.PromptSignature = self._create_signature(input_type)
        self.PromptSignature.instructions = self.prompt_template.template
    
        self.predictor = Predict(
            signature=self.PromptSignature,
            instructions = self.prompt_template.template,
        )

    def _create_signature(self, input_type):
        class DynamicSignature(Signature):
            query = InputField(description="The input query or question.")
            answer = OutputField(description="The generated answer to the query.")
        
        if input_type == "text":
            DynamicSignature.text_context = InputField(optional=False, description="Text-based context for the query.")
        elif input_type == "image":
            DynamicSignature.image_context = InputField(optional=False, description="Image-based context for the query.")
        elif input_type == "hybrid":
            DynamicSignature.text_context = InputField(optional=False, description="Text-based context for the query.")
            DynamicSignature.image_context = InputField(optional=False, description="Image-based context for the query.")
        else:
            raise ValueError("Unsupported input type")
        
        return DynamicSignature

    def predictors(self):
        return [self.predictor]
    
    def add_schema(self, prompt_template):
        if "{schema}" not in prompt_template:
            prompt_template = prompt_template + "\n\nYou must produce your answer in the following strict JSON format:\n{schema}"

        return prompt_template

    def forward(self, query, text_context=None, image_context=None):

        self.prompt_template.template = self.add_schema(self.predictor.signature.instructions)

        logging.info(f"Query: {query}\nPrompt Template: {self.prompt_template.template}")

        try:
            if self.input_type == "text":
                response = sync_text_based(query=query, chunks=text_context, prompt_template=self.prompt_template, model_type=model_type)
            elif self.input_type == "image":
                response = sync_image_based(query=query, pages=image_context, prompt_template=self.prompt_template, model_type=model_type)
            elif self.input_type == "hybrid":
                response = sync_hybrid(query=query, chunks=text_context, pages=image_context, prompt_template=self.prompt_template, model_type=model_type)
            else:
                raise ValueError("Unsupported input type")
            
            logging.info(f"Query: {query}\nPrompt Template: {self.prompt_template.template}\nAnswer: {response['answer']}")

        except Exception as e:
            logging.error(f"Error in generating answer: {e}")
            raise e
        
        return {"answer": response["answer"]}
    
    def predictors(self):
        return [self.predictor]


class PromptOptimizationPipeline:

    def __init__(self, input_type):
        self.input_type = input_type

        if input_type == "text":
            self.prompt_template = O1_TEXT_PROMPT
        elif input_type == "image":
            self.prompt_template = O1_IMAGE_PROMPT
        elif input_type == "hybrid":
            self.prompt_template = O1_HYBRID_PROMPT
        else:
            raise ValueError("Unsupported input type")

        log_dir = f".logs/{self.input_type}_prompt_optimization"
        os.makedirs(log_dir, exist_ok=True)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=os.path.join(log_dir, f"{log_model_type}_prompt_optimization.log"),
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    def prepare_data(self, task):
        if task == "FinanceBench":
            data = load_dataset("PatronusAI/financebench")["train"].to_pandas()
            data["page_num"] = data["evidence"].apply(lambda x: x[0]["evidence_page_num"])
            data["evidence"] = data["evidence"].apply(lambda x: x[0]["evidence_text"])
            return data

        elif task == "FinQA":
            dataset = load_dataset("ibm/finqa", trust_remote_code=True)
            data = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()])
            data.reset_index(drop=True, inplace=True)
            data = data[["id", "question", "answer", "gold_inds"]]
            data["Company"] = [row[0] for row in data.id.str.split("/")]
            data["Year"] = [row[1] for row in data.id.str.split("/")]
            data.id = data.id.map(lambda x: x.split("-")[0])
            data = data.sample(n=700, random_state=42)
            return data

        elif task == "Table_VQA":
            def process_qa_id(qa_id):
                splitted = qa_id.split(".")[0]
                return splitted.split("_")[0] + "/" + splitted.split("_")[1] + "/" + splitted.split("_")[2] + "_" + splitted.split("_")[3] + ".pdf"

            data = load_dataset("terryoo/TableVQA-Bench")["fintabnetqa"].to_pandas()[["qa_id", "question", "gt", "text_html_table"]]
            data.qa_id = data.qa_id.apply(process_qa_id)
            data["Company"] = [row[0] for row in data.qa_id.str.split("/")]
            data["Year"] = [row[1] for row in data.qa_id.str.split("/")]
            data = data.rename(columns={"qa_id": "id", "gt": "answer", "text_html_table": "evidence"})
            return data

    def load_text_context(self, task):
        with open(os.path.join(parent_dir, f".results/{task}/generation/text/context.json"), "r") as f:
            text_context = json.load(f)
        return text_context
    
    def load_image_context(self, task, batch):
        with open(os.path.join(parent_dir, f".results/{task}/generation/image/context_{batch}.json"), "r") as f:
            image_context = json.load(f)
        return image_context
        
    def __call__(self):
        program = MultiModalGeneration(input_type=self.input_type, prompt_template=self.prompt_template)

        def evaluate_answer(example, pred, trace = None):
            try:
                f1_score =compute_token_f1(
                        prediction=pred["answer"],
                        reference=example["expected_answer"],
                )

            except Exception as e: 
                logging.error(f"Error in evaluating answer: {e}")
                raise e
            
            return f1_score

        teleprompter = MIPROv2(
            task_model=lm,
            metric=evaluate_answer,
            verbose=True,
            num_candidates=5,
            init_temperature=0.5,
            num_threads=1,
        )

        trainset = []
        valset = []

        for task, size in zip(["FinanceBench", "Table_VQA", "FinQA"], [150, 250, 8500]):

            dataset = self.prepare_data(task=task)

            text_context = self.load_text_context(task)

            logging.info(f"Loaded {task} dataset with {len(dataset)} examples.")

            context_counter = 1  

            for batch in range(0, size, 500):

                if self.input_type == "image" or self.input_type == "hybrid":
                    image_context = self.load_image_context(task, context_counter)
                else:
                    image_context = {}

                temp_dataset = dataset[dataset.index.isin(range(batch, min(batch + 500, size)))]

                if len(temp_dataset) == 0:
                    continue

                try:
                    train_data, val_data = train_test_split(temp_dataset, test_size=0.2, random_state=42)
                except Exception as e:
                    train_data = temp_dataset
                    val_data = pd.DataFrame()

                for idx, row in train_data.iterrows():
                    trainset.append(
                        dspy.Example(
                            query=row["question"],
                            text_context= list(text_context[str(idx)].values()) if self.input_type == "text" or  self.input_type == "hybrid" else [],
                            image_context=tuple(image_context[str(idx)]) if self.input_type == "image" or  self.input_type == "hybrid" else [],
                            expected_answer=row["answer"]
                        ).with_inputs("query", "text_context", "image_context")
                    )

                for idx, row in val_data.iterrows():
                    valset.append(
                        dspy.Example(
                            query=row["question"],
                            text_context= list(text_context[str(idx)].values()) if self.input_type == "text" or  self.input_type == "hybrid" else [],
                            image_context=tuple(image_context[str(idx)]) if self.input_type == "image" or self.input_type == "hybrid" else [],
                            expected_answer=row["answer"]
                        ).with_inputs("query", "text_context", "image_context")
                    )

                image_context.clear()
                context_counter += 1

        logging.info(f"Loaded {len(trainset)} training examples and {len(valset)} validation examples.")

        optimized_prompt = teleprompter.compile(
            student=program,
            trainset=trainset,
            valset=valset,
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
            num_trials=5,
            minibatch_size=10,
            minibatch_full_eval_steps=7,
            minibatch=True,
            requires_permission_to_run=True,
        )

        final_instructions = optimized_prompt.predictors()[0].signature.instructions
        with open(f".results/prompts/{self.input_type}_optimized_prompt.json", "w") as f:
            json.dump(final_instructions, f, indent=4)

        logging.info("Optimization finished successfully. Optimized prompt saved.")

if __name__ == "__main__":
    input_type = "text"
    pipeline = PromptOptimizationPipeline(input_type)
    pipeline()
