import json
import os
import logging
import pandas as pd
from datasets import load_dataset
from Generation.generation import hybrid
import asyncio
import math 
import os 

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

# model_type = "openai-gpt-4o"
# log_model_type = "gpt-4o"

# model_type = "openai-o1"
# log_model_type = "o1"

# model_type = "claude-claude"
# log_model_type = "claude"

# model_type = "google-gemini-2.0-flash-exp"
# log_model_type = "gemini-2.0-flash-exp"

# model_type = "google-gemini-1.5-pro"
# log_model_type = "gemini-1.5-pro"

model_type = "google-gemini-2.0-flash-thinking-exp"
log_model_type = "gemini-2.0-flash-thinking-exp"

# model_type = "openrouter-meta-llama/llama-3.2-90b-vision-instruct"
# log_model_type = "llama-3.2-90b-vision-instruct"

# model_type = "openrouter-deepseek/deepseek-chat"
# log_model_type = "deepseek-reasoning"

# model_type = "openrouter-accounts/fireworks/models/qwen2-vl-72b-instruct"
# log_model_type = "fireworks-qwen-2-vl-72b-instruct"

# model_type = "openrouter-deepseek/deepseek-r1"
# log_model_type = "deepseek-r1"

class HybridPipeline:
    def __init__(self, task, max_concurrent_tasks=16):
        self.task = task
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

        log_dir = f".logs/{self.task}"
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(log_dir, f"{log_model_type}-hybrid_generation.log"),
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    def prepare_data(self, task):
        try:
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

        except Exception as e:
            logging.error(f"Error preparing data for task {task}: {e}")
            raise

    def load_contexts(self):
            with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json"), "r") as f:
                text_context = json.load(f)
            return text_context

    async def process_query(self, query_id, query, text_context, image_context):
        async with self.semaphore:
            try:
                answer = await hybrid(query=query, pages=image_context, chunks=text_context, model_type=model_type)
                logging.info(f"Generated answer for query {query_id}")
                return query_id, answer
            except Exception as e:
                logging.error(f"Error processing query {query_id}: {e}")
                return query_id, {"reasoning": "Error", "answer": "Error"}

    async def generate_answers(self, dataset, text_context, answers, batch_size=16):
    
        output_dir = os.path.join(parent_dir, f".results/{self.task}/generation/hybrid")
        os.makedirs(output_dir, exist_ok=True)

        context_counter = 1  
        chunk_size = 500 

        for i in range(0, len(dataset["question"]), chunk_size):
            logging.info(f"Processing chunk {i // chunk_size + 1}")
            
            context_file = os.path.join(parent_dir, f".results/{self.task}/generation/image/context_{context_counter}.json")

            if os.path.exists(context_file):
                with open(context_file, "r") as f:
                    image_context = json.load(f)
            else:
                print(f"Context file {context_file} not found")
                image_context = {}  

            for j in range(i, min(i + chunk_size, len(dataset["question"])), batch_size):
                logging.info(f"Processing batch {j // batch_size + 1} within chunk")
                
                tasks = []
                for query_id, query in dataset.iloc[j: j + batch_size]["question"].items():
                    if str(query_id) in answers.keys():
                        continue  
                    else:
                        text_context_for_query = text_context[str(query_id)]
                        image_context_for_query = image_context[str(query_id)]
                        tasks.append(self.process_query(query_id, query, text_context_for_query, image_context_for_query))

                batch_results = await asyncio.gather(*tasks)

                for query_id, answer in batch_results:
                    answers[query_id] = answer

                with open(os.path.join(output_dir, f"{log_model_type}_answers.json"), "w") as f:
                    json.dump(answers, f, indent=4)

            context_counter += 1

        return answers


    async def __call__(self):
        print(log_model_type)
        output_dir = os.path.join(parent_dir, f".results/{self.task}/generation/hybrid")
        os.makedirs(output_dir, exist_ok=True)
    
        try:
            dataset = self.prepare_data(self.task)
            text_context = self.load_contexts()

            if not os.path.exists(os.path.join(output_dir, f"{log_model_type}_answers.json")):
                answers = {}
            else:
                with open(os.path.join(output_dir, f"{log_model_type}_answers.json"), "r") as f:
                    answers = json.load(f)
                answers = {k: v for k, v in answers.items() if v != ""}
                answers = {k: v for k, v in answers.items() if v["answer"] != "Error"} 
                answers = {k: v for k, v in answers.items() if v["answer"] != ""}

            answers = await self.generate_answers(dataset, text_context, answers)

            with open(os.path.join(output_dir, f"{log_model_type}_answers.json"), "w") as f:
                json.dump(answers, f, indent=4)

            logging.info("Pipeline finished successfully.")
            print("Finished")

        except Exception as e:
            logging.error(f"Error in pipeline execution: {e}")
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    task = "Table_VQA"
    pipeline = HybridPipeline(task)
    asyncio.run(pipeline())
