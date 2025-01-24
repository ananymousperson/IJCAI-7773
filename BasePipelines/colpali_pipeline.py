import asyncio
import logging
import json
from abc import ABC, abstractmethod
from byaldi import RAGMultiModalModel
from Generation.generation import image_based
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

class ColpaliPipeline(ABC):
    def __init__(self, config, task, index, mode="generate", device="mps"):
        self.config = config
        self.task = task
        self.index = index
        self.mode = mode
        self.device = device

        logging.basicConfig(
            filename=f".logs/{self.task}/{log_model_type}-colpali_pipeline.log",
            filemode="w",  
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.aws_semaphore = asyncio.Semaphore(10)
        self.qrel_semaphore = asyncio.Semaphore(16)

        if self.mode == "retrieve":
            if os.path.exists(os.path.join(parent_dir, f"{task}/.byaldi/{self.index}")):
                self.RAG = RAGMultiModalModel.from_index(
                    index_path=os.path.join(parent_dir, f"{task}/.byaldi/{self.index}"), device=self.device
                )
            else:
                os.mkdir(os.path.join(parent_dir, f"{task}/.byaldi/"))
                self.config.download_s3_folder("byaldi", os.path.join(parent_dir, f"{task}/.byaldi/"))
                self.RAG = RAGMultiModalModel.from_index(
                    index_path=os.path.join(parent_dir, f"{task}/.byaldi/{self.index}"), device=self.device
                )

    @abstractmethod
    def prepare_dataset(self):
        pass

    @abstractmethod
    async def retrieve(self, idx, data, top_k):
        pass

    async def process_item(self, data, idx, top_k=5, context=None):
        async with self.qrel_semaphore:
            try:
                if self.mode == "retrieve":
                    query, retrieved = await self.retrieve(idx, data, top_k)
                    sorted_retrieved = dict(sorted(retrieved.items(), key=lambda item: item[1]["score"], reverse=True))
                    qrels = {k: v["score"] for k, v in sorted_retrieved.items()}
                    context = {k: v["base64"] for k, v in sorted_retrieved.items()}
                    answer = None

                elif self.mode == "generate":
                    if not context:
                        raise ValueError(f"Context for index {idx} not found in loaded context.")
                    query = data.loc[idx, "question"]
                    answer = await image_based(query, context, model_type=model_type)
                    qrels = None

                logging.info(f"Processed query {idx} successfully.")

            except Exception as e:
                logging.error(f"Error processing query {idx}: {e}")
                raise e

            return idx, qrels, answer, context

    async def process_all(self, qrels, answers, data, batch_size=5):
        context_counter = 1  

        for i in range(0, len(data), 500):  
            logging.info(f"Processing chunk {i // 500 + 1}")

            chunk_start = i
            chunk_end = min(i + 500, len(data))

            context_file = os.path.join(parent_dir, f".results/{self.task}/generation/image/context_{context_counter}.json")
            if os.path.exists(context_file):
                with open(context_file, "r") as f:
                    current_context = json.load(f)
            else:
                current_context = {}

            for j in range(chunk_start, chunk_end, batch_size):
                results = []
                tasks = []
                batch_start = j
                batch_end = min(j + batch_size, chunk_end)

                for k in range(batch_start, batch_end):

                    idx = data.index[k] if k < len(data) else None
                    if self.mode == "retrieve" and f'{idx}' in qrels.keys():
                        continue
                    elif self.mode == "generate" and f'{idx}' in answers.keys():
                        continue
                    else:
                        query_context = current_context[f'{idx}'] if self.mode == "generate" else None
                        tasks.append(self.process_item(data, idx, context=query_context))

                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)

                for idx, qrel, answer, query_context in results:

                    if self.mode == "retrieve":
                        qrels[idx] = qrel
                        current_context[f'{idx}'] = query_context

                    elif self.mode == "generate":
                        answers[idx] = answer

                if self.mode == "retrieve":

                    with open(context_file, "w") as f:
                        json.dump(current_context, f, indent=4)

                    with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/colpali/colpali_qrels.json"), "w") as f:
                        json.dump(qrels, f, indent=4)
                else:
                    with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/{log_model_type}_answers.json"), "w") as f:
                        json.dump(answers, f, indent=4)

            current_context.clear()
            context_counter += 1 

    async def __call__(self):
        data = self.prepare_dataset()
        print(log_model_type)
        
        if not os.path.exists(os.path.join(parent_dir, f".results/{self.task}/retrieval/colpali/colpali_qrels.json")):
            qrels = {}
        else:
            with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/colpali/colpali_qrels.json"), "r") as f:
                qrels = json.load(f)

        if not os.path.exists(os.path.join(parent_dir, f".results/{self.task}/generation/image/{log_model_type}_answers.json")):
            answers = {}
        else:
            with open(os.path.join(parent_dir, f".results/{self.task}/generation/image/{log_model_type}_answers.json"), "r") as f:
                answers = json.load(f)
                answers = {k: v for k, v in answers.items() if v != ""}
                answers = {k: v for k, v in answers.items() if v["answer"] != "Error"} 
                answers = {k: v for k, v in answers.items() if v["answer"] != ""}

        await self.process_all(qrels=qrels, answers=answers, data=data)

        print("Finished")
