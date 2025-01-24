import asyncio
import logging
import json
from abc import ABC, abstractmethod
from langchain_community.vectorstores import Chroma
from Generation.generation import text_based
import math
import os
import numpy as np

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

class Embedder:
    def __init__(self, vo, batch_size=64):
        self.batch_size = batch_size
        self.vo = vo

    def _normalize(self, embedding):
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm != 0 else embedding

    def _normalize_batch(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def embed_document(self, text):
        embedding = self.vo.embed(
            [text], model="voyage-3-large", input_type="document",
            output_dimension=2048, output_dtype="int8"
        ).embeddings[0]
        return self._normalize(embedding)

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.vo.embed(
                batch, model="voyage-3-large", input_type="document",
                output_dimension=2048, output_dtype="int8"
            ).embeddings
            normalized_batch = self._normalize_batch(batch_embeddings)
            embeddings.extend(normalized_batch)
        return embeddings

    def embed_query(self, query):
        embedding = self.vo.embed(
            [query], model="voyage-3-large", input_type="query",
            output_dimension=2048, output_dtype="int8"
        ).embeddings[0]
        return self._normalize(embedding)


class TextPipeline(ABC):
    def __init__(self, config, task, persist_directory=".chroma", mode="generate"):
        self.config = config
        self.task = task
        self.persist_directory = persist_directory
        self.mode = mode  # "retrieve" or "generate"
        self.embedder = Embedder(self.config.vo, batch_size=64)

        logging.basicConfig(
            filename=f".logs/{self.task}/{log_model_type}-text_pipeline.log",
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.qrel_semaphore = asyncio.Semaphore(16)

    @abstractmethod
    def prepare_dataset(self):
        pass

    @abstractmethod
    def read_chunks(self):
        pass

    async def create_db(self, chunks, batch_size=500):
        if os.path.exists(self.persist_directory):
            self.chroma_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedder)
            logging.info("Loaded existing ChromaDB.")
        else:
            self.chroma_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedder)
            logging.info("Created a new ChromaDB.")

            num_batches = math.ceil(len(chunks) / batch_size)
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, len(chunks))
                batch = chunks[batch_start:batch_end]

                self.chroma_db.add_texts(
                    texts=[chunk.page_content for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch]
                )
                logging.info(f"Added batch {i + 1}/{num_batches} to ChromaDB.")

        return self.chroma_db

    @abstractmethod
    async def retrieve(self, idx, data, top_n):
        pass

    async def rerank(self, query, documents, ids, reranker, top_k=5,):
        scores = {}

        if reranker == "cohere":
            rerank_response = await asyncio.to_thread(
                self.config.co.rerank,
                query=query,
                documents=documents,
                top_n=top_k,
                model='rerank-v3.5'
            )
            for result in rerank_response.results:
                scores[ids[result.index]] = {
                    "score": result.relevance_score,
                    "content": documents[result.index]
                }

            return scores

        elif reranker == "colbert":
            indexed_docs = [{"id": doc_id, "content": doc} for doc_id, doc in zip(ids, documents)]
            doc_contents = [doc["content"] for doc in indexed_docs]
            results = await asyncio.to_thread( self.config.colbert_model.rerank,
                query=query,
                documents=doc_contents,
                k=top_k  
            )
            for idx, result in enumerate(results):
                scores[indexed_docs[idx]["id"]] = {
                    "score": result["score"],
                    "content": result["content"]
                }

            return scores
        
        else:
            rerank_response = await asyncio.to_thread(
                self.config.vo.rerank,
                query=query,
                documents=documents,
                top_k=top_k,
                model='rerank-2'
            )
            for result in rerank_response.results:
                scores[ids[result.index]] = {
                    "score": result.relevance_score,
                    "content": documents[result.index]
                }

            return scores

    async def process_item(self, data, idx, top_n=10, context=None):
        async with self.qrel_semaphore:
            try:

                if self.mode == "retrieve":
                    query, ids, documents, scores = await self.retrieve(idx, data, top_n)
                    retrieved_qrels = {k: v for k, v in zip(ids, scores)}
                    sorted_qrels = dict(sorted(retrieved_qrels.items(), key=lambda item: item[1], reverse=True)[:5])
                    context = {k: v for k, v in zip(ids, documents)}

                    if self.rerank_:
                        reranked = await self.rerank(query, documents, ids, reranker=self.reranker)
                        sorted_retrieved = dict(sorted(reranked.items(), key=lambda item: item[1]["score"], reverse=True))
                        sorted_qrels = {k: v["score"] for k, v in sorted_retrieved.items()}
                        context = {k: v["content"] for k, v in sorted_retrieved.items()}

                    answer = None

                elif self.mode == "generate":
                    if not context:
                        raise ValueError(f"No context found for index {idx}")
                    query = data.loc[idx, "question"]
                    answer = await text_based(query, list(context.values()), model_type=model_type)
                    sorted_qrels = None

                logging.info(f"Processed query {idx} successfully.")
                return idx, sorted_qrels, answer, context

            except Exception as e:
                logging.error(f"Error processing query {idx}: {e}")
                return idx, {}, "", context

    async def process_all(self, data, qrels={}, answers={}, context={}, batch_size=8):
        for i in range(0, len(data), batch_size):
            tasks = []
            batch_data = data[i:i + batch_size]

            for idx in batch_data.index:
                if self.mode == "retrieve" and str(idx) in qrels:
                    continue
                if self.mode == "generate" and str(idx) in answers:
                    continue
                tasks.append(self.process_item(data, idx, context=context.get(str(idx))))

            results = await asyncio.gather(*tasks)

            for idx, qrel, answer, query_context in results:
                if self.mode == "retrieve":
                    qrels[idx] = qrel
                    context[str(idx)] = query_context
                if self.mode == "generate":
                    answers[idx] = answer

            if self.mode == "retrieve":
                with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/qrels.json"), "w") as f:
                    json.dump(qrels, f, indent=4)
                with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json"), "w") as f:
                    json.dump(context, f, indent=4)
            elif self.mode == "generate":
                with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/{log_model_type}_answers.json"), "w") as f:
                    json.dump(answers, f, indent=4)


    async def __call__(self):
        data = self.prepare_dataset()
        print(log_model_type)

        if self.mode == "retrieve":
            qrels = {}
            context = {}
            
            self.rerank_ = True
            self.reranker = "reranker-2"

            if os.path.exists(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json")):
                with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json"), "r") as f:
                    context = json.load(f)

            chunks = self.read_chunks()
            self.chroma_db = await self.create_db(chunks)

            with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/qrels.json"), "w") as f:
                json.dump(qrels, f, indent=4)

            with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json"), "w") as f:
                json.dump(context, f, indent=4)

        elif self.mode == "generate":
            qrels = {}

            context = {}
            if os.path.exists(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json")):
                with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/context.json"), "r") as f:
                    context = json.load(f)

            answers = {}
            if os.path.exists(os.path.join(parent_dir, f".results/{self.task}/generation/text/{log_model_type}_answers.json")):
                with open(os.path.join(parent_dir, f".results/{self.task}/generation/text/{log_model_type}_answers.json"), "r") as f:
                    answers = json.load(f)
                    answers = {k: v for k, v in answers.items() if v != ""}
                    answers = {k: v for k, v in answers.items() if v["answer"] != "Error"}
                    answers = {k: v for k, v in answers.items() if v["answer"] != ""}

        await self.process_all(data, qrels=qrels, answers=answers, context=context)

        print("Finished")
