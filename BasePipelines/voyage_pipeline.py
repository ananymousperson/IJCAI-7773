import asyncio
import logging
import json
from abc import ABC, abstractmethod
import os
import math
import numpy as np
from pinecone import Pinecone, ServerlessSpec

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

with open(f"/Users/emrekuru/Developer/VRAG/.keys/pinecone_api_key.txt",  "r") as file:
    api_key = file.read().strip()

class PineconeWithMetadata:
    def __init__(self, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def search(self, query_embedding, k=5, metadata_filter=None):
        filter_dict = metadata_filter if metadata_filter else {}
        
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            filter=filter_dict,
            include_metadata=True
        )
        formatted_results = [
            {
                "id": match.id,
                "score": float(match.score), 
                "metadata": match.metadata
            }
            for match in results.matches
        ]

        return formatted_results

class VoyageEmbedder:
    def __init__(self, voyage_client, batch_size=64):
        self.voyage_client = voyage_client
        self.batch_size = batch_size

    def embed_query(self, query):
        embedding_obj = self.voyage_client.multimodal_embed(
            inputs=[[query]],
            model="voyage-multimodal-3",
            input_type="query"
        )
        return embedding_obj.embeddings[0]

class VoyagePipeline(ABC):
    def __init__(self, config, task, pinecone_index_name, pinecone_env = "us-east-1"):
        self.config = config
        self.task = task
        self.embedder = VoyageEmbedder(config.vo)
        self.vector_db = PineconeWithMetadata(
            index_name=pinecone_index_name,
        )

        self.logger = logging.getLogger(self.task)
        handler = logging.FileHandler(f".logs/{self.task}voyage_retrieval.log", mode="w")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    def prepare_dataset(self):
        pass

    @abstractmethod
    async def process_query(self, data, idx):
        pass

    async def process_queries(self, data):
        tasks = [self.process_query(data, idx) for idx in data.index]
        results = await asyncio.gather(*tasks)

        qrels = {}
        for result in results:
            qrels.update(result)

        return qrels

    async def __call__(self):
        data = self.prepare_dataset()
        
        qrels = await self.process_queries(data)

        os.makedirs(os.path.join(parent_dir, f".results/{self.task}/retrieval/voyage"), exist_ok=True)
        
        with open(os.path.join(parent_dir, f".results/{self.task}/retrieval/voyage/voyage_qrels.json"), "w") as f:
            json.dump(qrels, f, indent=4)

        logging.info("Processing completed and results saved.")