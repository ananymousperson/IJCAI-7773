import asyncio
import logging
import os
from abc import ABC, abstractmethod
from PIL import Image
import fitz
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict

with open(f"/Users/emrekuru/Developer/VRAG/.keys/pinecone_api_key.txt",  "r") as file:
    api_key = file.read().strip()

class VoyageEmbedder:
    def __init__(self, vo):
        self.vo = vo

    def pdf_to_images(self, pdf_path, zoom=1.0):
        pdf_document = fitz.open(pdf_path)
        mat = fitz.Matrix(zoom, zoom)
        images = []
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        pdf_document.close()
        return images

    async def embed_document(self, file_path):
        images = self.pdf_to_images(file_path)
        embeddings = []
        for image in images:
            embedding_obj = await asyncio.to_thread(
                self.vo.multimodal_embed,
                inputs=[[image]],
                model="voyage-multimodal-3",
                input_type="document"
            )
            embeddings.append(embedding_obj.embeddings[0])
        return embeddings

class PineconeVectorStore:
    def __init__(self, dimension: int, index_name: str, api_key: str, environment: str):
        self.pc = Pinecone(api_key=api_key)
        
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region=environment  
                )
            )
        
        self.index = self.pc.Index(index_name)
        self.batch_size = 100
        self.pending_vectors = []

    def add(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        vectors = [
            (id_, emb, meta) 
            for id_, emb, meta in zip(ids, embeddings, metadatas)
        ]
        self.pending_vectors.extend(vectors)
        
        if len(self.pending_vectors) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self):
        if not self.pending_vectors:
            return
            
        self.index.upsert(vectors=self.pending_vectors)
        self.pending_vectors = []

    def save(self):
        self._flush_batch()

class VoyageIndexing(ABC):
    def __init__(
        self, 
        config, 
        task, 
        dimension=1024, 
        temp_dir="/tmp/docs/temp", 
        concurrency_limit=8,
        pinecone_api_key=api_key, 
        pinecone_env="us-east-1",   
    ):
        self.config = config
        self.task = task
        self.dimension = dimension
        self.temp_dir = temp_dir
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        self.vector_store = self.initialize_vector_store()
        self.embedder = self.initialize_embedder()

        logging.basicConfig(
            filename=f".logs/{task}-voyage-indexing.log",
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    def initialize_vector_store(self):
        return PineconeVectorStore(
            dimension=self.dimension,
            index_name=f"{self.task.lower()}-index",
            api_key=self.pinecone_api_key,
            environment=self.pinecone_env
        )

    def initialize_embedder(self):
        return VoyageEmbedder(vo=self.config.vo)

    @abstractmethod
    def prepare_metadata(self, file_key, page_number):
        pass

    async def embed_file(self, file_key):
        async with self.semaphore:
            local_file_path = os.path.join(self.temp_dir, os.path.basename(file_key))
            await asyncio.to_thread(
                self.config.s3_client.download_file,
                self.config.bucket_name,
                file_key,
                local_file_path,
            )
            try:
                embeddings = await self.embedder.embed_document(local_file_path)
                for page_number, embedding in enumerate(embeddings):
                    metadata = self.prepare_metadata(file_key)
                    self.vector_store.add(
                        ids=[f"{file_key}_page_{page_number}"],
                        embeddings=[embedding],
                        metadatas=[metadata]
                    )
                logging.info(f"Processed and added file: {file_key} with {len(embeddings)} pages.")
            except Exception as e:
                logging.error(f"Error processing file {file_key}: {e}")
            finally:
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)

    async def process_files(self, prefix=""):
        file_keys = self.config.list_s3_files(prefix=prefix)
        
        with open("/Users/emrekuru/Developer/VRAG/FinanceBench/doc_names.txt", "r") as f:
            file_keys = f.readlines()
            file_keys = [file_key.strip() + ".pdf" for file_key in file_keys]

        tasks = [self.embed_file(file_key) for file_key in file_keys]
        await asyncio.gather(*tasks)

    async def save_index(self):
        self.vector_store.save()
        logging.info("All vectors have been uploaded to Pinecone successfully.")

    async def run(self, prefix=""):
        os.makedirs(self.temp_dir, exist_ok=True)
        await self.process_files(prefix=prefix)
        await self.save_index()