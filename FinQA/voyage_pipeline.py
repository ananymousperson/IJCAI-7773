from datasets import load_dataset
from BasePipelines.voyage_pipeline import VoyagePipeline
import asyncio
from BasePipelines.config import Config
import pandas as pd 

class FinQAVoyagePipeline(VoyagePipeline):

    def prepare_dataset(self):
        dataset = load_dataset("ibm/finqa", trust_remote_code=True)
        data = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()])
        data.reset_index(drop=True, inplace=True)
        data = data[["id", "question", "answer", "gold_inds"]]
        data["Company"] = [row[0] for row in data.id.str.split("/")]
        data["Year"] = [row[1] for row in data.id.str.split("/")]
        data.id = data.id.map(lambda x: x.split("-")[0])
        return data 

    async def process_query(self, data, idx):
        try:
            query = data.loc[idx, "question"]
            company = data.loc[idx, "Company"]
            year = data.loc[idx, "Year"]

            query_embedding = self.embedder.embed_query(query)

            results = asyncio.to_thread( self.faiss_db.search, query_embedding, k=5, metadata_filter={"Company": company, "Year": year})

            qrels = {
                idx: {
                    (result["metadata"]["Company"] + "/" + result["metadata"]["Year"] + "/" + result["metadata"]["Filename"]): 1 / (1 + float(result["distance"]))
                    for result in results
                }
            }

            print("Done with query ", idx)

        except Exception as e:
            qrels = {idx: {}}
            
        self.logger.info(f"Done with query {idx}")
        return qrels

if __name__ == "__main__":
    async def main():
        config = Config(bucket_name="colpali-docs")
        pipeline = FinQAVoyagePipeline(config=config, task="FinQA")
        await pipeline()

    asyncio.run(main())
