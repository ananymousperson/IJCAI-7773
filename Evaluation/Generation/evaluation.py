import os
import evaluate
import pandas as pd
from datasets import load_dataset
import json
from collections import Counter
import re
from io import StringIO
import asyncio
import logging
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from langchain_openai import ChatOpenAI
from .faithfullnes import evaluate_multimodal_faithfulness

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))

with open(os.path.join(parent_dir, ".keys/openai_api_key.txt"), "r") as file:
    openai_key = file.read().strip()

os.environ["OPENAI_API_KEY"] = openai_key

correctness_metric = GEval(
    name="Correctness",
    criteria="Evaluate the factual correctness of the generated answer output given the correct expected answer.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    verbose_mode=False,
)

# Set up logging
logging.basicConfig(
    filename=f".logs/Evaluation/generation-evaluation.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

exact_match = evaluate.load("exact_match")
f1 = evaluate.load("f1")
meteor = evaluate.load("meteor")
bert_score = evaluate.load("bertscore")

CURRENCY_SYMBOLS = {"$", "€", "£", "¥"}  

def html_to_string(html_string):
    df = pd.read_html(StringIO(html_string))[0]
    df.set_index(0, inplace=True)
    df.columns = df.iloc[0]
    df = df[1:]
    df.index.name = None
    return df.to_string()

def normalize_numeric_token(token):
    token = "".join(ch for ch in token if ch not in CURRENCY_SYMBOLS)
    token = token.replace(",", "")
    try:
        float_val = float(token)
        token = str(float_val)
    except ValueError:
        pass
    
    return token

def preprocess_text(text):
    text = text.lower()
    text = text.encode('unicode_escape').decode('unicode_escape')
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    text = text.strip()
    return text

def tokenize(text):
    tokens = re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?", text)
    return tokens

def domain_normalize_tokens(tokens):
    normalized_tokens = []
    for t in tokens:
        if any(ch.isdigit() for ch in t):
            t = normalize_numeric_token(t)
        normalized_tokens.append(t)
    return normalized_tokens

def compute_token_f1(prediction, reference):
    pred_text = preprocess_text(prediction)
    ref_text = preprocess_text(reference)

    pred_tokens = tokenize(pred_text)
    ref_tokens = tokenize(ref_text)

    pred_tokens = domain_normalize_tokens(pred_tokens)
    ref_tokens = domain_normalize_tokens(ref_tokens)

    common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common_tokens.values())

    precision = num_common / len(pred_tokens) if pred_tokens else 0.0
    recall = num_common / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def prepare_data(task):
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

async def evaluate_query_async(query_idx, row, generations, textual_context, image_context, subfolder, simple_metrics, semaphore, metrics_to_be_calculated):
    async with semaphore:
        try:
            query = row["question"]
            golden_context = row["evidence"]
            expected_output = row["answer"]

            image_context_ = image_context[str(query_idx)]
            textual_context_ = list(textual_context[str(query_idx)].values())

            reasoning = generations[str(query_idx)]["reasoning"]
            reasoning = " ".join(reasoning) if isinstance(reasoning, list) else reasoning
            answer = generations[str(query_idx)]["answer"]
            actual_output = reasoning + "\n\n" + answer

            result = {"Subfolder": subfolder, "Index": query_idx}

            if "Exact Match" in metrics_to_be_calculated:
                result["Exact Match"] = exact_match.compute(
                    predictions=[preprocess_text(answer)],
                    references=[preprocess_text(expected_output)],
                )["exact_match"]

            if "F1-Score" in metrics_to_be_calculated:
                result["F1-Score"] = compute_token_f1(answer, expected_output)

            if "METEOR" in metrics_to_be_calculated:
                meteor_result = meteor.compute(
                    predictions=[preprocess_text(answer)],
                    references=[preprocess_text(expected_output)],
                )
                result["METEOR"] = meteor_result["meteor"]

            if "BERTScore" in metrics_to_be_calculated:
                bert_result = bert_score.compute(
                    predictions=[preprocess_text(answer)],
                    references=[preprocess_text(expected_output)],
                    model_type="bert-base-uncased",
                )
                result["BERTScore"] = bert_result["f1"][0]

            if "Correctness" in metrics_to_be_calculated:
                    
                def run_g_eval():
                    test_case = LLMTestCase(
                        input=golden_context,
                        actual_output=actual_output,
                        expected_output=expected_output
                    )
                    
                    correctness_metric.measure(test_case)

                    return {
                        "G-Eval Score": correctness_metric.score,
                        "G-Eval Reasoning": correctness_metric.reason,
                    }

                g_eval_result = await asyncio.to_thread(run_g_eval)
                result["Correctness"] = g_eval_result["G-Eval Score"]


            if "Faithfulness Score" in metrics_to_be_calculated:
                if subfolder == "hybrid":
                    faithfulness_score, faithfulness_reason = await evaluate_multimodal_faithfulness(
                        query, actual_output, textual_context_, image_context_
                    )
                elif subfolder == "text":
                    faithfulness_score, faithfulness_reason = await evaluate_multimodal_faithfulness(
                        query, actual_output, textual_context_, []
                    )
                elif subfolder == "image":
                    faithfulness_score, faithfulness_reason = await evaluate_multimodal_faithfulness(
                        query, actual_output, [], image_context_
                    )

                result["Faithfulness Score"] = faithfulness_score
                result["Faithfulness Reason"] = faithfulness_reason

            for metric in metrics_to_be_calculated:
                simple_metrics.loc[query_idx, metric] = result[metric]

            logging.info(f"Completed query {query_idx} for {subfolder}")

        except Exception as e:
            logging.error(f"Error evaluating query {query_idx} in {subfolder}: {e}")

async def evaluate_generation(task, generation_folder):
    semaphore = asyncio.Semaphore(16)
    data = prepare_data(task)
    metrics = ["Subfolder", "Index", "Exact Match", "F1-Score", "METEOR", "BERTScore"]

    with open(os.path.join(parent_dir, f".results/{task}/generation/text/context.json"), "r") as f:
        textual_context = json.load(f)

    with open(os.path.join(parent_dir, f".results/{task}/generation/image/context_1.json"), "r") as f:
        image_context = json.load(f)
    
    for subfolder in ["image", "text", "hybrid"]:
        subfolder_path = os.path.join(generation_folder, subfolder)

        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist. Skipping.")
            continue

        answer_files = [f for f in os.listdir(subfolder_path) if f.endswith("answers.json")]
        print(f"{task} - {subfolder}")
        print(answer_files)

        if not answer_files:
            print(f"No answers files found in {subfolder_path}. Skipping.")
            continue

        for answer_file in answer_files:
            model_type = answer_file.split(("_"))[0]

            answer_file_path = os.path.join(subfolder_path, answer_file)
            with open(answer_file_path, "r") as f:
                generations = json.load(f)
                generations = {k: v for k, v in generations.items() if v != ""}
                generations = {k: v for k, v in generations.items() if v != None}
                generations = {k: v for k, v in generations.items() if v["answer"] != "Error"}
                generations = {k: v for k, v in generations.items() if v["answer"] != ""}

            if os.path.exists(os.path.join(subfolder_path, f"{current_dir}/intermidiate_results/{task}/{subfolder}/{model_type}_metrics.csv")):
                simple_metrics = pd.read_csv(os.path.join(subfolder_path, f"{current_dir}/intermidiate_results/{task}/{subfolder}/{model_type}_metrics.csv"))
                metrics_to_be_calculated = [metric for metric in metrics if metric not in simple_metrics.columns]
            else:
                simple_metrics = pd.DataFrame([], columns=metrics)
                metrics_to_be_calculated = metrics
                for metric in metrics_to_be_calculated:
                    simple_metrics[metric] = None

            print(metrics_to_be_calculated)

            tasks = [
                evaluate_query_async(idx, row, generations, textual_context, image_context, subfolder, simple_metrics, semaphore, metrics_to_be_calculated)
                for idx, row in data.iterrows() 
            ]

            await asyncio.gather(*tasks)

            simple_df = pd.DataFrame(simple_metrics)
            simple_df.to_csv(os.path.join(subfolder_path, f"{current_dir}/intermidiate_results/{task}/{subfolder}/{model_type}_metrics.csv"), index=False)

            print(f"Results saved for {task} in {subfolder_path}")

if __name__ == "__main__":
    tasks = ["Table_VQA", "FinanceBench"]

    async def main():
        for task in tasks:
            generation_folder = os.path.join(parent_dir, f".results/{task}/generation")
            await evaluate_generation(task, generation_folder)

    asyncio.run(main())