import pandas as pd
from datasets import load_dataset
import json 
import os
import pytrec_eval
from Evaluation.significance_tests import compare_models

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)

def get_expected_qrels(task):
    if task == "FinanceBench":
        data = load_dataset("PatronusAI/financebench")["train"].to_pandas()
        data["page_num"] = data["evidence"].apply(lambda x: x[0]["evidence_page_num"])
        qrels = {str(idx): {f"{row.doc_name}.pdf_page_{row.page_num}": 1} for idx, row in data.iterrows()}
        return qrels
    
    elif task == "FinQA":
        dataset = load_dataset("ibm/finqa", trust_remote_code=True)
        data = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()])
        data.reset_index(drop=True, inplace=True)
        data = data[["id", "question", "answer", "gold_inds"]]
        data["Company"] = [row[0] for row in data.id.str.split("/")]
        data["Year"] = [row[1] for row in data.id.str.split("/")]
        data.id = data.id.map(lambda x: x.split("-")[0])
        qrels = {str(idx): {str(row.id): 1} for idx, row in data.iterrows()}
        return qrels

    elif task == "Table_VQA":
        def process_qa_id(qa_id):
            splitted = qa_id.split(".")[0]
            return splitted.split("_")[0] + "/" + splitted.split("_")[1] + "/" + splitted.split("_")[2] + "_" + splitted.split("_")[3] + ".pdf"

        data = load_dataset("terryoo/TableVQA-Bench")["fintabnetqa"].to_pandas()[["qa_id", "question", "gt"]]
        data.qa_id = data.qa_id.apply(process_qa_id)
        data["Company"] = [row[0] for row in data.qa_id.str.split("/")]
        data["Year"] = [row[1] for row in data.qa_id.str.split("/")]
        data = data.rename(columns={"qa_id": "id"})
        qrels = {str(idx): {str(row.id): 1} for idx, row in data.iterrows()}
        return qrels

def evaluate_pipeline(task, directory, expected_qrels):
    ndcg_scores = {}
    mrr_scores = {}
    precision_scores = {}
    recall_scores = {}  
    all_evaluations = {}

    directory = os.path.join(parent_dir, f".results/{task}/retrieval/{directory}")

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Skipping.")
        return ndcg_scores, mrr_scores, precision_scores, recall_scores

    for qrel_file in os.listdir(directory):
        with open(os.path.join(directory, qrel_file), "r") as f:
            qrels = json.load(f)

        qrels = {k: {doc_id: score for doc_id, score in v.items()} for k, v in qrels.items() if k in expected_qrels.keys()}

        truncated_qrels = {}
        for query_id, ranking in qrels.items():
            sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
            truncated_qrels[query_id] = {doc_id: score for doc_id, score in sorted_ranking[:5]}

        evaluator = pytrec_eval.RelevanceEvaluator(expected_qrels, {'ndcg', 'recip_rank', 'P_5', 'recall_5'}) 
        evaluation_results = evaluator.evaluate(truncated_qrels)
        all_evaluations[qrel_file] = evaluation_results

        ndcg_values = [metrics['ndcg'] for metrics in evaluation_results.values()]
        mrr_values = [metrics['recip_rank'] for metrics in evaluation_results.values()]
        precision_values = [metrics['P_5'] for metrics in evaluation_results.values()]
        recall_values = [metrics['recall_5'] for metrics in evaluation_results.values()]

        average_ndcg = np.mean(ndcg_values)
        average_mrr = np.mean(mrr_values)
        average_precision = np.mean(precision_values)
        average_recall = np.mean(recall_values)

        std_ndcg = np.std(ndcg_values)
        std_mrr = np.std(mrr_values)
        std_precision = np.std(precision_values)
        std_recall = np.std(recall_values)

        ndcg_scores[qrel_file] = (average_ndcg, std_ndcg)
        mrr_scores[qrel_file] = (average_mrr, std_mrr)
        precision_scores[qrel_file] = (average_precision, std_precision)
        recall_scores[qrel_file] = (average_recall, std_recall)

    best_qrel_file = max(ndcg_scores, key=lambda x: ndcg_scores[x][0])
    print(f"Best qrel file: {best_qrel_file}")
    print()
    best_evaluation = all_evaluations[best_qrel_file]

    return ndcg_scores, mrr_scores, precision_scores, recall_scores, best_evaluation

def main(task):
    expected_qrels = get_expected_qrels(task=task)
    pipeline_directories = ["text", "colpali", "hybrid", "voyage"]
    
    all_results = []
    evaluation_results = pd.DataFrame(index=pipeline_directories, columns=["nDCG@5", "nDCG@5_std", "MRR@5", "MRR@5_std", "Precision@5", "Precision@5_std", "Recall@5", "Recall@5_std"])  

    for directory in pipeline_directories:
        print(f"Evaluating pipeline: {directory}")
        ndcg, mrr, precision, recall, evaluation_result = evaluate_pipeline(task, directory, expected_qrels)
        evaluation_results.loc[directory, "nDCG@5"] = [metrics['ndcg'] for metrics in evaluation_result.values()]
        evaluation_results.loc[directory, "nDCG@5_std"] = [metrics['ndcg'] for metrics in evaluation_result.values()]
        evaluation_results.loc[directory, "MRR@5"] = [metrics['recip_rank'] for metrics in evaluation_result.values()]
        evaluation_results.loc[directory, "MRR@5_std"] = [metrics['recip_rank'] for metrics in evaluation_result.values()]
        evaluation_results.loc[directory, "Precision@5"] = [metrics['P_5'] for metrics in evaluation_result.values()]
        evaluation_results.loc[directory, "Precision@5_std"] = [metrics['P_5'] for metrics in evaluation_result.values()]
        evaluation_results.loc[directory, "Recall@5"] = [metrics['recall_5'] for metrics in evaluation_result.values()]
        evaluation_results.loc[directory, "Recall@5_std"] = [metrics['recall_5'] for metrics in evaluation_result.values()]

        for qrel_file, (mean, std) in ndcg.items():
            all_results.append({
                "Pipeline": directory,
                "File": qrel_file,
                "nDCG@5": mean,
                "nDCG@5_std": std,
                "MRR@5": mrr[qrel_file][0],
                "MRR@5_std": mrr[qrel_file][1],
                "Precision@5": precision[qrel_file][0],
                "Precision@5_std": precision[qrel_file][1],
                "Recall@5": recall[qrel_file][0],
                "Recall@5_std": recall[qrel_file][1]
            })

    significance_df = compare_models(evaluation_results)
    significance_df.to_excel(os.path.join(current_dir, f"{task}_retrieval_significance_tests.xlsx"))
    results_df = pd.DataFrame(all_results)
    results_df.sort_values(by=["Pipeline", "File"], inplace=True)
    results_df.to_excel(os.path.join(current_dir, f"{task}_retrieval_results.xlsx"), index=False)
    print(f"Results saved to {task}_retrieval_results.xlsx")
    print()
    
if __name__ == "__main__":
    tasks = ["FinQA", "Table_VQA", "FinanceBench"]
    for task in tasks:
        main(task)
