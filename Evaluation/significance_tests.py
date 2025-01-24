from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel, levene, ks_2samp
from statsmodels.stats.anova import AnovaRM
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_best_model_per_metric(ratings_df: pd.DataFrame) -> Dict[str, str]:
    best_models = {}
    for metric in ratings_df.columns:
        mean_scores = ratings_df[metric].apply(lambda x: np.mean(np.array(x)))
        best_models[metric] = mean_scores.idxmax()
        logger.info(f"Best model for {metric}: {best_models[metric]}")
    return best_models

def prepare_long_format_data(ratings_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    long_data_list = []
    
    n_subjects = len(next(iter(ratings_df[metric])))
    
    for subject_idx in range(n_subjects):
        for model_name in ratings_df.index:
            value = ratings_df.loc[model_name, metric][subject_idx]
            long_data_list.append({
                'Subject': subject_idx,
                'Model': model_name,
                'Value': value
            })
                    
    return pd.DataFrame(long_data_list)

def test_normality(data: np.ndarray) -> bool:
    _, p_value = ks_2samp(data, np.random.normal(np.mean(data), np.std(data), len(data)))
    return p_value >= 0.05

def perform_group_significance_test(ratings_df: pd.DataFrame, metric: str) -> Tuple[str, float]:
    normal_tests = []
    for method in ratings_df.index:
        data = np.array(ratings_df.loc[method, metric])
        is_normal = test_normality(data)
        normal_tests.append(is_normal)
        logger.info(f"Normality test for {method} on {metric}: {'normal' if is_normal else 'not normal'}")
    
    if all(normal_tests):
        long_data = prepare_long_format_data(ratings_df, metric)
        try:
            anova = AnovaRM(long_data, 'Value', 'Subject', within=['Model'])
            anova_result = anova.fit()
            return "Repeated Measures ANOVA", anova_result.anova_table['Pr > F'][0]
        except Exception as e:
            logger.warning(f"RM ANOVA failed: {e}, falling back to Friedman test")
    
    friedman_data = []
    for method in ratings_df.index:
        method_data = np.array(ratings_df.loc[method, metric])
        friedman_data.append(method_data)
    
    stat, p_value = friedmanchisquare(*friedman_data)
    return "Friedman Test", p_value

def perform_pairwise_comparison(
    best_scores: List,
    comparison_scores: List
) -> Tuple[str, float]:
    best_array = np.array(best_scores)
    comparison_array = np.array(comparison_scores)
    
    best_normal = test_normality(best_array)
    comparison_normal = test_normality(comparison_array)
    
    if best_normal and comparison_normal:
        _, homogeneity_p = levene(best_array, comparison_array)
        if homogeneity_p >= 0.05:
            _, p_value = ttest_rel(best_array, comparison_array)
            return "Paired t-test", p_value
    
    _, p_value = wilcoxon(best_array, comparison_array)
    return "Wilcoxon Signed-Rank Test", p_value

def compare_models(ratings_df: pd.DataFrame) -> pd.DataFrame:
    comparisons_data: List[Dict] = []
    
    best_models = get_best_model_per_metric(ratings_df)
    
    for metric in ratings_df.columns:
        logger.info(f"Processing metric: {metric}")
        best_model = best_models[metric]
        
        group_test_used, group_p_value = perform_group_significance_test(ratings_df, metric)
        logger.info(f"Group test ({group_test_used}) p-value: {group_p_value}")
        
        if group_p_value < 0.05:  
            logger.info("Significant group difference found, performing pairwise comparisons")
            
            for model_name in ratings_df.index:
                if model_name == best_model:
                    continue    
                
                try:
                    test_used, p_value = perform_pairwise_comparison(
                        ratings_df.loc[best_model, metric],
                        ratings_df.loc[model_name, metric]
                    )
                    
                    comparisons_data.append({
                        "Metric": metric,
                        "Group Test Used": group_test_used,
                        "Group Significance Value": group_p_value,
                        "Used Test": test_used,
                        "Best Method": best_model,
                        "Compared Method": model_name,
                        "Significance Value": p_value,
                        "Significant": p_value < 0.05
                    })
                    
                    logger.info(f"Comparison {best_model} vs {model_name}: {test_used}, p={p_value}")
                except Exception as e:
                    logger.warning(f"Comparison {best_model} vs {model_name} failed: {e}")
        else:
            logger.info("No significant group difference found, skipping pairwise comparisons")
            comparisons_data.append({
            "Metric": metric,
            "Group Test Used": group_test_used,
            "Group Significance Value": group_p_value,
            "Used Test": "None",
            "Best Method": best_model,
            "Compared Method": "None",
            "Significance Value": "None",
            "Significant": False
            })
                    
    return pd.DataFrame(comparisons_data)