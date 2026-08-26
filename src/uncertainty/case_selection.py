import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.config as config

logger = logging.getLogger("Uncertainty_CaseSelection")

def select_uncertainty_cases(df: pd.DataFrame, num_cases: int = 25) -> pd.DataFrame:
    """
    Selects 20-30 representative cases based on their uncertainty profiles and predictions.
    
    Includes:
      - High-confidence correct (Low uncertainty, correct prediction)
      - High-confidence incorrect (Low uncertainty, incorrect prediction - silent failures)
      - High-uncertainty correct (High uncertainty, correct prediction)
      - High-uncertainty incorrect (High uncertainty, incorrect prediction)
      - Small top-2 margin borderline cases
      - Grade-confusion cases (adjacent grades and severe mismatches)
      - Class-wise representatives
    """
    logger.info(f"Selecting {num_cases} representative uncertainty cases...")
    selected_dfs = []
    
    # Helpers to filter correct vs incorrect
    correct_df = df[df["is_correct"]]
    incorrect_df = df[~df["is_correct"]]
    
    def add_case(subset, reason, sort_col, ascending=True):
        if not subset.empty:
            sorted_subset = subset.sort_values(sort_col, ascending=ascending)
            case = sorted_subset.head(1).copy()
            case["selected_reason"] = reason
            selected_dfs.append(case)
            
    # 1. High-confidence correct (low entropy, correct)
    add_case(correct_df, "High-confidence correct (Low entropy)", "calib_entropy", ascending=True)
    
    # 2. High-confidence incorrect (low entropy, incorrect) - Silent Failures!
    add_case(incorrect_df, "High-confidence incorrect (Silent failure)", "calib_entropy", ascending=True)
    add_case(incorrect_df, "High-confidence incorrect (Low MC variance)", "predictive_variance", ascending=True)
    
    # 3. High-uncertainty correct (high entropy, correct) - Model was hesitant but right
    add_case(correct_df, "High-uncertainty correct (Hesitant success)", "calib_entropy", ascending=False)
    add_case(correct_df, "High MC variance correct", "predictive_variance", ascending=False)
    add_case(correct_df, "High mutual information correct", "mutual_information", ascending=False)
    
    # 4. High-uncertainty incorrect (high entropy, incorrect) - Identified failure
    add_case(incorrect_df, "High-uncertainty incorrect (Identified failure)", "calib_entropy", ascending=False)
    add_case(incorrect_df, "High MC variance incorrect", "predictive_variance", ascending=False)
    add_case(incorrect_df, "High mutual information incorrect", "mutual_information", ascending=False)
    
    # 5. Borderline cases (smallest top-2 margin)
    add_case(df, "Borderline prediction (Smallest margin)", "calib_margin", ascending=True)
    
    # 6. Class-wise representative correct cases
    for g in sorted(df["ground_truth"].unique()):
        g_correct = correct_df[correct_df["ground_truth"] == g]
        if not g_correct.empty:
            add_case(g_correct, f"Representative correct (Grade {g})", "calib_confidence", ascending=False)
            
    # 7. Class-wise representative incorrect cases
    for g in sorted(df["ground_truth"].unique()):
        g_incorrect = incorrect_df[incorrect_df["ground_truth"] == g]
        if not g_incorrect.empty:
            add_case(g_incorrect, f"Failure example (Grade {g})", "calib_confidence", ascending=False)
            
    # 8. Adjacent grade confusion (e.g. Mild 1 vs Moderate 2)
    adj_confusion = incorrect_df[abs(incorrect_df["ground_truth"] - incorrect_df["prediction"]) == 1]
    add_case(adj_confusion, "Adjacent grade confusion", "mutual_information", ascending=False)
    
    # 9. Severe confusion (prediction off by >= 2 grades)
    severe_confusion = incorrect_df[abs(incorrect_df["ground_truth"] - incorrect_df["prediction"]) >= 2]
    add_case(severe_confusion, "Severe confusion (>= 2 grades)", "calib_confidence", ascending=False)
    
    # Merge and deduplicate
    if selected_dfs:
        selected_df = pd.concat(selected_dfs).drop_duplicates(subset=["image_id"])
    else:
        selected_df = pd.DataFrame()
        
    # If we need more cases to hit num_cases, fill with highest entropy/variance cases that aren't already selected
    remaining = num_cases - len(selected_df)
    if remaining > 0:
        unselected = df[~df["image_id"].isin(selected_df["image_id"])]
        if not unselected.empty:
            fill_df = unselected.sort_values("calib_entropy", ascending=False).head(remaining).copy()
            fill_df["selected_reason"] = "High uncertainty representative"
            selected_df = pd.concat([selected_df, fill_df])
            
    return selected_df.head(num_cases)
