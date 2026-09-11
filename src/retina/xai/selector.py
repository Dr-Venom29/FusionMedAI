import pandas as pd
import numpy as np

def select_representative_cases(results_df, num_images=25):
    """
    Select representative cases based on scientific criteria.
    Returns a DataFrame containing only the selected cases and a 'selected_reason' column.
    """
    selected = []
    
    # Helpers
    correct = results_df[results_df['is_correct']]
    incorrect = results_df[~results_df['is_correct']]
    
    def add_case(df, reason, n=1):
        if not df.empty:
            if n == 1:
                sample = df.head(1).copy()
            else:
                samples = []
                if len(df) > 0:
                    samples.append(df.iloc[0:1])  # Top
                if len(df) > 2:
                    samples.append(df.iloc[len(df)//2:len(df)//2+1])  # Median
                if len(df) > 1:
                    k = max(2, int(len(df) * 0.1))
                    samples.append(df.iloc[:k].sample(1, random_state=42))  # Random among top-k
                
                sample = pd.concat(samples).drop_duplicates(subset=['image_id']).head(n).copy()
                
            sample['selected_reason'] = reason
            selected.append(sample)
            
    # 1. Most confident correct
    add_case(correct.sort_values('confidence', ascending=False), "Most confident correct")
    
    # 2. Least confident correct
    add_case(correct.sort_values('confidence', ascending=True), "Least confident correct")
    
    # 3. Most confident incorrect
    add_case(incorrect.sort_values('confidence', ascending=False), "Most confident incorrect")
    
    # 4. Least confident incorrect
    add_case(incorrect.sort_values('confidence', ascending=True), "Least confident incorrect")
    
    # 5. Highest entropy prediction
    add_case(results_df.sort_values('entropy', ascending=False), "Highest entropy prediction")
    
    # 6. Lowest entropy prediction
    add_case(results_df.sort_values('entropy', ascending=True), "Lowest entropy prediction")
    
    # 7. One representative correct image from every DR grade
    for grade in sorted(results_df['ground_truth'].unique()):
        subset = correct[correct['ground_truth'] == grade]
        add_case(subset.sort_values('confidence', ascending=False), f"Representative correct (Grade {grade})")
        
    # 8. One failure example from every DR grade
    for grade in sorted(results_df['ground_truth'].unique()):
        subset = incorrect[incorrect['ground_truth'] == grade]
        add_case(subset.sort_values('confidence', ascending=False), f"Failure example (Grade {grade})")
        
    # 9. One confusion example (prediction off by >= 2 grades)
    severe_errors = incorrect[abs(incorrect['ground_truth_idx'] - incorrect['prediction_idx']) >= 2]
    add_case(severe_errors.sort_values('confidence', ascending=False), "Severe confusion example (>1 grade error)")
    
    # 10. Fill the rest randomly to meet num_images
    if selected:
        selected_df = pd.concat(selected).drop_duplicates(subset=['image_id'])
    else:
        selected_df = pd.DataFrame()
        
    remaining = num_images - len(selected_df)
    if remaining > 0:
        unselected = results_df[~results_df['image_id'].isin(selected_df['image_id'])]
        if not unselected.empty:
            fill_sample = unselected.sample(n=min(remaining, len(unselected)), random_state=42).copy()
            fill_sample['selected_reason'] = "Random representative fill"
            selected_df = pd.concat([selected_df, fill_sample])
            
    return selected_df.head(num_images)
