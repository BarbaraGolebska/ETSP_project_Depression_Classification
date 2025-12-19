import pandas as pd
import ast

def summarize_test_results(df, group_cols=["Model_Name", "Data"]):
    summaries = []
    
    # Filter for rows that actually have test results
    if 'test_auc' in df.columns:
        df = df.dropna(subset=['test_auc'])
    else:
        return "Error: No 'test_auc' column found. Did you run the evaluation script?"

    grouped = df.groupby(group_cols)

    for group_key, group_df in grouped:
        model_name, data = group_key
        summaries.append(f"\n==========================================")
        summaries.append(f" MODEL: {model_name}  |  DATA: {data}")
        summaries.append(f"==========================================")

        # 1. Find the Best Performance on TEST Set (by Test AUC)
        # We look for the max 'test_auc' to see the best generalization
        #best_row = group_df.loc[group_df['test_auc'].idxmax()]
        best_row = group_df.loc[group_df['test_youden'].idxmax()]

        summaries.append(f"• Best Test Configuration (by Youden):")
        summaries.append(f"  - Oversampler: {best_row['Oversampler']}")
        summaries.append(f"  - Test AUC:    {best_row['test_auc']:.4f}")
        summaries.append(f"  - Test Youden: {best_row['test_youden']:.4f}")
        summaries.append(f"  - Precision:   {best_row['test_precision']:.4f}")
        summaries.append(f"  - Recall:      {best_row['test_recall']:.4f} (Sensitivity)")
        summaries.append(f"  - F1 Score:    {best_row['test_f1']:.4f}")
        summaries.append(f"  - Confusion Matrix: TP={int(best_row['test_TP'])}, FP={int(best_row['test_FP'])}, TN={int(best_row['test_TN'])}, FN={int(best_row['test_FN'])}")
        
        # 2. Compare with Dev Performance (Sanity Check)
        gap = best_row['AUC'] - best_row['test_auc']
        summaries.append(f"  - Generalization Gap: {gap:.4f} (Dev AUC - Test AUC)")
        if gap > 0.1:
            summaries.append("    ⚠️ HIGH OVERFITTING DETECTED")
        elif gap < -0.05:
            summaries.append("    ❓ Test score surprisingly higher than Dev")
            
        summaries.append(f"  - Best Params Used: {best_row['best_params']}")

        # 3. False Negative Analysis (Who did we miss in the TEST set?)
        # Note: Your CSV saving logic needs to capture 'test_false_negatives' to report this.
        # If you only have dev false negatives saved, we can't analyze test errors specifically here.

    return "\n".join(summaries)

if __name__ == "__main__":
    # Load the results file you updated in the previous step

    path="multimodal_fusion_model\EF_metrics_tree_models.csv"
    #path="early_fusion_results.csv"

    df = pd.read_csv(path) 

    output = summarize_test_results(df)
    print(output)