import pandas as pd

def summarize_test_results(df, group_cols=["model_type", "data"]):
    summaries = []

    # Ensure test_auc column exists
    if 'test_auc' not in df.columns:
        return "Error: No 'test_auc' column found. Did you run the evaluation script?"

    # Drop rows without test_auc
    df = df.dropna(subset=['test_auc'])

    grouped = df.groupby(group_cols)

    for group_key, group_df in grouped:
        model_name, data = group_key
        summaries.append("\n==========================================")
        summaries.append(f" MODEL: {model_name}  |  DATA: {data}")
        summaries.append("==========================================")

        # Find the best row by test_youden
        best_row = group_df.loc[group_df['test_youden'].idxmax()]

        summaries.append("• Best Test Configuration (by Youden):")
        summaries.append(f"  - Sampler: {best_row['sampler']}")
        summaries.append(f"  - Best Threshold: {best_row['best_threshold']:.4f}")
        summaries.append(f"  - Test AUC:    {best_row['test_auc']:.4f}")
        summaries.append(f"  - Test Youden: {best_row['test_youden']:.4f}")
        summaries.append(f"  - Precision:   {best_row['test_precision']:.4f}")
        summaries.append(f"  - Recall:      {best_row['test_recall']:.4f} (Sensitivity)")
        summaries.append(f"  - F1 Score:    {best_row['test_f1']:.4f}")
        summaries.append(f"  - Confusion Matrix: TP={int(best_row['test_tp'])}, FP={int(best_row['test_fp'])}, TN={int(best_row['test_tn'])}, FN={int(best_row['test_fn'])}")

        # Compare with Dev Performance (sanity check)
        gap = best_row['dev_auc'] - best_row['test_auc']
        summaries.append(f"  - Generalization Gap: {gap:.4f} (Dev AUC - Test AUC)")
        if gap > 0.1:
            summaries.append("    ⚠️ HIGH OVERFITTING DETECTED")
        elif gap < -0.05:
            summaries.append("    ❓ Test score surprisingly higher than Dev")

        # Add best parameters used
        param_cols = [col for col in df.columns if col.startswith("param_")]
        best_params = {col.replace("param_", ""): best_row[col] for col in param_cols}
        summaries.append(f"  - Best Params Used: {best_params}")

    return "\n".join(summaries)


if __name__ == "__main__":
    path="multimodal_fusion_model\EF_metrics_tree_models.csv"  # update with your actual path



    
    df = pd.read_csv(path)

    output = summarize_test_results(df)
    print(output)
