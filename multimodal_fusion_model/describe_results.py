import pandas as pd
import ast

def summarize_group(df, group_cols=["Model_Name", "Data"]):
    summaries = []
    grouped = df.groupby(group_cols)

    for group_key, group_df in grouped:
        model_name, data = group_key
        summaries.append(f"\n=== Model: {model_name} | Data: {data} ===")

        # Best rows by metric
        best_auc = group_df.loc[group_df['AUC'].idxmax()]
        best_youden = group_df.loc[group_df['Youden_Index'].idxmax()]

        summaries.append(f"\n• Best AUC:")
        summaries.append(
            f"  - Sampler: {best_auc['Sampler']}\n"
            f"  - Reduction Method: {best_auc['Reduction_Method']}\n"
            f"  - AUC: {best_auc['AUC']:.4f}\n"
            f"  - Youden: {best_auc['Youden_Index']:.4f}\n"
            f"  - TP={best_auc['TP']}, FP={best_auc['FP']}, TN={best_auc['TN']}, FN={best_auc['FN']}\n"
            f"  - False negatives: {best_auc['False_Negative_IDs']}\n"
            f"  - Best params: {best_auc['best_params']}"
        )

        summaries.append(f"\n• Best Youden Index:")
        summaries.append(
            f"  - Sampler: {best_youden['Sampler']}\n"
            f"  - Reduction Method: {best_youden['Reduction_Method']}\n"
            f"  - Youden: {best_youden['Youden_Index']:.4f}\n"
            f"  - AUC: {best_youden['AUC']:.4f}\n"
            f"  - TP={best_youden['TP']}, FP={best_youden['FP']}, TN={best_youden['TN']}, FN={best_youden['FN']}\n"
            f"  - False negatives: {best_youden['False_Negative_IDs']}\n"
            f"  - Best params: {best_youden['best_params']}"
        )

        # False-negative analysis
        all_fn_ids = []
        for ids in group_df["False_Negative_IDs"]:
            try:
                parsed = ast.literal_eval(ids)
                all_fn_ids.extend(parsed)
            except:
                pass

        if all_fn_ids:
            counts = pd.Series(all_fn_ids).value_counts()
            summaries.append("\n• Most frequent false-negative IDs:")
            for idx, cnt in counts.items():
                summaries.append(f"  - ID {idx}: {cnt} times")

    return "\n".join(summaries)


if __name__ == "__main__":
    # Load CSV
    df = pd.read_csv("EF_MLP_dev.csv")

    # Summarize
    output = summarize_group(df)

    # Print results
    print(output)
