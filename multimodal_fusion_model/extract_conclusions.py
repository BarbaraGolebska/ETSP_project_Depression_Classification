import pandas as pd

# Load CSV
csv_file = "multimodal_fusion_model\metrics_all.csv"  # replace with your CSV path


# Load CSV safely, handle quotes, skip malformed lines
df = pd.read_csv(csv_file, quotechar='"', on_bad_lines='skip')

# Convert numeric columns
numeric_cols = [
    'dev_auc', 'dev_youden', 'dev_precision', 'dev_recall', 'dev_f1',
    'test_auc', 'test_youden', 'test_precision', 'test_recall', 'test_f1'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing essential metrics
df = df.dropna(subset=['test_f1', 'test_auc'])

# Find best metrics
best_test_f1 = df['test_f1'].max()
best_test_auc = df['test_auc'].max()
best_test_youden = df['test_youden'].max()

best_f1_rows = df[df['test_f1'] == best_test_f1]
best_auc_rows = df[df['test_auc'] == best_test_auc]
best_youden_rows = df[df['test_youden'] == best_test_youden]

# Print best results
print("=== Best Test F1 ===")
print(best_f1_rows[['sampler', 'model_type', 'data', 'test_f1', 'test_auc', 'test_youden']])
print("\n=== Best Test AUC ===")
print(best_auc_rows[['sampler', 'model_type', 'data', 'test_f1', 'test_auc', 'test_youden']])
print("\n=== Best Test Youden ===")
print(best_youden_rows[['sampler', 'model_type', 'data', 'test_f1', 'test_auc', 'test_youden']])

# Simple conclusion
print("\n=== Conclusion ===")
if (best_f1_rows.iloc[0]['sampler'] == best_auc_rows.iloc[0]['sampler'] and
    best_f1_rows.iloc[0]['model_type'] == best_auc_rows.iloc[0]['model_type']):
    print(f"The best model is {best_f1_rows.iloc[0]['model_type']} with sampler {best_f1_rows.iloc[0]['sampler']}, "
          f"achieving test F1={best_test_f1:.3f} and test AUC={best_test_auc:.3f}.")
else:
    print("Different models achieved best F1 and best AUC:")
    print(f"- Best F1: {best_f1_rows.iloc[0]['model_type']} with sampler {best_f1_rows.iloc[0]['sampler']}, F1={best_test_f1:.3f}")
    print(f"- Best AUC: {best_auc_rows.iloc[0]['model_type']} with sampler {best_auc_rows.iloc[0]['sampler']}, AUC={best_test_auc:.3f}")
