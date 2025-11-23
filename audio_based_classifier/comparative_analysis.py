import pandas as pd
import numpy as np
import os

# --- Configuration ---
TREE_RESULTS_PATH = 'tree_results.csv'
BASELINE_RESULTS_PATH = 'baseline_results.csv'

def load_data(tree_path: str, baseline_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the tree and baseline results files into pandas DataFrames."""
    print("🤖 Loading data files...")
    
    if not os.path.exists(tree_path) or not os.path.exists(baseline_path):
        raise FileNotFoundError(f"One or both files not found. Check paths: {tree_path}, {baseline_path}")

    df_tree = pd.read_csv(tree_path)
    df_baseline = pd.read_csv(baseline_path)
    
    print(f"✅ Loaded {len(df_tree)} tree model results and {len(df_baseline)} baseline model results.")
    return df_tree, df_baseline

def inspect_and_describe(df_tree: pd.DataFrame, df_baseline: pd.DataFrame):
    """
    Prints initial inspection information, descriptive statistics, and best runs (by Youden Index).
    """
    
    print("\n" + "="*70)
    print("## 📊 Data Inspection and Descriptive Statistics (Youden Metric Focus)")
    print("="*70)

    # 1. Inspection and Structure
    for name, df in [("Tree Results", df_tree), ("Baseline Results", df_baseline)]:
        print(f"\n--- {name} ({len(df)} rows) ---")
        print("Data Types and Non-Null Counts:")
        df.info(verbose=False, memory_usage=False)
        print("\nFirst 5 Rows:")
        print(df.head())
    
    # 2. Descriptive Stats
    print("\n" + "-"*70)
    print("### Descriptive Stats for Key Metrics (Youden Index & AUC)")
    print("-" * 70)
    
    # Note: We still describe AUC for context, but focus on Youden for sorting later
    print("\n--- Tree Models ---")
    print(df_tree[['Youden_Index', 'AUC']].describe())

    print("\n--- Baseline Models ---")
    print(df_baseline[['Youden_Index', 'AUC']].describe())
    
    # 3. Best Runs (Sorted by Youden_Index)
    print("\n" + "-"*70)
    print("### Best Performing Runs (By Youden Index)")
    print("-" * 70)
    
    # ******* CRITICAL CHANGE: ID max by Youden_Index *******
    best_tree_youden = df_tree.loc[df_tree['Youden_Index'].idxmax()]
    best_baseline_youden = df_baseline.loc[df_baseline['Youden_Index'].idxmax()]

    print("\n🥇 Best Result in Tree Models:")
    print(best_tree_youden[['Model_Name', 'Data', 'Oversampler', 'Youden_Index', 'AUC', 'FN', 'TP']])

    print("\n🥇 Best Result in Baseline Models:")
    print(best_baseline_youden[['Model_Name', 'Data', 'Oversampler', 'Youden_Index', 'AUC', 'FN', 'TP']])

def comparative_analysis(df_tree: pd.DataFrame, df_baseline: pd.DataFrame):
    """
    Performs grouping analysis by categorical variables and prints the results,
    sorted by Mean Youden Index.
    """
    
    print("\n" + "="*70)
    print("## 📈 Performance Analysis (Mean Youden Index)")
    print("="*70)

    # ******* CRITICAL CHANGE: Group by Mean Youden_Index *******
    
    # 1. Mean Youden Index by Categorical Variables
    print("\n--- Analysis by Model Name (Tree Models) ---")
    tree_model_youden = df_tree.groupby('Model_Name')['Youden_Index'].mean().sort_values(ascending=False)
    print(tree_model_youden)

    print("\n--- Analysis by Feature Set (Tree Models) ---")
    tree_data_youden = df_tree.groupby('Data')['Youden_Index'].mean().sort_values(ascending=False)
    print(tree_data_youden)

    print("\n--- Analysis by Feature Set (Baseline Model) ---")
    bs_data_youden = df_baseline.groupby('Data')['Youden_Index'].mean().sort_values(ascending=False)
    print(bs_data_youden)

    print("\n--- Analysis by Oversampler (All Models) ---")
    # Combine data for a unified oversampler view
    df_combined = pd.concat([df_tree.assign(Category='Tree'), df_baseline.assign(Category='Baseline')])
    # Group by Youden Index and pivot for comparison
    oversampler_youden = df_combined.groupby(['Category', 'Oversampler'])['Youden_Index'].mean().unstack().T
    print(oversampler_youden.sort_values(by='Tree', ascending=False))
    
    print("\nAnalysis complete.")


def main():
    try:
        df_tree, df_baseline = load_data(TREE_RESULTS_PATH, BASELINE_RESULTS_PATH)
        inspect_and_describe(df_tree, df_baseline)
        comparative_analysis(df_tree, df_baseline)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please ensure your CSV files are in the same directory as this script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}")


if __name__ == "__main__":
    main()