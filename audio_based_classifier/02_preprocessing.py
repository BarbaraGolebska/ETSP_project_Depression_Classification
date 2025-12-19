import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split



def aggregate_features(df):
    # assume first column = timestamp
    X = df.iloc[:, 2:]
    summary = pd.Series({
        **{f"{col}_mean": X[col].mean() for col in X.columns},
        **{f"{col}_std": X[col].std() for col in X.columns},
    })
    return summary

def main():
    labels_df = pd.read_csv('data/raw/labels/detailed_labels.csv').rename(columns={"Participant": "participant_id", "Depression_label": "target_depr", "PTSD_label": "target_ptsd"})
    labels_df["participant_id"] = labels_df["participant_id"].astype(str)

    participant_list = labels_df["participant_id"].tolist()


    agg_features = []
    for pid in participant_list:
        print(f"Processing patient: {pid}")
        ftypes = {
            #"bow_egemaps": f"{pid}_BoAW_openSMILE_2.3.0_eGeMAPS.csv",
            #"bow_mfcc": f"{pid}_BoAW_openSMILE_2.3.0_MFCC.csv",
            #"densenet201": f"{pid}_densenet201.csv",
            "vgg16": f"{pid}_vgg16.csv",
            #"ek_egemaps":f"{pid}_OpenSMILE2.3.0_egemaps.csv",
            #"ek_mfcc":f"{pid}_OpenSMILE2.3.0_mfcc.csv"
        }

        patient_summary = []
        for ftype, path in ftypes.items():
            if ftype in ["ek_egemaps", "ek_mfcc"]:
                df = pd.read_csv(f"data/raw/features/{path}",  sep=';')

            elif ftype in ["densenet201", "vgg16"]:
                df = pd.read_csv(f"data/raw/features/{path}",  sep=',', decimal='.')
                

            else:
                df = pd.read_csv(f"data/raw/features/{path}", sep='[,;]', engine='python', header=None)
            agg = aggregate_features(df)
            agg.index = [f"{ftype}_{col}" for col in agg.index]
            patient_summary.append(agg)

        all_feats = pd.concat(patient_summary)
        all_feats["participant_id"] = pid
        agg_features.append(all_feats)

    df_features = pd.DataFrame(agg_features).set_index("participant_id")

    data = df_features.merge(labels_df[["participant_id", "target_depr", "target_ptsd", "split"]],
                                    on="participant_id", how="inner")
    
    data = data.set_index("participant_id")
    data.to_csv("data/processed/VGG16_aggregated_features.csv")

if __name__ == "__main__":
    main()