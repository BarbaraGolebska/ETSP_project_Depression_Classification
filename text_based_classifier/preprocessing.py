import re
from pathlib import Path
import pandas as pd


TRANSCRIPTS_DIR = Path("../daic_data/EDAIC_text/transcripts")
CONFIDENCE_THRESHOLD = 0.6  # the minimum value of confidence to consider a certain transcription


def main():

    # initialize a list of all documents (document = all participant's speech)
    docs = []

    # iterate through every csv file in the directory
    for file in sorted(TRANSCRIPTS_DIR.glob("*.csv")):

        # extract participant's ID
        pid = re.search(r"(\d+)", file.name).group(1)

        df = pd.read_csv(file)

        # filter low-confidence lines
        df = df[df["Confidence"] >= CONFIDENCE_THRESHOLD]

        # get one document (all phrases combined) per participant
        doc = " ".join(df["Text"].str.strip().tolist())

        # add to the list of documents
        docs.append({"participant_id": pid, "text": doc})

    # get a final df with participant_id and his/her corresponding text
    participants_df = pd.DataFrame(docs)

    # get labels
    labels_df = pd.read_csv('../daic_data/EDAIC_text/labels/detailed_lables.csv').rename(
        columns={"Participant": "participant_id", "Depression_label": "target_depr", "PTSD_label": "target_ptsd"})

    # ensure "participant_id" is string in both dfs
    participants_df["participant_id"] = participants_df["participant_id"].astype(str)
    labels_df["participant_id"] = labels_df["participant_id"].astype(str)

    # merge two dfs
    data = participants_df.merge(labels_df[["participant_id", "target_depr", "target_ptsd", "split"]],
                                 on="participant_id", how="inner")

    # save the data
    output_dir = Path("../daic_data/processed")
    output_dir.mkdir(exist_ok=True)
    data.to_csv(output_dir / "combined.csv")


if __name__ == "__main__":
    main()