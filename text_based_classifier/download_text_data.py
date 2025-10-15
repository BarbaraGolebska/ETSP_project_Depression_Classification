import gdown
import zipfile
import os
from tqdm import tqdm


FILE_ID = "1VEiTak0u56HjWthdwG2CMNnRQoFWzwP_"
OUTPUT = "../daic_data/edaic.zip"
EXTRACT_DIR = "../daic_data"


def main():
    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)

    print("Downloading E-DAIC Text Only dataset from Google Drive...")
    gdown.download(id=FILE_ID, output=OUTPUT, quiet=False)

    print("Extracting files...")
    with zipfile.ZipFile(OUTPUT, "r") as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm(files, desc="Unzipping", unit="file"):
            zip_ref.extract(member=file, path=EXTRACT_DIR)

    print(f"Done! Data extracted to '{EXTRACT_DIR}/'")


if __name__ == "__main__":
    main()