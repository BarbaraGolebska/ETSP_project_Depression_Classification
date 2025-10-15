import gdown
import zipfile
import os
from tqdm import tqdm


FILE_ID = "1c4AERdj2tpIBQV_aLofSNwUyGzKqSCLY"
OUTPUT = "../data/raw/EDAIC_text.zip"
EXTRACT_DIR = "../data/raw"


def main():
    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)

    print("Downloading E-DAIC Text Only dataset from Google Drive...")
    gdown.download(id=FILE_ID, output=OUTPUT, quiet=False)

    print("Extracting files (flattening top-level folder if present)...")
    with zipfile.ZipFile(OUTPUT, "r") as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm(files, desc="Unzipping", unit="file"):
            # skip directories
            if file.endswith("/"):
                continue

            # remove top-level folder (if present)
            parts = file.split("/", 1)
            new_name = parts[1] if len(parts) > 1 else parts[0]
            target_path = os.path.join(EXTRACT_DIR, new_name)

            # if file already exists, skip it
            if os.path.exists(target_path):
                tqdm.write(f"Skipping existing file: {new_name}")
                continue

            # create subdirectories if needed
            os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

            # copy file directly
            with zip_ref.open(file) as source, open(target_path, "wb") as target:
                target.write(source.read())

    print(f"\nDone! Data extracted directly into '{EXTRACT_DIR}/'")


if __name__ == "__main__":
    main()