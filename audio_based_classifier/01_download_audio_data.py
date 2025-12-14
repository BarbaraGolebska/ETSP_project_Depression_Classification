import os
import zipfile
import gdown
from tqdm import tqdm


# ----------------------- CONFIG --------------------------

# ZIP 1 — Hubert embeddings
ZIP1_ID = "1JoZxsUEdqIT3K6u7eD8vwcatMqO_GXOA"
ZIP1_OUTPUT = "data/raw/features/hubert_embeddings.zip"
ZIP1_EXTRACT_DIR = "data/raw/features"

# ZIP 2 — E-DAIC Audio dataset
ZIP2_ID = "1NS2v_9UtYNGGFOS8tn9NRufPhBKNqZPm"
ZIP2_OUTPUT = "data/raw/EDAIC_audio.zip"
ZIP2_EXTRACT_DIR = "data/raw"

# ----------------------------------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def download_zip(file_id: str, output_path: str):
    print(f"\nDownloading ZIP from Google Drive (ID={file_id})...")
    ensure_dir(os.path.dirname(output_path))

    gdown.download(id=file_id, output=output_path, quiet=False, resume=True)
    print(f"[INFO] Download finished -> {output_path}")


def extract_zip_flat(zip_path: str, extract_dir: str):
    print(f"\nExtracting and flattening ZIP into: {extract_dir}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        files = zip_ref.namelist()

        for file in tqdm(files, desc="Unzipping", unit="file"):
            if file.endswith("/"):  # skip folders
                continue

            parts = file.split("/", 1)
            new_name = parts[1] if len(parts) > 1 else parts[0]
            target_path = os.path.join(extract_dir, new_name)

            if os.path.exists(target_path):
                tqdm.write(f"Skipping existing file: {new_name}")
                continue

            ensure_dir(os.path.dirname(target_path))

            with zip_ref.open(file) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

    print(f"[INFO] Extraction complete → {extract_dir}")


def main():

    download_zip(ZIP1_ID, ZIP1_OUTPUT)
    download_zip(ZIP2_ID, ZIP2_OUTPUT)

    extract_zip_flat(ZIP1_OUTPUT, ZIP1_EXTRACT_DIR)
    extract_zip_flat(ZIP2_OUTPUT, ZIP2_EXTRACT_DIR)

    print("\nAll downloads and extractions finished successfully!")


if __name__ == "__main__":
    main()
