from pathlib import Path
import nltk

NLTK_DIR = Path("../data/nltk_data")
NLTK_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path.append(str(NLTK_DIR))

for resource in ["punkt", "stopwords", "wordnet"]:
    nltk.download(resource, download_dir=str(NLTK_DIR))

print(f"NLTK data downloaded to {NLTK_DIR.resolve()}")