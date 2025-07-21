import os
import re
from tokenizers import ByteLevelBPETokenizer

RAW_DIR    = "data/raw"
PROC_DIR   = "data/processed"
VOCAB_DIR  = "data/tokenizer"

def clean_code(code: str) -> str:
    # strip docstrings
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    # strip single-line comments
    code = re.sub(r"#.*", "", code)
    return code

def train_tokenizer(files: list[str]):
    tokenizer = ByteLevelBPETokenizer()
    valid_files = []
    for f in files:
        try:
            # quick test open
            with open(f, "rb") as fh:
                fh.read(1)
            valid_files.append(f)
        except OSError:
            print(f"⚠️ Skipping {f} (AV/permission error)")
    tokenizer.train(
        valid_files,
        vocab_size=50_000,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    )

def encode_and_save(tokenizer: ByteLevelBPETokenizer, filepath: str):
    rel_path = os.path.relpath(filepath, RAW_DIR)
    out_path = os.path.join(PROC_DIR, rel_path + ".ids")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        code = clean_code(f.read())
    tokens = tokenizer.encode(code).ids
    # save as space-separated ints, or torch.save(tokens, out_path), etc.
    with open(out_path, "w") as out:
        out.write(" ".join(map(str, tokens)))

def main():
    # 1) gather all .py file paths
    py_files = []
    for root, _, files in os.walk(RAW_DIR):
        for fn in files:
            if fn.endswith(".py"):
                py_files.append(os.path.join(root, fn))

    # 2) train tokenizer on a subset or all
    print("Training tokenizer...")
    train_tokenizer(py_files)

    # 3) load tokenizer
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(VOCAB_DIR, "vocab.json"),
        os.path.join(VOCAB_DIR, "merges.txt"),
    )

    # 4) encode every file
    print("Encoding files…")
    for path in py_files:
        encode_and_save(tokenizer, path)

if __name__ == "__main__":
    main()
