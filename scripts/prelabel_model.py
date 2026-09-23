import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Stub for model-based pre-labeling.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--classes", default="configs/classes.yaml")
    args = parser.parse_args()

    print(f"Model pre-labeling stub called for session {args.session} with model {args.model}.")
    print("Implementation pending first training run.")

if __name__ == "__main__":
    main()
