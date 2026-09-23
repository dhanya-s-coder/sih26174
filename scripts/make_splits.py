import argparse
import random
import yaml
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    sessions = [d.name for d in Path("data/frames").iterdir() if d.is_dir()]
    random.shuffle(sessions)

    n = len(sessions)
    test_n = max(1, int(n * args.test_ratio))
    val_n = max(1, int(n * args.val_ratio))
    
    test_sessions = sessions[:test_n]
    val_sessions = sessions[test_n:test_n+val_n]
    train_sessions = sessions[test_n+val_n:]
    
    if not train_sessions: # Fallback for very small datasets
        train_sessions = sessions
        test_sessions = sessions
        val_sessions = sessions

    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # We will just generate lists of images for each split, which YOLO accepts
    for split_name, split_sessions in zip(["train", "val", "test"], [train_sessions, val_sessions, test_sessions]):
        with open(splits_dir / f"{split_name}.txt", "w") as f:
            for s in split_sessions:
                s_dir = Path("data/frames") / s
                for img in s_dir.glob("*.jpg"):
                    # Exclude augmentations from val/test
                    if split_name != "train" and "_aug_" in img.name:
                        continue
                    f.write(f"{img.absolute()}\n")

    # Write ultralytics dataset.yaml
    with open("configs/classes.yaml", "r") as f:
        classes = yaml.safe_load(f)["classes"]
    
    names = {c["id"]: c["name"] for c in classes}
    
    dataset_yaml = {
        "path": str(splits_dir.absolute()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": names
    }
    
    with open(splits_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)
        
    print(f"Created splits: Train({len(train_sessions)}), Val({len(val_sessions)}), Test({len(test_sessions)})")
    print(f"Wrote {splits_dir / 'dataset.yaml'}")

if __name__ == "__main__":
    main()
