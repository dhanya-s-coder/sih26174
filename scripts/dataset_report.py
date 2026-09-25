import argparse
from pathlib import Path
import yaml
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    splits_dir = Path("data/splits")
    
    with open("configs/classes.yaml", "r") as f:
        classes_config = yaml.safe_load(f)["classes"]
    class_names = {c["id"]: c["name"] for c in classes_config}

    for split in ["train", "val", "test"]:
        split_txt = splits_dir / f"{split}.txt"
        if not split_txt.exists(): continue
        
        class_counts = defaultdict(int)
        empty_imgs = 0
        degenerate = 0
        total_imgs = 0
        
        with open(split_txt, "r") as f:
            for line in f:
                img_path = Path(line.strip())
                total_imgs += 1
                txt_path = Path("data/labels") / img_path.parent.name / (img_path.stem + ".txt")
                
                if not txt_path.exists() or txt_path.stat().st_size == 0:
                    empty_imgs += 1
                    continue
                    
                with open(txt_path, "r") as lf:
                    for lline in lf:
                        parts = lline.strip().split()
                        if len(parts) >= 5:
                            c = int(parts[0])
                            _, _, w, h = map(float, parts[1:5])
                            if w <= 0 or h <= 0 or w > 1.0 or h > 1.0:
                                degenerate += 1
                            else:
                                class_counts[c] += 1
                                
        print(f"--- {split.upper()} SPLIT ---")
        print(f"Total Images: {total_imgs}")
        print(f"Empty Labels: {empty_imgs}")
        print(f"Degenerate/OOB Boxes: {degenerate}")
        print("Class Balance:")
        for cid, name in class_names.items():
            count = class_counts[cid]
            print(f"  {name} (id:{cid}): {count}")
            if count < 50 and split == "train":
                print(f"    WARNING: Low representation for {name}!")
        print()

if __name__ == "__main__":
    main()
