import cv2
import argparse
from pathlib import Path

def draw_yolo(img, txt_path, classes):
    h, w = img.shape[:2]
    if txt_path.exists():
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    name = classes[c] if c < len(classes) else str(c)
                    cv2.putText(img, name, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()

    frames_dir = Path("data/frames") / args.session
    labels_dir = Path("data/labels") / args.session
    
    import yaml
    with open("configs/classes.yaml", "r") as f:
        cls_config = yaml.safe_load(f)["classes"]
        classes = {c["id"]: c["name"] for c in cls_config}

    for img_path in frames_dir.glob("*.jpg"):
        txt_path = labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is not None:
            res = draw_yolo(img, txt_path, classes)
            cv2.imshow("Review", res)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
