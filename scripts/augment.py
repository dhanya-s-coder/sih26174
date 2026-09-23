import cv2
import numpy as np
import albumentations as A
import argparse
from pathlib import Path
import random

def load_yolo_labels(txt_path):
    labels = []
    if txt_path.exists():
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(x) for x in parts[1:5]] + [int(parts[0])]) # x, y, w, h, class
    return labels

def save_yolo_labels(txt_path, labels):
    with open(txt_path, "w") as f:
        for lbl in labels:
            f.write(f"{int(lbl[4])} {lbl[0]:.6f} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f}\n")

def apply_copy_paste(img, bboxes, backgrounds_dir):
    """Simple synthetic copy-paste. Cuts out the objects and pastes on a random bg."""
    if len(bboxes) == 0 or not backgrounds_dir.exists():
        return img, bboxes
        
    bgs = list(backgrounds_dir.glob("*.jpg"))
    if not bgs:
        return img, bboxes
        
    bg_img = cv2.imread(str(random.choice(bgs)))
    bg_img = cv2.resize(bg_img, (img.shape[1], img.shape[0]))
    
    # Just a naive bounding box cut and paste
    out_img = bg_img.copy()
    h, w = img.shape[:2]
    
    new_bboxes = []
    for bbox in bboxes:
        cx, cy, bw, bh, cls = bbox
        x1 = max(0, int((cx - bw/2) * w))
        y1 = max(0, int((cy - bh/2) * h))
        x2 = min(w, int((cx + bw/2) * w))
        y2 = min(h, int((cy + bh/2) * h))
        
        obj_crop = img[y1:y2, x1:x2]
        if obj_crop.size == 0: continue
        
        # Paste at random location
        paste_x = random.randint(0, w - (x2-x1))
        paste_y = random.randint(0, h - (y2-y1))
        
        out_img[paste_y:paste_y+(y2-y1), paste_x:paste_x+(x2-x1)] = obj_crop
        
        ncx = (paste_x + (x2-x1)/2.0) / w
        ncy = (paste_y + (y2-y1)/2.0) / h
        new_bboxes.append([ncx, ncy, bw, bh, cls])
        
    return out_img, new_bboxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--multiplier", type=int, default=3)
    parser.add_argument("--bg_dir", default="", help="Directory with background images for copy-paste")
    args = parser.parse_args()

    frames_dir = Path("data/frames") / args.session
    labels_dir = Path("data/labels") / args.session

    # Geometric: full rotation, flips
    # Photometric: brightness, contrast, blur, noise, shadow. 
    # Hue shift is 0 because we rely heavily on color for "red_box" and "yellow_box".
    transform = A.Compose([
        A.Rotate(limit=180, p=0.8), # Albumentations Rotate range is [-limit, limit], so 180 means full 360
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5), # NO HUE SHIFT!
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=[]))

    for img_path in list(frames_dir.glob("*.jpg")):
        txt_path = labels_dir / (img_path.stem + ".txt")
        if not txt_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        bboxes = load_yolo_labels(txt_path)
        if not bboxes: continue
        
        for i in range(args.multiplier):
            try:
                # 20% chance to do copy-paste if bg_dir is valid
                do_cp = args.bg_dir and random.random() < 0.2
                if do_cp:
                    aug_img, aug_bboxes = apply_copy_paste(img, bboxes, Path(args.bg_dir))
                else:
                    transformed = transform(image=img, bboxes=bboxes)
                    aug_img = transformed['image']
                    aug_bboxes = transformed['bboxes']

                if len(aug_bboxes) > 0:
                    out_img_path = frames_dir / f"{img_path.stem}_aug_{i}.jpg"
                    out_txt_path = labels_dir / f"{img_path.stem}_aug_{i}.txt"
                    cv2.imwrite(str(out_img_path), aug_img)
                    save_yolo_labels(out_txt_path, aug_bboxes)
            except ValueError:
                # Bbox outside image due to rotation can cause ValueError in albumentations
                pass
                
    print(f"Augmentation complete for {args.session}.")

if __name__ == "__main__":
    main()
