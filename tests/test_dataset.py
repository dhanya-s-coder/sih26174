import os
import cv2
import numpy as np
from scripts.extract_frames import dhash, variance_of_laplacian
from scripts.augment import load_yolo_labels, save_yolo_labels
import albumentations as A

def test_dhash():
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    assert bin(dhash(img1) ^ dhash(img2)).count('1') == 0
    
    img2[50:60, 50:60] = 255
    assert bin(dhash(img1) ^ dhash(img2)).count('1') > 0

def test_variance_of_laplacian():
    img_sharp = np.zeros((100, 100, 3), dtype=np.uint8)
    img_sharp[50:60, 50:60] = 255
    
    img_blur = cv2.GaussianBlur(img_sharp, (15, 15), 0)
    
    assert variance_of_laplacian(img_sharp) > variance_of_laplacian(img_blur)

def test_yolo_roundtrip(tmp_path):
    txt_path = tmp_path / "test.txt"
    labels = [[0.5, 0.5, 0.2, 0.2, 1], [0.1, 0.2, 0.05, 0.05, 0]]
    save_yolo_labels(txt_path, labels)
    
    loaded = load_yolo_labels(txt_path)
    assert len(loaded) == 2
    assert abs(loaded[0][0] - 0.5) < 1e-5
    assert int(loaded[1][4]) == 0

def test_rotated_bbox():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Box in center
    bboxes = [[0.5, 0.5, 0.2, 0.2, 0]]
    
    transform = A.Compose([A.Rotate(limit=[90, 90], p=1.0)], bbox_params=A.BboxParams(format='yolo'))
    res = transform(image=img, bboxes=bboxes)
    
    # A center box rotated 90 deg around center should have same center, w/h swapped. 
    # Since w=h=0.2, it should be identical.
    out_box = res['bboxes'][0]
    assert abs(out_box[0] - 0.5) < 1e-5
    assert abs(out_box[1] - 0.5) < 1e-5
    
    # Top left box
    bboxes = [[0.2, 0.2, 0.1, 0.1, 0]]
    res = transform(image=img, bboxes=bboxes)
    out_box = res['bboxes'][0]
    # Albumentations 90 deg rotation results in (0.2, 0.8) for (0.2, 0.2)
    assert abs(out_box[0] - 0.2) < 1e-5
    assert abs(out_box[1] - 0.8) < 1e-5
