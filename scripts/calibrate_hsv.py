import cv2
import numpy as np
import yaml
from pathlib import Path

def nothing(x):
    pass

def main():
    config_path = Path("configs/hsv_ranges.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            ranges = yaml.safe_load(f)
    else:
        ranges = {
            "yellow_box": {"lower": [20,100,100], "upper": [30,255,255]},
            "red_box": {"lower": [170,100,100], "upper": [180,255,255], "lower2": [0,100,100], "upper2": [10,255,255]}
        }

    cv2.namedWindow("Calibrate")
    
    cv2.createTrackbar("H_L", "Calibrate", 0, 179, nothing)
    cv2.createTrackbar("S_L", "Calibrate", 0, 255, nothing)
    cv2.createTrackbar("V_L", "Calibrate", 0, 255, nothing)
    cv2.createTrackbar("H_U", "Calibrate", 179, 179, nothing)
    cv2.createTrackbar("S_U", "Calibrate", 255, 255, nothing)
    cv2.createTrackbar("V_U", "Calibrate", 255, 255, nothing)

    print("Keys: [1] Load Yellow, [2] Load Red, [S] Save, [Q] Quit")
    
    cap = cv2.VideoCapture(0)
    current_color = "yellow_box"

    def load_trackbars(c):
        cv2.setTrackbarPos("H_L", "Calibrate", ranges[c]["lower"][0])
        cv2.setTrackbarPos("S_L", "Calibrate", ranges[c]["lower"][1])
        cv2.setTrackbarPos("V_L", "Calibrate", ranges[c]["lower"][2])
        cv2.setTrackbarPos("H_U", "Calibrate", ranges[c]["upper"][0])
        cv2.setTrackbarPos("S_U", "Calibrate", ranges[c]["upper"][1])
        cv2.setTrackbarPos("V_U", "Calibrate", ranges[c]["upper"][2])

    load_trackbars(current_color)

    while True:
        ret, frame = cap.read()
        if not ret: break

        hl = cv2.getTrackbarPos("H_L", "Calibrate")
        sl = cv2.getTrackbarPos("S_L", "Calibrate")
        vl = cv2.getTrackbarPos("V_L", "Calibrate")
        hu = cv2.getTrackbarPos("H_U", "Calibrate")
        su = cv2.getTrackbarPos("S_U", "Calibrate")
        vu = cv2.getTrackbarPos("V_U", "Calibrate")

        lower = np.array([hl, sl, vl])
        upper = np.array([hu, su, vu])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Display
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.putText(frame, f"Tuning: {current_color}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_color = "yellow_box"
            load_trackbars(current_color)
        elif key == ord('2'):
            current_color = "red_box"
            load_trackbars(current_color)
        elif key == ord('s'):
            ranges[current_color]["lower"] = [hl, sl, vl]
            ranges[current_color]["upper"] = [hu, su, vu]
            with open(config_path, "w") as f:
                yaml.dump(ranges, f)
            print(f"Saved {current_color} to {config_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
