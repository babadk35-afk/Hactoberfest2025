"""
Webcam Motion Detector + Activity Plot
- Detects motion using frame differencing and logs activity timeline
Usage:
  python 19_motion_detector.py
Dependencies: opencv-python, numpy, matplotlib
"""
import cv2, numpy as np, time, matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
ret, prev = cap.read()
if not ret: raise SystemExit("No camera")
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
activity = []
t0 = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion = cv2.countNonZero(th)
        activity.append((time.time()-t0, motion))
        cv2.imshow("motion", th)
        prev_gray = gray
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release(); cv2.destroyAllWindows()
    import matplotlib.pyplot as plt
    ts, val = zip(*activity) if activity else ([], [])
    plt.figure(); plt.plot(ts, val); plt.title("Motion Activity"); plt.xlabel("seconds"); plt.ylabel("motion pixels"); plt.show()
