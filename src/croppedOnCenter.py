import cv2 as cv

# --- User settings ---
CROP_W = 1500    # width of the cropped area
CROP_H = 1100    # height of the cropped area
CAM_W = 4656    # width of full webcam frame you want
CAM_H = 3496    # height of full webcam frame you want
# ----------------------

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam")

# Force the camera to capture at higher resolution
cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_H)

# Make a window that matches the crop size exactly
cv.namedWindow("Crop", cv.WINDOW_NORMAL)
cv.resizeWindow("Crop", CROP_W, CROP_H)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame grab failed")
        break

    h, w = frame.shape[:2]

    if CROP_W > w or CROP_H > h:
        print(f"❌ Crop size ({CROP_W}×{CROP_H}) is larger than camera frame ({w}×{h})")
        break

    # Center crop coordinates
    x0 = (w - CROP_W) // 2
    y0 = (h - CROP_H) // 2
    x1 = x0 + CROP_W
    y1 = y0 + CROP_H

    cropped = frame[y0:y1, x0:x1]

    cv.imshow("Crop", cropped)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
