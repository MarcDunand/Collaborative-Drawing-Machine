import cv2

# Open the camera device (0 is usually the default camera)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the desired resolution (Width and Height)
desired_width = 4600 
desired_height = 3400

cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Set the desired frame rate (FPS)
desired_fps = 1  # Example frame rate

cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Retrieve and display the actual settings
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Resolution set to: {actual_width} x {actual_height}")
print(f"Frame rate set to: {actual_fps} FPS")

# Capture frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    cv2.imshow('Camera Output', frame)

    #press 's' to take picture
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('./captured_photos/high_resolution_image.png', frame)
        print("Image saved successfully.")



# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
