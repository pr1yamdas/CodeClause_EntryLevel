import cv2

def detect_faces_live():
    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam successfully opened.")

    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces_detected = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the faces
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Start live face detection
if __name__ == "__main__":
    print("Starting live face detection...")
    detect_faces_live()
    print("Live face detection stopped.")
