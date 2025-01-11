import cv2

# Load Haar Cascade file with the correct path
a = cv2.CascadeClassifier(r"C:\\Users\\MSii\\Desktop\\face_recog\\haarcascade_frontalface_default.xml")

# Check if the file was loaded successfully
if a.empty():
    print("Error: Haar Cascade file not loaded. Check the file path.")
    exit()

# Open webcam
b = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    c_rec, d_rec = b.read()

    # Convert frame to grayscale for better detection
    e = cv2.cvtColor(d_rec, cv2.COLOR_BGR2GRAY)

    # Detect faces
    f = a.detectMultiScale(e, 1.3, 6)

    # Draw rectangles around detected faces
    for (x1, y1, w1, h1) in f:
        cv2.rectangle(d_rec, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 5)

    # Display the frame
    cv2.imshow('img', d_rec)

    # Break loop on pressing 'q'
    h = cv2.waitKey(1)
    if h & 0xFF == ord('q'):
        break

# Release resources
b.release()
cv2.destroyAllWindows()
