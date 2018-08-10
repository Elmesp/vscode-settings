import cv2

head_profile_classifier = cv2.CascadeClassifier("lbpcascade_profileface.xml")

cap = cv2.VideoCapture(0)

while 1:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    both_img = cv2.flip(gray, 1)

    profile_right = head_profile_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)
    profile_left = head_profile_classifier.detectMultiScale(both_img, scaleFactor=1.1, minNeighbors=15)

    if len(profile_right) > 0:
        cv2.putText(frame, "DERECHA", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    if len(profile_left) > 0:
        cv2.putText(frame, "IZQUIERDA", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # Display the resulting frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# video.release()
cv2.destroyAllWindows()
