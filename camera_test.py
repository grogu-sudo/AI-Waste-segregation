import cv2

for i in range(5):

    print(f"Testing camera index {i}")

    cap = cv2.VideoCapture(i)

    ret, frame = cap.read()

    if ret:

        print(f"Working camera at index {i}")

        cv2.imshow(f"Camera {i}", frame)

        cv2.waitKey(2000)

    cap.release()

cv2.destroyAllWindows()
