import cv2

for i in range(10):  # Test indices from 0 to 9
  cap = cv2.VideoCapture(i,  cv2.CAP_DSHOW)
  if cap.isOpened():
    print(f"Press any key to close preview of index {i}")
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      cv2.imshow(f"Camera Index {i}", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()
