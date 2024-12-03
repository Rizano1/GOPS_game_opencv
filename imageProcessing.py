import cv2
import numpy as np

def preprocessFrame(frame, lowH, lowS, lowV, highH, highS, highV, minContourArea):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  lowerColor = np.array([lowH, lowS, lowV])
  upperColor = np.array([highH, highS, highV])

  mask = cv2.inRange(hsv, lowerColor, upperColor)
  mask = cv2.bitwise_not(mask)
  kernel = np.ones((3, 3), np.uint8)
  mask = cv2.erode(mask, kernel, iterations=3)
  mask = cv2.dilate(mask, kernel, iterations=3)
  mask = cv2.GaussianBlur(mask, (5, 5), 0)
  warpeds = []
  boundings = []
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for contour in contours:
    if cv2.contourArea(contour) < minContourArea:
      continue

    boundings.append(cv2.boundingRect(contour))
    x, y, w, h = boundings[-1]
    
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
      for point in approx:
        cv2.circle(frame, tuple(point[0]), 5, (0, 255, 0), -1)

      corners = np.array([point[0] for point in approx], dtype="float32")
      sumCorners = corners.sum(axis=1)
      diffCorners = np.diff(corners, axis=1)

      orderedCorners = np.zeros((4, 2), dtype="float32")
      orderedCorners[0] = corners[np.argmin(sumCorners)]
      orderedCorners[1] = corners[np.argmin(diffCorners)]
      orderedCorners[2] = corners[np.argmax(sumCorners)]
      orderedCorners[3] = corners[np.argmax(diffCorners)]

      width, height = 200, 300
      destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
      transformMatrix = cv2.getPerspectiveTransform(orderedCorners, destination)
      warpeds.append( cv2.warpPerspective(frame, transformMatrix, (width, height)))
  return frame, mask, warpeds, boundings