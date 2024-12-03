import cv2
import numpy as np
import tensorflow as tf
import os
import yaml
from imageProcessing import preprocessFrame

# def imageProcessing():
  

def loadHsvValues(filePath):
  if os.path.exists(filePath):
    with open(filePath, 'r') as f:
      hsvValues = yaml.safe_load(f)
    return (hsvValues['LowH'], hsvValues['LowS'], hsvValues['LowV'],
            hsvValues['HighH'], hsvValues['HighS'], hsvValues['HighV'])
  return 0, 100, 100, 10, 255, 255

def loadClassMapping(mapping_file):
  if os.path.exists(mapping_file):
    with open(mapping_file, 'r') as f:
      class_indices = yaml.safe_load(f)
      class_names = {v: k for k, v in class_indices.items()}  
      print(class_names)
    return class_names
  else:
    print(f"Mapping file '{mapping_file}' not found.")
    return None

if __name__ == "__main__":
  hsvFilePath = 'config/hsv_values.yaml'
  lowH, lowS, lowV, highH, highS, highV = loadHsvValues(hsvFilePath)

  minContourArea = 9000
  classMappingFile = 'model/class_mapping.yaml'
  classNames = loadClassMapping(classMappingFile)
  print(classNames)

  model = tf.keras.models.load_model('model/cnn_model.h5')
  cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)


  while True:
    ret, frame = cap.read()
    if not ret:
      break
    widht = frame.shape[1]
    height = frame.shape[0]
    
    frame, mask, warpeds, boundings = preprocessFrame(frame, lowH, lowS, lowV, highH, highS, highV, minContourArea)
    if len(warpeds) > 0:
      warpeds_resized = [cv2.resize(warped, (200, 200)) for warped in warpeds]
      grid = cv2.hconcat(warpeds_resized)  
      cv2.imshow('Warped', grid)

    for warped, bounding in zip(warpeds, boundings):
        x, y, _, _ = bounding
        warped = warped.astype(np.float32) / 255.0
        
        prediction = model.predict(warped[np.newaxis, ...], verbose=0)
        predictedClass = np.argmax(prediction)
        className = classNames.get(predictedClass, "Unknown") if classNames else "Unknown"
        
        if x >= int(widht / 3) and y <= (height/2):
          p1Card = className 
          
        if x >= int(widht / 3) and y >= (height/2):
          p2Card = className 
          
        if x <= int(widht / 3):
          deck = className 

        cv2.putText(frame, f'Class: {className}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    cv2.line(frame, (int(widht / 3), 0), (int(widht / 3), height), (0, 0, 255), 5)
    cv2.putText(frame, 'P1', (widht - 50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, 'P2', (widht - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, 'DECK', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
  
    cv2.line(frame, (int(widht / 3), int(height / 2)), (widht, int(height / 2)), (0, 0, 255), 5)
    # print("p1 Card : ", p1Card)
    # print("p2 Card : ", p2Card)
    # print("deck : ",deck)
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
