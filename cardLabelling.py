import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, Button, Label, Entry
from PIL import Image, ImageTk
import os
import time
import yaml
from imageProcessing import preprocessFrame

def saveHsvValues():
  with open("config/hsv_values.yaml", "w") as file:
    hsv_values = {
      "LowH": lowH,
      "LowS": lowS,
      "LowV": lowV,
      "HighH": highH,
      "HighS": highS,
      "HighV": highV
    }
    yaml.dump(hsv_values, file)
  print("HSV values saved to config/hsv_values.yaml")

def updateHsv(dummy=None):
  global lowH, lowS, lowV, highH, highS, highV
  lowH = scaleLowH.get()
  lowS = scaleLowS.get()
  lowV = scaleLowV.get()
  highH = scaleHighH.get()
  highS = scaleHighS.get()
  highV = scaleHighV.get()

def toggleSaveImages():
  global isSaveImage, startTime
  isSaveImage = not isSaveImage
  startTime = time.time()
  print("Saving images:", "Enabled" if isSaveImage else "Disabled")

def processFrame():
  global lastSaveTime
  ret, frame = cap.read()
  if not ret:
    return

  frame, mask, warpeds, _ = preprocessFrame(frame, lowH, lowS, lowV, highH, highS, highV, minContourArea)
  for warped in warpeds:
    if isSaveImage:
      if (time.time() - startTime < captureDuration):
        currentTime = time.time()
        if currentTime - lastSaveTime >= (1 / captureFps):
          name = textField.get()
          os.makedirs(f"{outputDir}/{name}", exist_ok=True)
          timestamp = time.strftime("%Y%m%d-%H%M%S")
          filename = f"{outputDir}/{name}/{timestamp}.jpg"
          cv2.imwrite(filename, warped)
          print(f"{time.time() - startTime}. Saved {filename}")
          lastSaveTime = currentTime
      else:
        toggleSaveImages()
          
  displayImage(frame, originalLabel)
  displayImage(mask, maskLabel, isGray=True)

  window.after(10, processFrame)

def displayImage(img, label, isGray=False):
  if isGray:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

  imPil = Image.fromarray(img)
  imgtk = ImageTk.PhotoImage(image=imPil)
  label.imgtk = imgtk
  label.configure(image=imgtk)

if __name__ == "__main__":
  lowH, lowS, lowV = 44, 224, 0
  highH, highS, highV = 62, 255, 151
  minContourArea = 9000
  captureFps = 0.5
  captureDuration = 20
  isSaveImage = False
  startTime = 0
  outputDir = "dataset"
  lastSaveTime = 0

  if os.path.exists("config/hsv_values.yaml"):
    with open("config/hsv_values.yaml", "r") as file:
      hsv_values = yaml.safe_load(file)
      lowH = hsv_values.get("LowH", lowH)
      lowS = hsv_values.get("LowS", lowS)
      lowV = hsv_values.get("LowV", lowV)
      highH = hsv_values.get("HighH", highH)
      highS = hsv_values.get("HighS", highS)
      highV = hsv_values.get("HighV", highV)

  window = tk.Tk()
  window.title("HSV Threshold and Perspective Transform UI")

  cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

  lowFrame = tk.Frame(window)
  scaleLowH = Scale(lowFrame, from_=0, to=179, orient="horizontal", label="LowH", command=updateHsv)
  scaleLowH.set(lowH)
  scaleLowH.pack(side="left")
  scaleLowS = Scale(lowFrame, from_=0, to=255, orient="horizontal", label="LowS", command=updateHsv)
  scaleLowS.set(lowS)
  scaleLowS.pack(side="left")
  scaleLowV = Scale(lowFrame, from_=0, to=255, orient="horizontal", label="LowV", command=updateHsv)
  scaleLowV.set(lowV)
  scaleLowV.pack(side="left")
  lowFrame.grid(row=0, column=0)

  highFrame = tk.Frame(window)
  scaleHighH = Scale(highFrame, from_=0, to=179, orient="horizontal", label="HighH", command=updateHsv)
  scaleHighH.set(highH)
  scaleHighH.pack(side="left")
  scaleHighS = Scale(highFrame, from_=0, to=255, orient="horizontal", label="HighS", command=updateHsv)
  scaleHighS.set(highS)
  scaleHighS.pack(side="left")
  scaleHighV = Scale(highFrame, from_=0, to=255, orient="horizontal", label="HighV", command=updateHsv)
  scaleHighV.set(highV)
  scaleHighV.pack(side="left")
  highFrame.grid(row=1, column=0)

  originalLabel = Label(window)
  originalLabel.grid(row=3, column=1, rowspan=3)

  maskLabel = Label(window)
  maskLabel.grid(row=0, column=1, rowspan=3)

  Button(window, text="Save HSV Values", command=saveHsvValues).grid(row=3, column=0)

  textField = Entry(window)
  textField.grid(row=3, column=0, rowspan=2)

  Button(window, text="Save Images", command=toggleSaveImages).grid(row=5, column=0)

  processFrame()
  window.mainloop()
  cap.release()
  cv2.destroyAllWindows()
