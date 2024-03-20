
import cv2 
from matplotlib import pyplot as plt
  

def face_cropper(img, depth_img):
    # Convert into grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Load the cascade 
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml') 
    # Detect faces 
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) 

    for (x, y, w, h) in faces: 
        # h = 370
        # w = 370
        img_cropped = img[y:y + h, x:x + w] 
        depth_cropped = depth_img[y:y + h, x:x + w]


    return img_cropped, depth_cropped 
      