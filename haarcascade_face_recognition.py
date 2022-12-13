'''
Haarcascade Face Recognition
Code by:
Indira Puspita Margariza 

This code is used to detect face within image and then extract the cropped face.
This code can also detect eyes, but the accuracy is not that good.
To be able to use this result of face detection for face verification usage it needs face embeddings extractor model to compare 2 images. 

Pretrained haarcascade classifier from OpenCV is used here.
'''
import cv2
import numpy as np
import glob
import argparse

# construct the argument parser and parse the arguments
arg = argparse.ArgumentParser()
arg.add_argument("-f", "--folder_path", type=str, default="classifier/", help="path to pre-trained classifier folder contains .xml files")
arg.add_argument("-i", "--image_folder", type=str, default="images/", help="path to images folder")
arg.add_argument("-r", "--image_result_folder", type=str, default=None, help="path to save images with bounding boxes")
arg.add_argument("-c", "--crop_faces_folder", type=str, default="face-result/", help="path to save cropped face")
arg.add_argument("--eye_detect", default=True, help="whether to do eye detection or not")

args = vars(arg.parse_args())

#Initializing pretrained classifier for detection particular object paths
#In this face detection case, we will need a classifier for detecting face, eyes, and lips.
folder_path = args["folder_path"]
detector_files = {
	"face": "haarcascade_frontalface_default.xml",
	"eyes": "haarcascade_eye.xml",
}
#Initializing sample images folder
#There are several images used, those contain with single face or multiface.
image_paths = sorted(glob.glob(args["image_folder"] + "*.jpg"))

#Load pretrained classifier for face, eye, and smile detection
detectors = {}
for name, file_path in detector_files.items():
  detectors[name] = cv2.CascadeClassifier(folder_path + file_path)

#Iterate process through the whole image folder
for image_path in image_paths:
  #load and preprocess the input image to suit the detector
  image = cv2.imread(image_path)
  image_save = np.copy(image)
  print(image_path)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  #do inference process to detect faces in the image
  face_bboxes = detectors["face"].detectMultiScale(gray_image, scaleFactor=1.1, 
                minNeighbors=2, minSize=(150, 150), 
                flags=cv2.CASCADE_SCALE_IMAGE)
  
  #iterate through face bounding boxes
  for idx, (x, y, w, h) in enumerate(face_bboxes):
    #crop face bounding_box
    face = image[y:y+h, x:x+w]
    #save cropped face
    if args["crop_faces_folder"] is not None:
      face_crop_save= args["crop_faces_folder"] + image_path.split("/")[-1].split(".")[0] + "_" + str(idx) + ".jpg"
      cv2.imwrite(face_crop_save, face)
    gray_face = gray_image[y:y+h, x:x+w]

    #draw bounding box in original image
    if args["image_result_folder"] is not None:
      image_save = cv2.rectangle(image_save, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    #do inference to detect eye
    if args["eye_detect"] == True:
      eyes_bboxes = detectors["eyes"].detectMultiScale(gray_face, 
                    scaleFactor=1.2, minNeighbors=7, minSize=(40, 40), 
                    flags=cv2.CASCADE_SCALE_IMAGE)
      if args["image_result_folder"] is not None:
        for (x_e, y_e, w_e, h_e) in eyes_bboxes:
          image_save= cv2.rectangle(image_save, (x_e+x, y_e+y), 
                      (x_e+x+w_e, y_e+y+h_e), (0, 255, 255), 2)

  if args["image_result_folder"] is not None:
    cv2.imwrite(args["image_result_folder"] + image_path.split('/')[-1], image_save)