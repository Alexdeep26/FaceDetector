import cv2

#Pre-Trained data on different face frontals (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

#Detecting faces in an image file :

#Choose an image to detect the face
img = cv2.imread('img.jpg')

#Must convert the selected image to grayscale
grayscaled_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangle around the faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#View the image with faces
cv2.imshow('Face Detector',img)
cv2.waitKey()


