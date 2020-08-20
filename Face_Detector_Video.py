import cv2

#Pre-Trained data on different face frontals (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

#Detecting faces from webcam :

#Choose an webcam source to detect the face
webcam = cv2.VideoCapture(0)

#Choose a video source to detect the face
webcam = cv2.VideoCapture('video.mp4')

#Iterate over frames in the video
while True:
    #Read frames from the feed
    successfull_frame_read, frame = webcam.read()

    #convert the frames to grayscale
    grayscaled_img = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    #get the face coordinates from the frames
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw recatngle arounf all the detected faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    #Show the frames with drawn rectangles
    cv2.imshow('Face Detector',frame)
    key = cv2.waitKey(1)

    #Stop the feed when Q is pressed
    if key==81 or key==113:
        break

webcam.release()