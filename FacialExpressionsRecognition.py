from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

#this file is used to detect the face from camera
face_classifier = cv2.CascadeClassifier(r'C:\Users\Tajwar\Downloads\CaFoscari University of Venice\Year 1\Artificial Intelligence Machine Learning\Machine Learning Project\haarcascade_frontalface_default.xml')

#we are loading the model that we have trained and validate all the data
classifier =load_model(r'C:\Users\Tajwar\Downloads\CaFoscari University of Venice\Year 1\Artificial Intelligence Machine Learning\Machine Learning Project\Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

#Calling the Camera
#Taking the Inputs from camera and recognizing the facial Expressions
cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video
    ret, frame = cap.read() #Read the camera and name it as frame
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Convert the frames into blackNwhite

   #Graying the image and scale down frame
    #because the camera frame has large pixels so need to scale down
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        #Drawing the rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #Reason of interest (Face)
        roi_gray = gray[y:y+h,x:x+w] #rectangle into gray color
        #resizing the image by 48x48
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0 #lowering the pixel size
            roi = img_to_array(roi) #for mathematical calclation we are converting it to array

            #Expending the dimentions
            roi = np.expand_dims(roi,axis=0) #

        # make a prediction on the ROI(reason of interest), then lookup the class

            preds = classifier.predict(roi)[0]

            #argmax is a function used to pick the prediction which have more accuraccy like if
            # the model predicts angry 70% and sad 30% if will predict image as angry

            label=class_labels[preds.argmax()]

            #Labeling the frame on upper side of the rectangle
            label_position = (x,y)

            #simply putting the lable on the frame if sad happy angry
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    #Showing the frame
    cv2.imshow('Emotion Detector',frame)
    #if you press q button the frame will stop to avoid camera hanging
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























