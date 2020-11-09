import cv2
import numpy as np
import dlib
from math import hypot
from scipy.spatial import distance as dist
from imutils import face_utils
import argparse
import world
import math
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
import age_detection

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX

def loadModel():
    num_classes = 7
    model = Sequential()

	#1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

	#2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	#3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
    
    model.add(Flatten())

	#fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_classes, activation='softmax'))
	
	#----------------------------
    model.load_weights(filepath='facial_expression_model_weights.h5')
    return model

emotion_model = loadModel()


while True :
    ret , frame  = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x,y = face.left() , face.top()
        x1,y1 = face.right() , face.bottom()
        w = face.right() - face.left()
        h = face.bottom() - face.top()
        cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
        
        roi_gray=gray[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        print(img_pixels[0,:,:,:].shape)

        predictions = emotion_model.predict(img_pixels)
        print(predictions)
        
        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    #cv2.putText(frame,result['dominant_emotion'],(50,50),font,3,(0,0,255),2,cv2.LINE_4)
    cv2.imshow('demovideo',frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break 

cap.release()    
cv2.destroyAllWindows()