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

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

face3Dmodel = world.ref3DModel()

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

def midpoint(p1,p2) :
    return int((p1.x+p2.x)/2) , int((p1.y+p2.y)/2)

def eye_aspect_ratio(eye_points, facial_landmarks) :
    A = dist.euclidean(facial_landmarks[eye_points[1]],facial_landmarks[eye_points[5]])
    B = dist.euclidean(facial_landmarks[eye_points[2]],facial_landmarks[eye_points[4]])
    C = dist.euclidean(facial_landmarks[eye_points[0]],facial_landmarks[eye_points[3]])
    ear = (A + B) / (2.0 * C)
    return ear

'''
def eye_blink_ratio(eye_points, facial_landmarks) :
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
        
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        ver_line_length = hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))
        hor_line_length = hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
        ratio = hor_line_length/ver_line_length
        return ratio
'''

def draw(img, corners, imgpts):
    img = cv2.line(img, corners, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corners, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corners, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

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

        landmarks = predictor(gray,face)
        landmarks = face_utils.shape_to_np(landmarks)

        for n in range(0,68) :
            x = landmarks[n][0]
            y = landmarks[n][1]
            cv2.circle(frame,(x,y),3,(0,255,0),-1)
        
        #left_eye_ratio = eye_blink_ratio([36,37,38,39,40,41], landmarks)
        #right_eye_ratio = eye_blink_ratio([42,43,44,45,46,47], landmarks)
        left_eye_ratio = eye_aspect_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = eye_aspect_ratio([42,43,44,45,46,47], landmarks)

        ear = ((left_eye_ratio + right_eye_ratio) /2) 
        #print(ear)
        if ear < EYE_AR_THRESH :
            COUNTER += 1
        else :
            if COUNTER >= EYE_AR_CONSEC_FRAMES :
                TOTAL += 1
            COUNTER = 0
        '''
        if blink ratio > 5.7 :
            blink_count = blink_count + 1
            cv2.putText(frame,'BLINK : {}'.format(str(blink_count)),(50,50),font,3,(0,0,255),2,cv2.LINE_4)
        '''
        refImgPts = world.ref2dImagePoints(landmarks)

        cameraMatrix = world.cameraMatrix(frame.shape)

        mdists = np.zeros((4, 1), dtype=np.float64)

        # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, cameraMatrix, mdists)

        #noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoints3D = np.float32([[500.0,0,0], [0,500.0,0], [0,0,1000.0]]).reshape(-1,3)
        noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)
        #print(tuple(noseEndPoint2D[2].ravel()))
        
        #  draw nose line
        p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
        p2 = tuple(noseEndPoint2D[2].ravel())
        #cv2.line(frame, p1, p2, (110, 220, 0),
        #            thickness=2, lineType=cv2.LINE_AA)
        draw(frame,p1,noseEndPoint2D)
        

        # calculating euler angles
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        roll = 180*math.atan2(-rmat[2][1], rmat[2][2])/math.pi
        pitch = 180*math.asin(rmat[2][0])/math.pi
        yaw = 180*math.atan2(-rmat[1][0], rmat[0][0])/math.pi

        #print("Roll: ", roll)
        #print("Pitch: ", pitch)
        #print("yaw: ", yaw)

        if angles[1] < -15:
            GAZE = "Looking: Left"
        elif angles[1] > 15:
            GAZE = "Looking: Right"
        else:
            GAZE = "Forward"    

        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 90
        print(ang1)
        if ang1 >= 48:
            NOD ='Head down'
        elif ang1 <= -48:
            NOD ='Head up'
        else :
            NOD = "Forward"  
        #print(ang1)

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),font , 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (400, 30),font , 0.7, (0, 0, 255), 2)
        cv2.putText(frame, GAZE, (10, 400),font, 1, (0, 255, 80), 2)
        cv2.putText(frame, NOD, (390, 400),font, 1, (0, 255, 80), 2)

    cv2.imshow('demo video',frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break 

cap.release()    
cv2.destroyAllWindows()