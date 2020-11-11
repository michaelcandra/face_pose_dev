import cv2
import numpy as np
import dlib
from math import hypot
from imutils import face_utils
import argparse
import world
import utils
import math


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

#Open dlib and get predictor for facial Landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# EAR Threshold 
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

face3Dmodel = world.ref3DModel()
emotion_model = utils.emotion_loadModel()

# Define the codec and create VideoWriter object
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
fps = cap.get(5)
   
size = (frame_width, frame_height)
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('output.mp4',fourcc, fps, size)
#out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1920, 1080))
out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640,480))


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
        
        img_pixels = utils.face_processing(gray,x,y,w,h)
        predictions = emotion_model.predict(img_pixels)
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
        left_eye_ratio = utils.eye_aspect_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = utils.eye_aspect_ratio([42,43,44,45,46,47], landmarks)
        ear = ((left_eye_ratio + right_eye_ratio) /2) 

        #print(ear)
        if ear < EYE_AR_THRESH :
            COUNTER += 1
        else :
            if COUNTER >= EYE_AR_CONSEC_FRAMES :
                TOTAL += 1
            COUNTER = 0
        
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),font , 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (400, 30),font , 0.7, (0, 0, 255), 2)
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
        utils.draw(frame,p1,noseEndPoint2D)
        
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
        '''
        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 90
        '''
        print(angles[0])
        if angles[0] < -100 :
            NOD ='Head down'
        elif angles[0] > 100:
            NOD ='Head up'
        else :
            NOD = "Forward"  
        #print(ang1)

        cv2.putText(frame, GAZE, (10, 400),font, 1, (0, 255, 80), 2)
        cv2.putText(frame, NOD, (390, 400),font, 1, (0, 255, 80), 2)

    out.write(frame)
    cv2.imshow('demo video',frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break 

cap.release() 
out.release()   
cv2.destroyAllWindows()