import cv2
import numpy as np
from scipy.spatial import distance as dist
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

def emotion_loadModel():
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


def midpoint(p1,p2) :
    return int((p1.x+p2.x)/2) , int((p1.y+p2.y)/2)


def eye_aspect_ratio(eye_points, facial_landmarks) :
    A = dist.euclidean(facial_landmarks[eye_points[1]],facial_landmarks[eye_points[5]])
    B = dist.euclidean(facial_landmarks[eye_points[2]],facial_landmarks[eye_points[4]])
    C = dist.euclidean(facial_landmarks[eye_points[0]],facial_landmarks[eye_points[3]])
    ear = (A + B) / (2.0 * C)
    return ear


def draw(img, corners, imgpts):
    img = cv2.line(img, corners, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corners, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corners, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


def face_processing(gray,x,y,w,h) :
    roi_gray=gray[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
    roi_gray=cv2.resize(roi_gray,(48,48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255
    #print(img_pixels[0,:,:,:].shape)
    return img_pixels


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