import numpy as np

def ref3DModel():
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)


def ref2dImagePoints(landmarks):
    imagePoints = [[landmarks[30][0], landmarks[30][1]],
                   [landmarks[8][0], landmarks[8][1]],
                   [landmarks[36][0], landmarks[36][1]],
                   [landmarks[45][0], landmarks[45][1]],
                   [landmarks[48][0], landmarks[48][1]],
                   [landmarks[54][0], landmarks[54][1]]]
    return np.array(imagePoints, dtype=np.float64)

'''
def cameraMatrix(fl, center):
    mat = [[fl, 1, center[0]],
                    [0, fl, center[1]],
                    [0, 0, 1]]
    return np.array(mat, dtype=np.float)
    '''

# Camera internals
def cameraMatrix(size):
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
    return camera_matrix