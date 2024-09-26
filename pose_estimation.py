import cv2
import numpy as np
from utils.mediapipe_utils import extract_skeleton

def pose_estimation(frame, pose):
    
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    skeleton_data = extract_skeleton(frame, pose)

    if skeleton_data is not None:
        
        flattened_skeleton = np.array(skeleton_data).flatten()
        return flattened_skeleton
    return None
