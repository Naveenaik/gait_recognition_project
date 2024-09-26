import cv2
import numpy as np
from utils.mediapipe_utils import extract_skeleton

def pose_estimation(frame, pose):
    """
    Extracts skeleton keypoints from a given frame using an already initialized MediaPipe pose estimator.
    
    Args:
        frame: The video frame to process.
        pose: The pre-initialized MediaPipe pose estimator.
        
    Returns:
        Flattened list of skeleton keypoints if detected, None otherwise.
    """
    # Pre-process frame
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extract skeleton structure using MediaPipe
    skeleton_data = extract_skeleton(frame, pose)

    if skeleton_data is not None:
        # Flatten the (33, 3) shape into a single list of 99 values
        flattened_skeleton = np.array(skeleton_data).flatten()
        return flattened_skeleton
    return None
