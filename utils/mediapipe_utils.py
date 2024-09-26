import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def initialize_mediapipe_pose(static_image_mode=False, min_detection_confidence=0.5):
    return mp_pose.Pose(static_image_mode=static_image_mode, min_detection_confidence=min_detection_confidence)

def extract_skeleton(frame, pose):
    results = pose.process(frame)
    if results.pose_landmarks:
        return [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
    return None