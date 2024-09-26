import os
import cv2
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical
from pose_estimation import pose_estimation  # Ensure this is correct
from utils.mediapipe_utils import initialize_mediapipe_pose  # Import pose initialization

# Load video files and labels
video_files = [f for f in os.listdir('data/videos') if f.endswith('.mp4')]
labels = pd.read_csv('data/labels.csv')

output_dir = 'data/skeleton_images'
os.makedirs(output_dir, exist_ok=True)

# Initialize pose estimation once
pose = initialize_mediapipe_pose(static_image_mode=False, min_detection_confidence=0.5)

# Extract skeleton data from each video file
skeleton_data = {}
for video_file in video_files:
    cap = cv2.VideoCapture(os.path.join('data/videos', video_file))
    skeleton_data[video_file] = []  # Initialize list for this video's skeleton data
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        skeleton = pose_estimation(frame, pose=pose)  # Pass initialized pose to the function

        if skeleton is not None:
            print(f"Skeleton for {video_file}, Frame {frame_count}: {skeleton}")
            skeleton_data[video_file].append(skeleton)

            h, w, _ = frame.shape  

            for kp in skeleton:
                if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                    x, y = kp[0], kp[1]
                    
                    
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                       
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            x = x * w
                            y = y * h

                       
                        if 0 <= x < w and 0 <= y < h:

                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                        else:
                            print(f"Keypoint out of bounds: x={x}, y={y}, Frame Size: w={w}, h={h}")
                    else:
                        print(f"Invalid keypoint coordinates: x={x}, y={y}")
                # else:
                #     print(f"Invalid keypoint detected: {kp}")

            # Save the frame with skeleton points drawn
            output_path = os.path.join(output_dir, f"{video_file}_frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
        
        frame_count += 1
    cap.release()


# Pad or truncate all skeleton data sequences to the same length (30 frames)
frame_number = 30  # Assume 30 frames per video
key_points = 33 * 3  # Assume 33 key points with 3 coordinates each (x, y, z) flattened

def pad_or_truncate(data, frame_number, key_points):
    """Pad or truncate the data to have a fixed number of frames."""
    if len(data) > frame_number:
        return data[:frame_number]
    else:
        # Pad with zeros if the data is shorter than `frame_number`
        padding = [[0]*key_points for _ in range(frame_number - len(data))]
        return data + padding

# Apply padding/truncation to all skeleton data
skeleton_data = {k: pad_or_truncate(v, frame_number, key_points) for k, v in skeleton_data.items()}

# Convert skeleton data to lists to ensure JSON serialization works
skeleton_data_list = {k: np.array(v).tolist() for k, v in skeleton_data.items()}

# Save skeleton data as a JSON file
with open('data/skeleton_data.json', 'w') as f:
    json.dump(skeleton_data_list, f)

# Create a dictionary to map video file names to labels
video_label_map = {}
for index, row in labels.iterrows():
    video_label_map[row['video_file']] = row['label']

# Reshape data into lists of samples and prepare labels
skeleton_data_array = [np.array(skeleton_data_dict) for skeleton_data_dict in skeleton_data_list.values()]
labels_list = [video_label_map[video_file] for video_file in skeleton_data.keys()]

# Convert the labels to integer encoding and one-hot encode them
unique_labels = np.unique(labels_list)
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
int_labels = [label_to_int[label] for label in labels_list]
y_encoded = to_categorical(int_labels, num_classes=len(unique_labels))

# Ensure all skeleton data entries are converted to NumPy arrays with consistent shapes
skeleton_data_array = np.array(skeleton_data_array)

# Check if there are enough samples for a train-test split
if len(skeleton_data_array) > 1:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(skeleton_data_array, y_encoded, test_size=0.5, random_state=42)
else:
    # If only one sample, use all data for both training and testing (not ideal, but a workaround)
    X_train, X_test = np.array(skeleton_data_array), np.array(skeleton_data_array)
    y_train, y_test = np.array(y_encoded), np.array(y_encoded)

# Define the model architecture
input_layer = Input(shape=(frame_number, key_points))
x = LSTM(64, return_sequences=False)(input_layer)
x = Dense(64, activation='relu')(x)
x = Dense(len(unique_labels), activation='softmax')(x)  # Corrected output layer
model = Model(inputs=input_layer, outputs=x)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


train_loss, train_accuracy = model.evaluate(X_train, y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
# Save the trained model

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save('models/gait_recognition_model.h5')

# Load test video file and perform prediction
test_video_file = 'devesh.mp4'
test_cap = cv2.VideoCapture(test_video_file)

# Extract skeleton structure from each frame
test_skeleton_data = []

while test_cap.isOpened():
    ret, frame = test_cap.read()
    if not ret:
        break 
    skeleton = pose_estimation(frame, pose=pose)  # Pass initialized pose to the function
    if skeleton is not None:
        test_skeleton_data.append(skeleton)  # Append skeleton data if valid

# Pad or truncate the test skeleton data to 30 frames
test_skeleton_data = pad_or_truncate(test_skeleton_data, frame_number, key_points)

# Reshape the test skeleton data to match the model input shape
test_skeleton_data = np.array(test_skeleton_data).reshape(1, frame_number, key_points)

# Perform prediction
prediction = model.predict(test_skeleton_data)
predicted_index = np.argmax(prediction)

# Map the index back to the corresponding label (person's name)
predicted_person = unique_labels[predicted_index]

# Display the predicted person's name
print(f"Predicted person: {predicted_person}")
