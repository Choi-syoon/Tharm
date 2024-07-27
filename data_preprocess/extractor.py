import os
import cv2
import numpy as np
import mediapipe as mp

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

class Extractor:
    # def normalization(npy):  # keypoints normalization
    #     scaler = MinMaxScaler()
    #     flat_npy = np.reshape(npy, (-1, np.shape(npy)[-1]))
    #     scaled_flat_npy = scaler.fit_transform(flat_npy)
    #     scaled_npy = np.reshape(scaled_flat_npy, np.shape(npy))
    #     return scaled_npy  
    
    def mp_hands_keypoints(self, image):    # multi hands keypoints extraction 
        hands_model = mp.mp_hands.Hands(max_num_hands = 2,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) 
        
        hands_results = hands_model.process(image)

        if hands_results.multi_hand_landmarks:
            hands_keypoints = []
            for hand_landmarks in hands_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    hands_keypoints.append([landmark.x, landmark.y, landmark.z])
            self.HANDS_KEYPOINTS.append(hand_landmarks)

    def mp_pose_keypoints(self, image): # pose keypoints extraction
        pose_model = mp.solution.pose.Pose(min_detection_confidence=0.5, 
                                           min_tracking_confidence=0.5)
        
        pose_results = pose_model.process(image)

        if pose_results.pose_landmarks:
            pose_keypoints = []
            for landmark in pose_results.pose_landmarks.landmark:
                pose_keypoints.append([landmark.x, landmark.y, landmark.z])
            self.POSE_KEYPOINTS.append(pose_keypoints)

    def mp_facemesh_keypoints(self, image): # face mesh keypoints extraction 
        faceMesh_model =  mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                                           max_num_faces=1,
                                                           refine_landmarks=True,
                                                           min_detection_confidence=0.5)
        
        faceMesh_results = faceMesh_model.process(image)

        if faceMesh_results.multi_face_landmarks:
            faceMesh_keypoints = []
            for landmark in faceMesh_results.multi_face_landmarks.landmark:
                faceMesh_keypoints.append([landmark.x, landmark.y, landmark.z])
            self.FACEMESH_KEYPOINTS.append(faceMesh_keypoints)

    def worker(self, SOURCE_PATH, MODEL, DEST_PATH):
        for file in tqdm(os.listdir(SOURCE_PATH)):
            filename = str(os.path.splfitext(file)[0])
            file_path = os.path.join(SOURCE_PATH, file)

            self.POSE_KEYPOINTS = []
            self.FACEMESH_KEYPOINTS = []
            self.HANDS_KEYPOINTS = []
            
            cap = cv2.VideoCapture(file_path)

            while True:
                open, image = cap.read()
                if not open:
                    break
                    # continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if 'pose' in MODEL:
                    self.mp_pose_keypoints(image_rgb)
                if 'hands' in MODEL:
                    self.mp_hands_keypoints(image_rgb)
                if 'facemesh' in MODEL:
                    self.mp_facemesh_keypoints(image_rgb)

            if self.POSE_KEYPOINTS:
                np.save(f"{DEST_PATH}/{filename}_pose.npy", self.POSE_KEYPOINTS)

            if self.FACEMESH_KEYPOINTS:
                np.save(f"{DEST_PATH}/{filename}_facemesh.npy", self.FACEMESH_KEYPOINTS)

            if self.HANDS_KEYPOINTS:
                np.save(f"{DEST_PATH}/{filename}_hands.npy", self.HANDS_KEYPOINTS)