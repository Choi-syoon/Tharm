import os
import cv2
import argparse
import numpy as np
import pandas as pd
import mediapipe as mp

from tqdm import tqdm
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler

from extractor import Extractor
extractor = Extractor.worker()

class get_class():
    def __init__(self):
        self.sentiment_model = pipeline(model="WhitePeak/bert-base-cased-Korean-sentiment")

    def _emotion_inference(self, string):
        result = self.sentiment_model(string)
        
        if result[0]['label'] == 'LABEL_0':
            return "negative"
        elif result[0]['label'] == 'LABEL_1':
            return "positive"
        

# def main():
#     for file in tqdm(os.listdir(DEST_PATH)):
#         filename = str(os.path.splitext(file)[0])
#         file_path = os.path.join(DEST_PATH)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-srcPath', required=True, help='Source Path')
    parser.add_argument('-destPath', required=True, help='Feature Destination Path')
    parser.add_argument('-model', nargs='+', required=False, help='Use Models')
    args = parser.parse_args()

    SOURCE_PATH = args.srcPath
    DEST_PATH = args.destPath
    MODEL = args.model

