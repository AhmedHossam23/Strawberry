#packages for roboflow
import inference_sdk
from inference_sdk import InferenceHTTPClient
import streamlit as st
#packages for computer vision
import cv2
import cvlib
from cvlib.object_detection import draw_bbox
import numpy as np
import matplotlib.pyplot as plt
#libraries for handling request with the camera
import requests
import urllib.request
import time

# Initialize Streamlit
st.title("Ripe Strawberry Real-Time Detection Model")
stframe = st.empty()

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="GmRcGwYcYUQfjQ331pz6"
)

# Initialize the webcam
cap = cv2.VideoCapture(0)

def get_frame():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame")
        return None
    return frame

def infer_and_draw(frame):
    try:
        result = CLIENT.infer(frame, model_id="strawberry-detection-msf0m/3")
        # Draw bounding boxes and labels
        for prediction in result.get("predictions", []):
            x0 = prediction["x"]
            y0 = prediction["y"]
            w = prediction["width"]
            h = prediction["height"]
            class_name = prediction["class"]

            x1, y1 = int(x0 - w/2), int(y0 - h/2)
            x2, y2 = int(x0 + w/2), int(y0 + h/2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    except Exception as e:
        st.error(f"Inference error: {e}")
    
    return frame

while True:
    frame = get_frame()
    if frame is not None:
        frame = infer_and_draw(frame)
        # Convert frame to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")
    else:
        break

cap.release()
