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
from PIL import Image
from io import BytesIO
import aiohttp
import asyncio

# Function to load images from URL
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Function to load images from URL
# performanc optimization 

# Asynchronous function to load images from URL
async def load_image_from_url_2(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                imgnp = np.array(bytearray(data), dtype=np.uint8)
                return cv2.imdecode(imgnp, -1)
            else:
                st.error(f"Failed to load image from URL: {response.status}")
                return None


# def load_image_from_url_2(url):
#     try:
#         response = requests.get(url, stream=True, timeout=5)
#         response.raise_for_status()
#         imgnp = np.array(bytearray(response.raw.read()), dtype=np.uint8)
#         return cv2.imdecode(imgnp, -1)
#     except requests.RequestException as e:
#         st.error(f"Failed to load image from URL: {e}")
#         return None
# Initialize Streamlit
st.title("Real-Time Object Detection")

# Project Description Section
st.header("Project Description")
st.write("""
This project demonstrates a real-time object detection system using a webcam feed.
The model detects objects in the video stream and displays bounding boxes around them.

### Features:
- Real-time object detection with bounding boxes and labels.
- Interactive interface allowing users to start and stop the detection.
- Display of the camera feed along with detection results.
- Detailed project description with images and technology overview.

### Technologies Used:
- **OpenCV**: For capturing video from the webcam and processing the frames.
- **Streamlit**: For creating the web application and displaying the video stream and results.
- **Roboflow**: For providing the pre-trained object detection model and API.

Below are some examples of object detection in action:
""")

# Display example images in the description section
example_image_urls = [
    "https://www.mdpi.com/agriculture/agriculture-14-00751/article_deploy/html/images/agriculture-14-00751-g002.png",  # Replace with actual URLs of example images
    "https://ars.els-cdn.com/content/image/1-s2.0-S0168169923007482-gr7.jpg"   # Replace with actual URLs of example images
]
for url in example_image_urls:
    image = load_image_from_url(url)
    if image:
        st.image(image, caption="Example Object Detection", use_column_width=True)

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="0RomofAgLdzPXK31LjzR"
)

# Webcam capture initialization
cap = None

def get_frame():
    global cap
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

# Start/Stop Button
# if st.button("Start Object Detection"):
#     cap = cv2.VideoCapture(0)
#     stframe = st.empty()

#     while True:
#         frame = get_frame()
#         if frame is not None:
#             frame = infer_and_draw(frame)
#             # Convert frame to RGB for Streamlit
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             stframe.image(frame, channels="RGB")
#         else:
#             break



# URL Input for Image
cam_url = st.text_input("Enter the Camera URL", key="cam_url")

# Input to stop the detection loop
stop_input = st.text_input("Stop the detection (enter 0)", key="stop_input")

# Fetch and Process Image Button
if st.button("Start Object Detection", key="start_button"):
    if not cam_url:
        st.error("Please enter a valid camera URL.")
    else:
        stframe = st.empty()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def process_frames():
            while True:
                frame = await load_image_from_url_2(cam_url)
                if frame is not None:
                    frame = infer_and_draw(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame, channels="RGB")
                else:
                    st.error("Failed to load image from URL")
                    break
                if stop_input == "0":
                    break
        
        loop.run_until_complete(process_frames())



# Fetch and Process Image Button
# if st.button("Start Object Detection", key="start_button"):
#     while stop_trigger != "0":

#         cap = cv2.VideoCapture(cam_url)
#         stframe = st.empty()

#         while cap.isOpened():
            
    
#             frame = load_image_from_url_2(cam_url)
#             # ret, frame = cap.read()
#             if frame is not None:
#                 frame = infer_and_draw(frame)
#                 # Convert frame to RGB for Streamlit
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
#                 stframe.image(frame, channels="RGB")
                    
#             else:
#                 st.error("Failed to load image from URL")
#                 break
