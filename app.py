# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode


# Loading pre-trained parameters for the cascade classifier
try:
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # load model
    model = load_model("Final_model_CNN.h5")
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']  # Emotion that will be predicted
except Exception:
    st.write("Error loading cascade classifiers")
    
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        label=[]
        img = frame.to_ndarray(format="bgr24")
        face_detect = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3,1)
        
        b = img.copy()  # Initialize b with a copy of img
        
        for (x,y,w,h) in faces:
            a=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0) ## reshaping the cropped face image for prediction
            prediction = model.predict(roi)[0]   #Prediction
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            b=cv2.putText(a,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
               
        return b


def main():
    # Face Analysis Application #
    activiteis = ["Webcam Face Detection", "About"]  # Corrected choice labels
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Hikmet Topak """)
    if choice == "Webcam Face Detection":
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Emotion Recognition WebApp</h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

    elif choice == "About":  # Corrected choice label
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Emotion Recognition WebApp</h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion recognition application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Hikmet Topak using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose.</h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass



if __name__ == "__main__":
    main()
