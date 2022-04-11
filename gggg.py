#!/usr/bin/env python
# coding: utf-8

# In[48]:


from keras.models import load_model
import sys


# In[49]:


from keras.applications.inception_v3 import InceptionV3
from keras.models import Model


# In[50]:


from tensorflow.keras.models import load_model
import tensorflow as tf


# In[51]:


model=tf.keras.applications.inception_v3.InceptionV3()


# In[7]:


pip install opencv-python


# In[ ]:





# In[52]:


import cv2
import numpy as np


def predict(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0


# In[53]:


import streamlit as st 


# In[54]:


#Functions
def predict(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # Predict with the Inceptionv3 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        label  = ("{}: {:.2f}%".format(label, prob * 100))

    st.markdown(label)


# In[55]:


def predict2(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # Predict with the Inceptionv3 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        pred_class = label
       
    return pred_class


# In[56]:


def object_detection(search_key,frame, model):
    label = predict2(frame,model)
    label = label.lower()
    if label.find(search_key) > -1:
        st.image(frame, caption=label)

        return sys.exit()
    else:
        pass
        st.text('Not Found')


# In[58]:


def main():
    """Deployment"""
    st.title("Visualization ")
    st.text("Streamlit and Inceptionv3 App")

    activities = ["Detect Objects", "About"]
    choice = st.sidebar.selectbox("Choose Activity", activities)
    
    if choice == "Detect Objects":
        st.subheader("Upload the video")

        video_file = st.file_uploader("upload video...", type=["mp4", "mov"])

        if video_file is not None:
            path = video_file.name
            with open(path,mode='wb') as f: 
                f.write(video_file.read())         
                st.success("Saved Video")
                video_file = open(path, "rb").read()
                st.video(video_file)
            cap = cv2.VideoCapture(path)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
            
            if st.button("Get Frames"):
                
                # Start video prediction
                while cap.isOpened():
                    ret, frame = cap.read()
    
                    if not ret:
                        break
    
                    # Object detection
                    predict(frame, model)
    
                    #Display the frame
                    st.image(frame, caption='Video Stream', use_column_width=True)
    
                cap.release()
                output.release()
                cv2.destroyAllWindows()
                
            key = st.text_input('Search')
            key = key.lower()
            
            if key is not None:
            
                if file_is_of_right_size(file_size):
                    user_input = st.text_input("Name of objects to detect: ")

                if st.button('Search'):
                    video_processor(video_file, user_input)
                    saved_frame_path = 'predict_2'
                    if not dirIsEmpty(saved_frame_path):
                        st.subheader(user_input)
                        images = Path(saved_frame_path).glob('*.png')
                        for img in images:
                            st.image(load_image(img),caption=user_input, width=20)
                    else:
                        st.subheader('No frames of '+user_input+' found!')



                        # Perform object detection
                        object_detection(key,frame, model)


                        cap.release()
                        output.release()
                        cv2.destroyAllWindows()


# In[59]:


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




