#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model
import sys


# In[2]:


from keras.applications.inception_v3 import InceptionV3
from keras.models import Model


# In[3]:


from tensorflow.keras.models import load_model
import tensorflow as tf


# In[4]:


model=tf.keras.applications.inception_v3.InceptionV3()


# In[7]:


pip install opencv-python


# In[ ]:





# In[ ]:





# In[8]:


import cv2
import numpy as np


def predict(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0


# In[9]:


import streamlit as st 


# In[10]:


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


# In[12]:


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


# In[13]:


def object_detection(search_key,frame, model):
    label = predict2(frame,model)
    label = label.lower()
    if label.find(search_key) > -1:
        st.image(frame, caption=label)

        return sys.exit()
    else:
        pass
        #st.text('Not Found')
        #return sys.exit()


# In[15]:


def main():
    """Deployment"""
    st.title("Visualization ")
    st.text("Built with Streamlit and Inceptionv3")

    activities = ["Detect Objects", "About"]
    choice = st.sidebar.selectbox("Choose Activity", activities)
    
    if choice == "Detect Objects":
        st.subheader("Upload the video to detect")

        video_file = st.file_uploader("upload the video...", type=["mp4", "avi"])

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
            
            if st.button("Detect Objects"):
                
                # Start the video prediction loop
                while cap.isOpened():
                    ret, frame = cap.read()
    
                    if not ret:
                        break
    
                    # Perform object detection
                    #frame = object_detection(frame, model)
                    predict(frame, model)
    
                    # Display the resulting frame
                    #st.image(frame, caption='Video Stream', use_column_width=True)
    
                cap.release()
                output.release()
                cv2.destroyAllWindows()
                
            key = st.text_input('Search key')
            key = key.lower()
            
            if key is not None:
            
                if st.button("Search for the object"):
                    
                    
                    # Start the video prediction loop
                    while cap.isOpened():
                        ret, frame = cap.read()
        
                        if not ret:
                            break
        
                        # Perform object detection
                        object_detection(key,frame, model)
                        #frame = predict(frame, model)
        
                        # Display the resulting frame
                        #st.image(frame, caption='Video Stream', use_column_width=True)
        
                    cap.release()
                    output.release()
                    cv2.destroyAllWindows()



# In[16]:


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




