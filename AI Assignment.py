#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.inception_v3 import InceptionV3
from keras.models import Model


# In[2]:


from tensorflow.keras.models import load_model
import tensorflow as tf


# In[3]:


model=tf.keras.applications.inception_v3.InceptionV3()


# In[4]:


print("model weights: ", model.get_weights())


# In[5]:


import pickle

# save
with open('model.pkl','wb') as f:
    pickle.dump(model,f)


# In[6]:


def predict_svm(to_predict):
    with open("'model.pkl'",'rb') as f_input:
        clf = pickle.loads(f_input) # maybe handled with a singleton to reduce loading for multiple predictions
    return clf.predict(to_predict)


# In[7]:


import streamlit as st


# In[8]:


def main():
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov"])
    if uploaded_video is not None:
        file_size = uploaded_video.size
        
        if file_is_of_right_size(file_size):
            user_input = st.text_input("Name of objects to detect: ")
            
            if st.button('Search'):
                video_processor(uploaded_video, user_input)
                saved_frame_path = 'predict_2'
                if not dirIsEmpty(saved_frame_path):
                    st.subheader(user_input)
                    images = Path(saved_frame_path).glob('*.png')
                    for img in images:
                        st.image(load_image(img),caption=user_input, width=20)
                else:
                    st.subheader('No frames of '+user_input+' found!')
if __name__ == '__main__':
    st.title("Object Detection using Inceptionv3")
    
    st.subheader("upload video with 2mb".title())
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




