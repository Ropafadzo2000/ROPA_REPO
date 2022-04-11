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


st.title('Video upload')
st.write('This is a web app to predict frames of a video based on        several features that you can see in the sidebar. Please adjust the        value of each feature. After that, click on the Predict button at the bottom to        see the prediction of the regressor.')


# In[ ]:




