import pandas as pd   # data preprocessing
import numpy as np    # mathematical computation
import pickle
from sklearn import *
import streamlit as st
import sys

# Load the model and dataset
model = pickle.load(open('lr_model.pkl','rb'))
df = pickle.load(open('data.pkl','rb'))

st.title('Product selling price')
st.header('Fill the details to predict product selling  Price')




Product=st.selectbox('Product',df['Product'].unique())
year=st.selectbox('year is',[2015, 2014, 2013, 2016])
Profit=st.selectbox('Profit',df['Profit'].unique())
Quantity=st.selectbox('Quantity',df['Quantity'].unique())



if st.button('Predict Product selling Price'):
        

        pred = model.predict([[Product,year,Profit,Quantity]])
        output = round(pred[0],2)
        if pred < 0: # handeling negative outputs.
            st.error('The input values must be irrelevant, try again by giving relevent information.')
      
        write = str(pred) 
        st.success(write)


