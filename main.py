import streamlit as st
import pickle
import numpy as np


pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
st.title('Laptop Price Predictor')
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight')
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])
ips = st.selectbox('IPS', ['Yes', 'No'])
screen_size = st.selectbox('Screen Size (in Inches)', [14, 15.6])
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x980', '3840x2160', '3200x1800',
                                                '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('Cpu Brand', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 32, 128, 500, 1000, 2000])
ssd = st.selectbox('SSD(in GB)', [0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1000, 1024])
gpu = st.selectbox('Gpu Brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())
if st.button('Predict Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = (((X_res ** 2) + (Y_res ** 2)) ** 0.5) / screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)
    price = int(np.exp(pipe.predict(query)[0]))
    st.title(price)






