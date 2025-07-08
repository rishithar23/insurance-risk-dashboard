import streamlit as st
import pandas as pd

st.title("insurance risk dashboard")
st.write("sample risk exposure dashboard")

data=pd.DataFrame({
'policy type':['health','auto','home'],
'risk score':[0.7,0.4,0.9]})
st.bar_chart(data.set_index('Policy type'))