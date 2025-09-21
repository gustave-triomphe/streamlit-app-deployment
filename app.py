
from __future__ import annotations
import pandas as pd
import streamlit as st
from src.utils import *

st.title('Penguins Sex Classifier')
st.write("This app uses a Random Forest classifier to predict penguin sex based on body measurements.")


data_load_state = st.text('Loading data...')
@st.cache_data
def load_data():
    return load_penguins()
data = load_data()
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

clean_data = clean_data(data)
if st.checkbox('Show cleaned data'):
    st.subheader('Cleaned data')
    st.write(clean_data)


selected_X, y = select_data(clean_data)
if st.checkbox('Show selected data'):
    st.subheader('Selected data')
    st.write(selected_X, y)


clf, report, cm, classes = train_rf(selected_X, y)
st.subheader('Report')
report_df = pd.DataFrame(report).transpose()
st.table(report_df)



st.subheader('Confusion Matrix')
st.pyplot(plot_confusion_matrix(cm, classes))

