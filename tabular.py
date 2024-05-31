from operator import index
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from pycaret import classification , regression
from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report
import os 
from sklearn.model_selection import train_test_split


if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
def highlight_cols(s):
    color = 'grey'
    return 'background-color: %s' % color

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoNickML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df.head())

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = ProfileReport(df, title="Profiling Report")
    st_profile_report(profile_df)

if choice == "Modelling": 
    model_type = st.selectbox('Choose the problem type', ('Classification', 'Regression'))
    chosen_target = st.selectbox('Choose the Target Column to predict', df.columns)
    testsize = st.slider('unseen test-data size', 0.1, 0.5)
    train_data = df
    unseen_data = train_data.tail(int(len(train_data)*testsize))
    train_data.drop(unseen_data.index)

    if model_type == "Classification":
        plot_col1, plot_col2 , plot_col3, plot_col4  = st.columns(4)
        with plot_col1:
            class_plot1 = st.selectbox('Choose a model performance plot1', ('confusion_matrix', 'feature', 'error' ,'learning' , 'class_report', 'None'))
        with plot_col2:
            class_plot2 = st.selectbox('Choose a model performance plot2', ('confusion_matrix', 'feature', 'error' ,'learning', 'class_report','None'))
        with plot_col3:
            class_plot3 = st.selectbox('Choose a model performance plot3', ('confusion_matrix', 'feature', 'error' ,'learning','class_report','None'))
        with plot_col4:
            class_plot4 = st.selectbox('Choose a model performance plot4', ('confusion_matrix', 'feature', 'error' ,'learning','class_report', 'None'))
        if st.button('Run Modelling'): 
            classification.setup(train_data, target=chosen_target)
            setup_df = classification.pull()
            st.dataframe(setup_df)
            best_model = classification.compare_models()
            compare_df = classification.pull()
            st.dataframe(compare_df)
            if class_plot1 != 'None':
                classimg1 = classification.plot_model(best_model, plot=class_plot1, display_format="streamlit", save=True)
            if class_plot2 != 'None':
                classimg2 = classification.plot_model(best_model, plot=class_plot2, display_format="streamlit", save=True)
            if class_plot3 != 'None':
                classimg3 = classification.plot_model(best_model, plot=class_plot3, display_format="streamlit", save=True)
            if class_plot4 != 'None':
                classimg4 = classification.plot_model(best_model, plot=class_plot4, display_format="streamlit", save=True)
            st.image(classimg1)
            st.image(classimg2)
            st.image(classimg3)
            st.image(classimg4)
            predictions = classification.predict_model(best_model , data= unseen_data)
            st.dataframe(predictions.style.applymap(highlight_cols, subset=['prediction_label']))

    if model_type == "Regression":
        TimeSeries = st.checkbox('TimeSeries Data')
        plot_col1, plot_col2 , plot_col3, plot_col4  = st.columns(4)
        with plot_col1:
            chosen_plot1 = st.selectbox('Choose a model performance plot1', ('residuals', 'feature', 'error' ,'learning' , 'None'))
        with plot_col2:
            chosen_plot2 = st.selectbox('Choose a model performance plot2', ('residuals', 'feature', 'error' ,'learning', 'None'))
        with plot_col3:
            chosen_plot3 = st.selectbox('Choose a model performance plot1', ('residuals', 'feature', 'error' ,'learning', 'None'))
        with plot_col4:
            chosen_plot4 = st.selectbox('Choose a model performance plot2', ('residuals', 'feature', 'error' ,'learning', 'None'))
        tempdata = train_data.drop(chosen_target, axis=1)
        top_feature = st.selectbox('Choose the Column to compare', tempdata.columns)
        if st.button('Run Modelling'): 
            regression.setup(train_data, target=chosen_target)
            setup_df = regression.pull()
            st.dataframe(setup_df)
            best_model = regression.compare_models()
            compare_df = regression.pull()
            st.dataframe(compare_df)
            if chosen_plot1 != 'None':
                img1 = regression.plot_model(best_model, plot=chosen_plot1, display_format="streamlit", save=True)
            if chosen_plot2 != 'None':
                img2 = regression.plot_model(best_model, plot=chosen_plot2, display_format="streamlit", save=True)
            if chosen_plot3 != 'None':
                img3 = regression.plot_model(best_model, plot=chosen_plot3, display_format="streamlit", save=True)
            if chosen_plot4 != 'None':
                img4 = regression.plot_model(best_model, plot=chosen_plot4, display_format="streamlit", save=True)
            st.image(img1)
            st.image(img2)
            st.image(img3)
            st.image(img4)
            predictions = regression.predict_model(best_model , data= unseen_data)
            st.dataframe(predictions.style.applymap(highlight_cols, subset=['prediction_label']))
            if TimeSeries:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_data[top_feature], y=train_data[chosen_target], mode='lines', name='Actual Train', marker=dict(color='blue')))
                fig.add_trace(go.Scatter(x=unseen_data[top_feature], y=unseen_data[chosen_target], mode='lines', name='Actual Test', marker=dict(color='green')))
                fig.add_trace(go.Scatter(x=unseen_data[top_feature], y=predictions['prediction_label'], mode='lines', name='Predicted Test', marker=dict(color='red')))
                fig.update_layout(xaxis_title=top_feature,yaxis_title=chosen_target)
                st.plotly_chart(fig)
            
            