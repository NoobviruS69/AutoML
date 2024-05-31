import streamlit as st
from vgg_classification import sub_app_1
from yoloV8_OD import sub_app_2

def main():

    options = ["Home", "Simple classification using VGG16", "Sub App 2"]
    option = st.sidebar.selectbox("Select Sub App", options)

    if option == "Simple classification using VGG16":
        sub_app_1()
    elif option == "Sub App 2":
        sub_app_2()
    # Add an empty condition or default case if needed
    else:
        if option == "Home":
            st.title("Wecome to Image segemnt of DataLensAI")

if __name__ == "__main__":
    main()