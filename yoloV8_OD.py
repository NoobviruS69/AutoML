import os
import shutil
import random
import numpy as np
import cv2
import tempfile
from PIL import Image , ImageDraw
import streamlit as st
from zipfile import ZipFile
import yaml
from ultralytics import YOLO

def sub_app_2():

    def save_files_to_folders(uploaded_files, folder_name):
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(folder_name, uploaded_file.name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

    def split_train_val_data(data_dir, val_percentage):
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

        images = os.listdir(os.path.join(data_dir, 'images'))
        num_val = int(len(images) * val_percentage)
        val_images = random.sample(images, num_val)

        for img in images:
            src_img_path = os.path.join(data_dir, 'images', img)
            src_label_path = os.path.join(data_dir, 'labels', img[:-4] + '.txt')
            if os.path.exists(src_img_path) and os.path.exists(src_label_path):
                if img in val_images:
                    shutil.move(src_img_path, os.path.join(val_dir, 'images', img))
                    shutil.move(src_label_path, os.path.join(val_dir, 'labels', img[:-4] + '.txt'))
                else:
                    shutil.move(src_img_path, os.path.join(train_dir, 'images', img))
                    shutil.move(src_label_path, os.path.join(train_dir, 'labels', img[:-4] + '.txt'))
                
        shutil.rmtree(os.path.join(data_dir, 'images'))
        shutil.rmtree(os.path.join(data_dir, 'labels'))

    def create_yaml(classes, data_dir):
        class_dict = {int(idx) : class_name for idx, class_name in enumerate(classes)}
        yolo_data = {'train': '/Users/abhinavkrishna/Desktop/AutoML/yolo_train_data/train', 'val': '/Users/abhinavkrishna/Desktop/AutoML/yolo_train_data/val', 'nc': len(classes), 'names': class_dict}
        with open(os.path.join(data_dir, 'data.yaml'), 'w') as yaml_file:
            yaml.dump(yolo_data, yaml_file, default_flow_style=False)
                
    def upload_segment():
        st.title("Upload Data for Object Detection")
        st.write("### Upload Images and Labels")

        # Upload images and labels
        images = st.file_uploader("Upload Images (ZIP or Folder)", type=["zip", "png", "jpg"], accept_multiple_files=True)
        labels = st.file_uploader("Upload Labels (ZIP or Folder)", type=["zip", "txt"], accept_multiple_files=True)

        # Specify number of classes
        num_classes = st.number_input("Number of Classes", min_value=1, max_value=10, value=1)

        # Input class names
        class_names = []
        for i in range(num_classes):
            class_names.append(st.text_input(f"Class {i + 1} Name"))
        
        if st.button("Start Training"):
            if images and labels:
                # Save uploaded files into 'yolo_train_data' folder
                save_files_to_folders(images, 'yolo_train_data/images')
                save_files_to_folders(labels, 'yolo_train_data/labels')

                # Split data into train and val folders
                split_train_val_data('yolo_train_data', val_percentage=0.1)
                create_yaml(class_names, 'yolo_train_data')
                st.success("Data saved successfully!")

                model = YOLO("yolov8n.pt")
                model.train(data="/Users/abhinavkrishna/Desktop/AutoML/yolo_train_data/data.yaml", epochs=5) 
                st.success("training succesfull successfully!")

    
    def detect_objects(model_path, processed_image):
        # Perform object detection using the pre-trained YOLO model
        model = YOLO(model_path)
        results = model.predict(source=processed_image)
        boxes = results[0].boxes
        res_plotted = results[0].plot()[:, :, ::-1]
        return res_plotted , boxes
    
    def detect_objects_video(model_path, video_file):
        pass

    def detect_segment():
        st.title("Object Detection using YOLOv8")
        uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "png", "mp4"])
        detect_button = st.button("Detect Objects")

        if detect_button and uploaded_file:
            # Path to your best.pt model file
            model_path = '/Users/abhinavkrishna/Desktop/AutoML/runs/detect/train/weights/best.pt'  
            
            uploaded_file_type = uploaded_file.type
            if uploaded_file_type.startswith('image'):
                uploaded_data = Image.open(uploaded_file)
                res_plotted = detect_objects(model_path, uploaded_data)
            elif uploaded_file_type.startswith('video'):
                #res_plotted = detect_objects_video(model_path, uploaded_file)
                pass
            else:
                st.error("Unsupported file format. Please upload an image (jpg/png) or an mp4 video.")
                return

            st.image(res_plotted,caption='Detected Image', use_column_width=True)
    
                

    # Sidebar navigation
    st.sidebar.title("Navigation")
    segment_choice = st.sidebar.radio("Go to", ["Upload", "Detect"])

    # Main app
    if segment_choice == "Upload":
        upload_segment()

    if segment_choice == "Detect":
        detect_segment()