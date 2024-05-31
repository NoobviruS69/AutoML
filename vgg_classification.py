import streamlit as st
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout , BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model

def sub_app_1():
#side bar with app segment choices
    with st.sidebar: 
        st.title("DataLensAI")
        segment_choice = st.radio("Navigation", ["Upload","Training", "Detect"])
        st.info("This project application helps you build and explore your data.")


    def folder_upload():
        st.write("### Upload Images for Different Classes")
        
        class_folders = {}
        class_count = st.session_state.get('class_count', 2) 

        for i in range(min(class_count, 5)):
            class_name = st.text_input(f"Class {i + 1} Name")
            class_files = st.file_uploader(f"Upload {class_name} images", type=["jpg", "png"], accept_multiple_files=True, key=f"class_{i}")
            if class_name and class_files:
                class_folders[class_name] = class_files

        return class_folders

    def save_images(class_images):
        project_dir = os.path.dirname('/Users/abhinavkrishna/Desktop/AutoML/')  # Change this to your project directory
        user_images_dir = os.path.join(project_dir, 'user_images')

        if not os.path.exists(user_images_dir):
            os.makedirs(user_images_dir)

        for class_name, files in class_images.items():
            class_dir = os.path.join(user_images_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            for i, file in enumerate(files):
                image = Image.open(file)
                try:
                    image.save(os.path.join(class_dir, f"{class_name}_{i}.jpg"))
                except (OSError) as e:
                    st.warning(f"Skipping file '{file.name}' due to: {str(e)}")

    # image upload segment
    if segment_choice == "Upload":
        st.title('Image Recognition App with Folder Uploads')
        class_images = folder_upload()
        # Add Class button to dynamically add folder upload boxes
        if st.button("Add Class"):
            st.session_state.class_count = st.session_state.get('class_count', 2) + 1
        if st.button("Upload Images"):
            save_images(class_images)
            st.success("Images saved successfully!")

    if segment_choice == "Training":
        st.title('Model Training')
        user_images_dir = '/Users/abhinavkrishna/Desktop/AutoML/user_images'
        num_classes = len(os.listdir(user_images_dir))
        image_gen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values to [0, 1]
            rotation_range=20,  # Rotate images randomly within the specified range
            width_shift_range=0.1,  # Shift images horizontally by a fraction of total width
            height_shift_range=0.1,  # Shift images vertically by a fraction of total height
            shear_range=0.1,  # Apply shear transformation
            zoom_range=0.1,  # Zoom in/out on images
            horizontal_flip=True,  # Flip images horizontally
            validation_split=0.2  # Split the data into training/validation sets
        )

        # Generating batches of tensor image data from the directory
        data_generator = image_gen.flow_from_directory(
            user_images_dir,
            target_size=(150, 150),  # Resize images to a fixed size
            batch_size=32,  # Set batch size as needed
            class_mode='categorical',  # Adjust class mode based on your requirement
            subset='training'  # Specify 'training' for generating training data
        )
        validation_data_generator = image_gen.flow_from_directory(
        user_images_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'  
        )
        def train_model():
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
            for layer in base_model.layers:
                layer.trainable = False

            x = Flatten()(base_model.output)
            x = Dense(256, activation='relu')(x)
            output = Dense(num_classes, activation='softmax')(x) 

            model = Model(inputs=base_model.input, outputs=output)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



        # Training the model using data generator
            model.fit(data_generator, epochs=10 , steps_per_epoch=len(data_generator), verbose=1)
            model.save('trained_model.h5')

        if st.button("Start Training"):
            epochs = 10  # Define the number of epochs
            train_model()
            st.success("Training completed. Model saved as 'trained_model.h5'")
            model = load_model('trained_model.h5')
            evaluation = model.evaluate(validation_data_generator)
            print("Validation Accuracy:", evaluation[1])

    if segment_choice == "Detect":
        st.title('Image Classification')
        model = load_model('trained_model.h5')
        user_images_dir = '/Users/abhinavkrishna/Desktop/AutoML/user_images'
        class_names = sorted(os.listdir(user_images_dir))
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            # Preprocess the uploaded image to match model input
            img = img.resize((150, 150))  # Resize the image to match the input shape used during training
            img_array = np.array(img)  # Convert image to numpy array
            img_array = img_array / 255.0  # Normalize pixel values

            # Expand dimensions to match model input shape (batch size of 1)
            img_array = np.expand_dims(img_array, axis=0)

            # Make predictions using the loaded model
            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction)
            predicted_class = class_names[predicted_class_idx]

            # Display the prediction results
            st.write("### Prediction:")
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Confidence: {prediction[0][predicted_class_idx]:.4f}")