from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Load your trained model
model = load_model('trained_model.h5')

user_images_dir = '/Users/abhinavkrishna/Desktop/AutoML/user_images'
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

validation_data_generator = image_gen.flow_from_directory(
    user_images_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  
    )

# Evaluate the model on the validation dataset
evaluation = model.evaluate(validation_data_generator)

print("Validation Accuracy:", evaluation[1])