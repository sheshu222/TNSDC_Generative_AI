import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories for the training data
train_dir = './data'  # Replace with the directory containing training images

# Create an ImageDataGenerator for preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)

# Load training images from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the ImageDataGenerator
model.fit(train_generator, epochs=10)

# Load and preprocess a single test image
test_image_path = 'test_img/Test.jpeg'  # Replace with the path to your test image
test_image = load_img(test_image_path, target_size=(32, 32))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Normalize pixel values

# Make predictions on the test image
predictions = model.predict(test_image)

# Display predictions
label = ["apple","banana","cherry","Chickoo","grapes","kiwi","mango","orange","strawberry"]
predicted_class = np.argmax(predictions[0])
print("Predicted class:", label[predicted_class])