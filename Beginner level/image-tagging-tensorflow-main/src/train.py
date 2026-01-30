import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = data.flow_from_directory(
    "data/train",
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

test_data = data.flow_from_directory(
    "data/train",
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, validation_data=test_data, epochs=5)

model.save("models/image_model.h5")
print("DONE! Model saved")
