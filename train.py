# ==============================
# IMPORT LIBRARIES
# ==============================
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# LOAD DATASET
# ==============================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ==============================
# NORMALIZE DATA (IMPORTANT FIX)
# ==============================
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ==============================
# RESHAPE FOR CNN
# ==============================
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ==============================
# ONE HOT ENCODING
# ==============================
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ==============================
# BUILD IMPROVED CNN MODEL
# ==============================
model = Sequential()

# Block 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

# Block 2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

# Block 3 (extra improvement)
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

# Flatten
model.add(Flatten())

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(10, activation='softmax'))

# ==============================
# COMPILE MODEL
# ==============================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# EARLY STOPPING (IMPORTANT)
# ==============================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# ==============================
# TRAIN MODEL
# ==============================
history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# EVALUATE MODEL
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(f"\n🔥 Final Test Accuracy: {test_accuracy * 100:.2f}%")


# SAVE MODEL

model.save("digit_model.h5")

print("✅ Model saved successfully as digit_model.h5")