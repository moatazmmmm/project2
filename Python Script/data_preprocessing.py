# data_preprocessing.py

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # Split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.30, random_state=42, stratify=y_all)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # Normalize
    X_train, X_val, X_test = X_train/255.0, X_val/255.0, X_test/255.0

    # One-hot encode
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    datagen.fit(X_train)

    # Generators
    train_gen = datagen.flow(X_train, y_train, batch_size=64)
    val_gen = ImageDataGenerator().flow(X_val, y_val, batch_size=64)
    test_gen = ImageDataGenerator().flow(X_test, y_test, batch_size=64, shuffle=False)

    return train_gen, val_gen, test_gen

if __name__ == "__main__":
    train_gen, val_gen, test_gen = load_and_preprocess_data()
    print("✅ Data Generators Ready for Training")
