import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import cv2
import tensorflow as tf


# # Wczytanie danych i przetworzenie obrazów
def load_data(data_dir, img_size):
    image_data = []
    labels = []
    for label, class_dir in enumerate(sorted(os.listdir(data_dir), key=lambda x: int(x))):
        class_path = os.path.join(data_dir, class_dir)
        print(f"Przetwarzanie katalogu: {class_path}")
        for image_file in os.listdir(class_path):
            if image_file.endswith('.ppm'):
                img = cv2.imread(os.path.join(class_path, image_file))
                if img is not None:
                    img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                    image_data.append(img_resized)
                    labels.append(label)
                else:
                    print(f"Nie można wczytać obrazu: {os.path.join(class_path, image_file)}")
            else:
                print(f"Pomijanie pliku: {os.path.join(class_path, image_file)}")
    return np.array(image_data), np.array(labels)


data_dir = "Final_Training/Images"
img_size = (32, 32)
X, y = load_data(data_dir, img_size)

X = X.astype('float32') / 255.0
y = to_categorical(y, num_classes=43)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Definicja modelu sieci CNN
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# input_shape = (32, 32, 3)
# num_classes = len(np.unique(y))
# model = create_cnn_model(input_shape, num_classes)

num_classes = 43
input_shape = (32, 32, 3)
model = create_cnn_model(input_shape, num_classes)

# Kompilacja i uczenie modelu
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, y_test))

# Zapisz model do pliku
model.save('cnn_model.h5')

# Zapisz historię treningu do pliku CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)


# Ewaluacja modelu
def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


plot_history(history)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {test_acc}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion matrix:\n", conf_matrix)

# Zapisz macierz pomyłek do pliku CSV
conf_matrix_df = pd.DataFrame(conf_matrix)
conf_matrix_df.to_csv('confusion_matrix.csv', index=False)