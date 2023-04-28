import threading
import sys
import tkinter as tk
from tkinter import filedialog, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import cv2
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import queue


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


def process_data(queue, history_queue):
    img_size = (32, 32)
    X, y = load_data(data_dir, img_size)

    X = X.astype('float32') / 255.0
    y = to_categorical(y, num_classes=43)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_classes = 43
    input_shape = (32, 32, 3)
    model = create_cnn_model(input_shape, num_classes)

    # Kompilacja i uczenie modelu
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, y_test))
    history_queue.put(history)

    plot_history(history)

    # Zapisz model do pliku
    model.save('cnn_model.h5')

    # Zapisz historię treningu do pliku CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('training_history.csv', index=False)

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


def redirect_output(text_widget):
    class TextRedirector:
        def __init__(self, widget):
            self.widget = widget

        def write(self, text):
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)

        def flush(self):
            pass

    sys.stdout = TextRedirector(text_widget)


def process_data_thread(queue, history_queue):
    thread = threading.Thread(target=process_data, args=(queue, history_queue))
    thread.start()


def update_output_text(text_widget, queue):
    while True:
        text = queue.get()
        text_widget.insert(tk.END, text)
        text_widget.see(tk.END)


def browse_button():
    global data_dir
    data_dir = filedialog.askdirectory()
    print("Selected directory:", data_dir)
    process_data_thread()


def browse_directory(queue):
    global data_dir
    data_dir = filedialog.askdirectory()
    queue.put("Selected directory: " + data_dir)
    process_data_thread(queue)


def display_plots(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Test'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Test'], loc='upper left')

    fig.tight_layout()

    plot_window = tk.Toplevel()
    plot_window.title("Training Plots")
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


def main():
    root = tk.Tk()
    root.title("CNN Image Classifier")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    output_queue = queue.Queue()

    history_queue = queue.Queue()
    browse_button = tk.Button(frame, text="Browse for Image Directory",
                              command=lambda: process_data_thread(output_queue, history_queue))
    browse_button.pack()

    plot_label = tk.Label(root, text="Train model first before click \"Show Training Plots\" ")
    plot_label.pack()

    plot_button = tk.Button(frame, text="Show Training Plots", command=lambda: display_plots(history_queue.get()))
    plot_button.pack()

    output_frame = tk.Frame(root)
    output_frame.pack(padx=10, pady=10)

    output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=60, height=20)
    output_text.pack()

    redirect_output(output_text)

    update_thread = threading.Thread(target=update_output_text, args=(output_text, output_queue))
    update_thread.daemon = True
    update_thread.start()

    root.mainloop()


if __name__ == "__main__":
    main()
