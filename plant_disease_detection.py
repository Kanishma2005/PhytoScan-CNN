import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3  # Import the sqlite3 module
import sys  # Import the sys module


def load_and_preprocess_data(data_dir, image_size=(224, 224), test_size=0.2, random_state=42):
    """
    Loads, resizes, and preprocesses images from a directory, and splits the data.

    Args:
        data_dir (str): Path to the directory containing the image data.
        image_size (tuple, optional): The desired size of the images (width, height).
            Defaults to (224, 224).
        test_size (float, optional): The proportion of data to use for the test set.
            Defaults to 0.2 (20%).  The remaining data is used for training.
        random_state (int, optional): Random state for data splitting.  Defaults to 42.

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels, class_names)
            -     train_images:  NumPy array of training images, normalized to [0, 1].
            -     train_labels:  NumPy array of one-hot encoded training labels.
            -     test_images:   NumPy array of test images, normalized to [0, 1].
            -     test_labels:   NumPy array of one-hot encoded test labels.
            -     class_names:   List of class names (folder names).
    """
    images = []
    labels = []
    class_names = []

    try:
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.isdir(folder_path):
                class_names.append(folder_name)
                for filename in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, image_size)
                            images.append(img)
                            labels.append(folder_name)
                        else:
                            print(f"Error: Could not read image {img_path}")
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")

    except FileNotFoundError:
        print(f"Error: Directory not found: {data_dir}")
        return [], [], [], [], []

    if not images:
        print(f"No images found in the directory: {data_dir}")
        return [], [], [], [], []

    images = np.array(images)
    labels = np.array(labels)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    label_to_id = {label: i for i, label in enumerate(class_names)}
    train_labels_numeric = np.array([label_to_id[label] for label in train_labels])
    test_labels_numeric = np.array([label_to_id[label] for label in test_labels])

    train_labels_one_hot = to_categorical(train_labels_numeric)
    test_labels_one_hot = to_categorical(test_labels_numeric)

    return train_images, train_labels_one_hot, test_images, test_labels_one_hot, class_names


def create_cnn_model(input_shape, num_classes):
    """
    Creates a Convolutional Neural Network (CNN) model for image classification.
    This is a more complex model, good for real-world scenarios.

    Args:
        input_shape (tuple): Shape of the input images (e.g., (224, 224, 3)).
        num_classes (int): Number of classes.

    Returns:
        tf.keras.Model: A Keras model instance.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(model, train_images, train_labels, val_images, val_labels, class_names, epochs=20, batch_size=32):
    """
    Trains the CNN model and stores training history in an SQLite database.

    Args:
        model (tf.keras.Model): The model to train.
        train_images (np.ndarray): Training images.
        train_labels (np.ndarray): One-hot encoded training labels.
        val_images (np.ndarray): Validation images.
        val_labels (np.ndarray): One-hot encoded validation labels.
        class_names (list): List of class names.
        epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 32.

    Returns:
        tf.keras.Model: The trained model.
        History:
    """
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # Create or connect to an SQLite database
    conn = sqlite3.connect('training_history.db')
    cursor = conn.cursor()

    # Create a table to store training history if it doesn't exist
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                epoch INTEGER,
                accuracy REAL,
                loss REAL,
                val_accuracy REAL,
                val_loss REAL,
                class_name TEXT,
                PRIMARY KEY (epoch, class_name)
            )
        """)
        conn.commit()
        print("SQLite: Created table training_history (if not exists)")  # Added print
    except Exception as e:
        print(f"SQLite Error creating table: {e}")
        conn.close()
        sys.exit(1)

    history = model.fit(train_images, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(val_images, val_labels),
                        callbacks=[early_stopping],
                        verbose=1)  # Ensure verbose is set to 1 to see training progress

    # Store training history in the database, handling potential duplicates
    for epoch, (acc, loss, val_acc, val_loss) in enumerate(zip(
            history.history['accuracy'], history.history['loss'],
            history.history['val_accuracy'], history.history['val_loss'])):
        for class_name in class_names:  # Store history for each class
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO training_history (epoch, accuracy, loss, val_accuracy, val_loss, class_name) VALUES (?, ?, ?, ?, ?, ?)",
                    (epoch + 1, acc, loss, val_acc, val_loss, class_name)
                )
                conn.commit()  # Commit the data insertion
                print(f"SQLite: Inserted/Updated data for epoch {epoch + 1}, class {class_name}")  # Added print
            except Exception as e:
                print(f"SQLite Error inserting/updating data: {e}")  # Print any other exception
                print(f"SQLite: Failed to insert/update data for epoch {epoch + 1}, class {class_name}")  # Added print
                conn.close()
                sys.exit(1)

    # Close the database connection
    conn.close()
    print("SQLite: Closed connection")  # Added print

    return model, history


def evaluate_model(model, test_images, test_labels, class_names):
    """
    Evaluates the CNN model and prints results.

    Args:
        model (tf.keras.Model): The trained model.
        test_images (np.ndarray): Test images.
        test_labels (np.ndarray): One-hot encoded test labels.
        class_names (list): List of class names.
    """
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    data_dir = 'plant_disease_dataset'  # Replace with the path to your dataset. USE THE CORRECT PATH.
    image_size = (224, 224)
    test_size = 0.2
    random_state = 42  # For reproducibility

    train_images, train_labels, test_images, test_labels, class_names = load_and_preprocess_data(
        data_dir, image_size, test_size, random_state
    )
    # Check if any images were loaded
    if len(train_images) > 0:  # Changed from train_images.size
        print(f"Loaded {len(train_images)} training images and {len(test_images)} test images.")
        print(f"Class names: {class_names}")

        input_shape = train_images.shape[1:]
        num_classes = len(class_names)

        model = create_cnn_model(input_shape, num_classes)
        model.summary()  # Display the model architecture

        # Split training data into training and validation sets
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=0.1, random_state=random_state, stratify=train_labels
        )
        trained_model, history = train_model(model, train_images, train_labels, val_images, val_labels, class_names, epochs=20)
        evaluate_model(trained_model, test_images, test_labels, class_names)

        # Save the trained model
        model_save_path = 'plant_disease_detection.h5'
        trained_model.save(model_save_path)
        print(f"Trained model saved to: {model_save_path}")

    else:
        print("Failed to load data. Check the dataset path and folder structure.")
        sys.exit()  # Exit if no data is loaded.