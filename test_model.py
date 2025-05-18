import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Define the absolute path to your project directory
project_dir = r"C:\Users\LENOVO\Desktop\opencv_project"  # <--- MAKE SURE THIS PATH IS CORRECT!!!

# Define the paths to your model and script using the project directory
model_path = os.path.join(project_dir, "plant_disease_detection.h5")
# model_module_path = os.path.join(project_dir, "plant_disease_detection.py") # Not needed if model is only loaded
test_image_path = os.path.join(project_dir, "sample.jpg")  # <--- CHANGE THIS PATH IF NEEDED

# Add the project directory to sys.path
if project_dir not in sys.path:
    sys.path.append(project_dir)
    print(f"Added {project_dir} to sys.path")

# Print out the paths being used for debugging
print(f"Project directory: {project_dir}")
print(f"Model path: {model_path}")
# print(f"Model module path: {model_module_path}") # Not needed if model is only loaded
print(f"Test image path: {test_image_path}")
print(f"sys.path: {sys.path}")

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

# # Check if the plant_disease_model.py file exists AND is a file. # Not needed if model is only loaded
# if not os.path.isfile(model_module_path):
#     print(f"Error: plant_disease_model.py not found at {model_module_path} or is not a file.")
#     sys.exit(1)

# # Try to get the absolute path # Not needed if model is only loaded
# try:
#     model_module_path_abs = os.path.abspath(model_module_path)
#     print(f"Absolute path to plant_disease_model.py: {model_module_path_abs}")
# except Exception as e:
#     print(f"Error getting absolute path: {e}")

# # Check the contents of the directory
print("Files in project directory:")
try:
    files = os.listdir(project_dir)
    for file in files:
        print(f"- {file}")
except Exception as e:
    print(f"Error listing directory contents: {e}")

# # Now, try the import again, with a try-except block specifically for the import # Not needed if model is only loaded
# try:
#     import plant_disease_model  # Import the file where your model is defined
#     print("Successfully imported plant_disease_model")
# except ModuleNotFoundError as e:
#     print(f"Error: ModuleNotFoundError: {e}")
#     sys.exit(1)
# except Exception as e:
#     print(f"Error during import: {e}")
#     sys.exit(1)

# Load the trained model
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    sys.exit(1)

# Function to preprocess the image (same as in your training script)
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict the class of the image
def predict_class(img_path, model):
    img_array = preprocess_image(img_path)
    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]  # Get confidence
        return predicted_class, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1, 0.0  # Return -1 for error, 0 confidence

# Function to get the class name from the class index (same as in your training script)
def get_class_name(class_index):
    class_names = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
                   'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato_mosaic_virus', 'healthy']
    if 0 <= class_index < len(class_names):
        return class_names[class_index]
    else:
        return "Unknown"

def main():
    # Test with a single image
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        return

    predicted_class_index, confidence = predict_class(test_image_path, model)  # Pass the model
    if predicted_class_index != -1:
        predicted_class_name = get_class_name(predicted_class_index)
        print(f"Predicted class: {predicted_class_name} with confidence: {confidence:.4f}")

        # Display the image with the prediction (optional, requires OpenCV)
        img = cv2.imread(test_image_path)
        if img is not None:  # check if the image was read correctly.
            img = cv2.resize(img, (800, 600))  # Resize for display
            cv2.putText(img, f"Predicted: {predicted_class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(img, f"Confidence: {confidence:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.imshow("Prediction", img)
            cv2.waitKey(0)  # Wait for a key press
            cv2.destroyAllWindows()
        else:
            print(f"Error: Could not read the image at {test_image_path} using cv2.imread().")
    else:
        print("Prediction failed.")

if __name__ == "__main__":
    main()