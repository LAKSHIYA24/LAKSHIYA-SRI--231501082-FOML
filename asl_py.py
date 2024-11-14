import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define dataset directory
DATASET_DIR =r"C:\asl" # Change this to your dataset path
CATEGORIES = [str(i) for i in range(10)]  # 0 to 9
IMG_SIZE = 64  # Define the image size   

data = []
labels = []

# Load dataset
for category in CATEGORIES:
    path = os.path.join(DATASET_DIR, category)  # Construct the path
    class_num = CATEGORIES.index(category)  # Label for the category
    
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized_array)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image: {e}")

# Convert data to numpy arrays and normalize
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(CATEGORIES), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_cnn_model()

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Uncomment if you want to save the model after training
# model.save('asl_model.h5')

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to HSV (Hue, Saturation, Value) for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color detection
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply some morphology to clean up the noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Find contours in the mask to detect hand
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour, assuming it's the hand
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle around the largest contour (hand)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw a rectangle around the detected hand
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
        
        # Extract the region of interest (ROI) from the frame
        roi = frame[y:y+h, x:x+w]
        
        # Preprocess the ROI for the model
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))
        normalized_frame = resized_frame / 255.0
        reshaped_frame = normalized_frame.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        
        # Make predictions
        predictions = model.predict(reshaped_frame)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions)

        # Display the prediction result on the frame
        cv2.putText(frame, f'Predicted: {CATEGORIES[class_index]} ({confidence:.2f})', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the frame
    cv2.imshow('ASL Sign Language Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
