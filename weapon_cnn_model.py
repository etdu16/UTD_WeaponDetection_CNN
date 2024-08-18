import tensorflow as tf
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model(
    r"C:\Users\aupik\Downloads\transfer_learningmodel_inception_ETHAN.h5"
)

class_names = ['automatic rifle', 'handgun', 'knife', 'sniper']

# Define the confidence threshold
confidence_threshold = 0.5

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

# Create a named window with specific size
cv2.namedWindow('Live Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Detection', 800, 600)  # Adjust size to your preference

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (336, 336))  # Resize according to your model input
    preprocessed_frame = resized_frame / 255.0  # Normalize if required
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Check if confidence is above the threshold
    if confidence >= confidence_threshold:
        label = class_names[predicted_class]  # Replace with your class labels
        cv2.putText(frame, f'Class: {label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No Weapon Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Live Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
