import cv2
import json
import numpy as np
import onnxruntime as ort

class Classifier:
    """
    A class to load the ONNX age/gender model and perform inference.
    """
    def __init__(self, model_path, labels_path):
        """
        Initializes the classifier by loading the model and labels.
        """
        # Load the human-readable labels from the JSON file
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)

        # Create an ONNX Runtime inference session
        self.session = ort.InferenceSession(model_path)

        # Get the model's expected input shape and name
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape

    def preprocess(self, face_image):
        """
        Prepares a single face image to be fed into the model.
        """
        # Get the required height and width from the model's input shape
        _, _, height, width = self.input_shape

        # Resize the image to the required dimensions
        resized_image = cv2.resize(face_image, (width, height))
        
        # Convert image from BGR (OpenCV's default) to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Convert the data type to float32 and normalize pixel values to be between 0 and 1
        normalized_image = rgb_image.astype(np.float32) / 255.0

        # Transpose the dimensions from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        transposed_image = np.transpose(normalized_image, (2, 0, 1))

        # Add a batch dimension to make it NCHW (Batch, Channel, Height, Width)
        # The model expects a batch of images, even if we're only sending one.
        return np.expand_dims(transposed_image, axis=0)

    def predict(self, face_image):
        """
        Runs the full prediction pipeline on a single face image.
        """
        # 1. Preprocess the input image
        processed_image = self.preprocess(face_image)

        # 2. Run inference using the ONNX session
        outputs = self.session.run(None, {self.input_name: processed_image})
        
        # 3. Process the model's output
        # The model outputs raw scores (logits). We use a softmax function
        # to convert them into probabilities that sum to 1.0.
        gender_logits = outputs[0][0]
        age_logits = outputs[1][0]
        
        gender_probs = np.exp(gender_logits) / np.sum(np.exp(gender_logits))
        age_probs = np.exp(age_logits) / np.sum(np.exp(age_logits))

        # 4. Find the most likely prediction
        gender_index = np.argmax(gender_probs)
        age_index = np.argmax(age_probs)

        # 5. Get the human-readable labels and confidence scores
        predicted_gender = self.labels['gender'][gender_index]
        gender_confidence = float(gender_probs[gender_index])
        
        predicted_age = self.labels['age'][age_index]
        age_confidence = float(age_probs[age_index])
        
        return predicted_gender, gender_confidence, predicted_age, age_confidence