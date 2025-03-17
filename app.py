from flask import Flask, request, jsonify
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import tensorflow as tf
from PIL import Image
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model
model = tf.keras.models.load_model("C:/Users/thoma/Nuro/Backend/BrainTumorAIFinalLarge.keras")

# Define class names
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"})
    
    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        # Open the image file
        img = Image.open(img_file)
        # Resize the image to the expected size
        img = img.resize((256, 256))  # Resize to 256x256
        img_array = np.array(img)
        
        # Ensure the image has 3 channels (RGB)
        if img_array.shape[-1] != 3:
            return jsonify({"error": "Invalid image format"})
        
        # Preprocess image: Expand dims to match model's input shape
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])  # Get the index of the highest probability
        predicted_class = class_names[predicted_class_index]  # Map index to class name
        confidence = float(np.max(prediction[0]))  # Get the confidence score
        
        # Return the prediction result
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Ensure it uses the PORT variable
    app.run(host='0.0.0.0', port=port)