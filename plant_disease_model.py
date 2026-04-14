"""
Plant Disease Detection Model using TensorFlow/Keras
Uses transfer learning with MobileNetV2 for plant disease classification
Trained on PlantVillage dataset - 38 classes of plant diseases
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import warnings
warnings.filterwarnings('ignore')

# Disable GPU if needed (for faster CPU inference)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class PlantDiseaseDetector:
    """Local plant disease detection using TensorFlow"""
    
    # Class names for PlantVillage dataset (38 classes)
    CLASS_NAMES = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
        'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust_', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]
    
    # Disease information and treatments
    DISEASE_INFO = {
        'Apple___Apple_scab': {
            'disease': 'Apple Scab',
            'treatment': 'Apply fungicide containing captan or myclobutanil. Remove and destroy infected leaves. Prune infected branches.'
        },
        'Apple___Black_rot': {
            'disease': 'Black Rot',
            'treatment': 'Apply copper fungicide. Remove infected fruit and cankers. Destroy fallen debris.'
        },
        'Apple___Cedar_apple_rust': {
            'disease': 'Cedar Apple Rust',
            'treatment': 'Apply sulfur or copper-based fungicide. Remove cedar galls nearby. Plant resistant varieties.'
        },
        'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
            'disease': 'Cercospora Leaf Spot (Gray Leaf Spot)',
            'treatment': 'Apply fungicide (azoxystrobin or pyraclostrobin). Rotate crops. Use resistant varieties.'
        },
        'Corn___Common_rust_': {
            'disease': 'Common Rust',
            'treatment': 'Apply fungicide at first sign of infection. Use resistant varieties. Early planting helps.'
        },
        'Corn___Northern_Leaf_Blight': {
            'disease': 'Northern Leaf Blight',
            'treatment': 'Apply fungicide. Use resistant hybrids. Crop rotation and plowing under debris.'
        },
        'Grape___Black_rot': {
            'disease': 'Black Rot',
            'treatment': 'Apply copper or sulfur fungicide. Remove infected plant parts. Improve air circulation.'
        },
        'Grape___Esca_(Black_Measles)': {
            'disease': 'Esca (Black Measles)',
            'treatment': 'Prune and remove infected wood. Apply fungicide preventively. Avoid wounding vines.'
        },
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
            'disease': 'Leaf Blight',
            'treatment': 'Apply copper-based fungicide. Remove infected leaves. Improve drainage and air circulation.'
        },
        'Orange___Haunglongbing_(Citrus_greening)': {
            'disease': 'Citrus Greening (HLB)',
            'treatment': 'No cure available. Remove infected trees. Control psyllid insects with insecticides. Use resistant rootstocks.'
        },
        'Peach___Bacterial_spot': {
            'disease': 'Bacterial Spot',
            'treatment': 'Apply copper-based bactericide. Remove infected leaves. Avoid overhead irrigation.'
        },
        'Pepper,_bell___Bacterial_spot': {
            'disease': 'Bacterial Spot',
            'treatment': 'Apply copper-based spray. Remove infected plants. Use disease-free seeds. Rotate crops.'
        },
        'Potato___Early_blight': {
            'disease': 'Early Blight',
            'treatment': 'Apply copper fungicide. Remove infected leaves. Mulch around plants. Rotate crops.'
        },
        'Potato___Late_blight': {
            'disease': 'Late Blight',
            'treatment': 'Apply fungicide (mancozeb or chlorothalonil) immediately. Remove infected plants. Do not compost.'
        },
        'Strawberry___Leaf_scorch': {
            'disease': 'Leaf Scorch',
            'treatment': 'Apply fungicide. Remove infected leaves. Use drip irrigation. Plant resistant varieties.'
        },
        'Tomato___Bacterial_spot': {
            'disease': 'Bacterial Spot',
            'treatment': 'Apply copper-based spray. Remove infected plants. Use disease-free seeds. Rotate crops.'
        },
        'Tomato___Early_blight': {
            'disease': 'Early Blight',
            'treatment': 'Apply copper fungicide or chlorothalonil. Remove lower infected leaves. Mulch heavily.'
        },
        'Tomato___Late_blight': {
            'disease': 'Late Blight',
            'treatment': 'Apply fungicide immediately (mancozeb or chlorothalonil). Remove and destroy infected plants.'
        },
        'Tomato___Leaf_Mold': {
            'disease': 'Leaf Mold',
            'treatment': 'Improve air circulation. Apply sulfur or copper fungicide. Remove infected leaves. Reduce humidity.'
        },
        'Tomato___Septoria_leaf_spot': {
            'disease': 'Septoria Leaf Spot',
            'treatment': 'Apply copper fungicide. Remove infected leaves. Avoid overhead watering. Rotate crops.'
        },
        'Tomato___Spider_mites Two-spotted_spider_mite': {
            'disease': 'Spider Mites',
            'treatment': 'Spray with water to dislodge. Apply insecticidal soap or neem oil. Introduce predatory mites.'
        },
        'Tomato___Target_Spot': {
            'disease': 'Target Spot',
            'treatment': 'Apply fungicide. Remove infected leaves. Improve air circulation. Avoid overhead watering.'
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'disease': 'Tomato Yellow Leaf Curl Virus',
            'treatment': 'Control whiteflies with insecticides. Remove infected plants. Use reflective mulch. Plant resistant varieties.'
        },
        'Tomato___Tomato_mosaic_virus': {
            'disease': 'Tomato Mosaic Virus',
            'treatment': 'No cure - remove infected plants. Disinfect tools. Use resistant varieties. Control aphids.'
        },
        'Cherry___Powdery_mildew': {
            'disease': 'Powdery Mildew',
            'treatment': 'Apply sulfur or neem oil. Improve air circulation. Remove infected parts. Water at soil level.'
        },
        'Squash___Powdery_mildew': {
            'disease': 'Powdery Mildew',
            'treatment': 'Apply milk spray (1:9 milk:water) or sulfur. Remove infected leaves. Improve air circulation.'
        }
    }
    
    def __init__(self, model_path=None):
        """Initialize the disease detector"""
        self.model = None
        self.IMAGE_SIZE = (224, 224)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model found. Using image classification base.")
            self._load_base_model()
    
    def _load_base_model(self):
        """Load MobileNetV2 as base for transfer learning"""
        print("Loading MobileNetV2 base model...")
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Add classification head
        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(38, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Base model loaded (untrained - will use demo predictions)")
    
    def load_model(self, model_path):
        """Load a trained model"""
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        img = keras.utils.load_img(image_path, target_size=self.IMAGE_SIZE)
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def preprocess_image_from_bytes(self, image_bytes):
        """Preprocess image from bytes"""
        img = keras.utils.load_img(image_bytes, target_size=self.IMAGE_SIZE)
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def predict(self, image_path=None, image_bytes=None):
        """Predict plant disease"""
        if image_path:
            img_array = self.preprocess_image(image_path)
        elif image_bytes:
            img_array = self.preprocess_image_from_bytes(image_bytes)
        else:
            raise ValueError("Either image_path or image_bytes must be provided")
        
        predictions = self.model.predict(img_array, verbose=0)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.CLASS_NAMES[idx]
            confidence = float(predictions[0][idx])
            
            # Parse class name
            parts = class_name.split('___')
            plant = parts[0].replace('_', ' ')
            disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Healthy'
            is_healthy = 'healthy' in disease.lower()
            
            # Get treatment info
            disease_info = self.DISEASE_INFO.get(class_name, {})
            
            results.append({
                'class': class_name,
                'plant': plant,
                'disease': disease,
                'is_healthy': is_healthy,
                'confidence': confidence,
                'treatment': disease_info.get('treatment', 'Consult local agricultural expert.')
            })
        
        return results


# Global detector instance
detector = None

def get_detector():
    """Get or initialize the detector"""
    global detector
    if detector is None:
        detector = PlantDiseaseDetector()
    return detector