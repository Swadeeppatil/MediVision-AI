import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input

class ModelHandler:
    def __init__(self):
        self.fracture_types = {
            "transverse": {
                "description": "Clean break across the bone, typically caused by direct high-energy trauma.",
                "severity": "Moderate",
                "treatment": "1. Immobilization via splinting/casting\n2. Reduction if displaced\n3. Regular X-ray monitoring"
            },
            "oblique": {
                "description": "Angled break across the long axis of the bone, common in twisting injuries.",
                "severity": "Moderate to Severe",
                "treatment": "1. Orthopedic surgical evaluation\n2. Possible internal fixation (pins/plates)\n3. Targeted physical therapy"
            },
            "compound": {
                "description": "Open fracture where bone fragment pierces through the surrounding skin tissue.",
                "severity": "Severe (Medical Emergency)",
                "treatment": "1. Immediate emergency surgical intervention\n2. Broad-spectrum intravenous antibiotics\n3. Surgical debridement & wound care"
            },
            "stress": {
                "description": "Microscopic cracks in bone structure resulting from repetitive force or overuse.",
                "severity": "Mild to Moderate",
                "treatment": "1. Rest and strict non-weight-bearing\n2. Activity modification and supportive footwear\n3. Progressive rehabilitation"
            }
        }
        self.model = None

    def load_model(self):
        if self.model is None:
            base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            predictions = tf.keras.layers.Dense(len(self.fracture_types), activation='softmax')(x)
            self.model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        return self.model

    def predict(self, image_path):
        if self.model is None:
            self.load_model()
            
        img_data = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img_data)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = self.model.predict(x, verbose=0)
        idx = np.argmax(preds[0])
        fracture_key = list(self.fracture_types.keys())[idx]
        confidence = float(np.max(preds[0]) * 100)
        
        return fracture_key, confidence, self.fracture_types[fracture_key]
