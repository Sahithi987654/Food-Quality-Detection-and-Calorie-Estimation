import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hides all the messy terminal logs

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image

# --- DATA ---
my_10_classes = ['apple_pie', 'beef_tartare', 'caesar_salad', 'cheesecake', 
                 'chicken_wings', 'french_fries', 'hamburger', 'pizza', 'sushi', 'waffles']

food_info = {
    'apple_pie': {'calories': 237, 'protein': '1.9g', 'fat': '11g', 'carbs': '34g'},
    'beef_tartare': {'calories': 150, 'protein': '20g', 'fat': '8g', 'carbs': '1g'},
    'caesar_salad': {'calories': 44, 'protein': '2.1g', 'fat': '2.1g', 'carbs': '4.6g'},
    'cheesecake': {'calories': 321, 'protein': '5.5g', 'fat': '23g', 'carbs': '26g'},
    'chicken_wings': {'calories': 203, 'protein': '18g', 'fat': '14g', 'carbs': '0g'},
    'french_fries': {'calories': 312, 'protein': '3.4g', 'fat': '15g', 'carbs': '41g'},
    'hamburger': {'calories': 250, 'protein': '13g', 'fat': '10g', 'carbs': '27g'},
    'pizza': {'calories': 266, 'protein': '11g', 'fat': '10g', 'carbs': '33g'},
    'sushi': {'calories': 143, 'protein': '4g', 'fat': '0.7g', 'carbs': '30g'},
    'waffles': {'calories': 291, 'protein': '8g', 'fat': '14g', 'carbs': '33g'}
}

@st.cache_resource
def load_my_model():
    model_path = 'food_model.keras'
    try:
        # We try to load it normally first
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        # RECONSTRUCTION MODE: Now matching your specific Colab architecture
        model = tf.keras.Sequential([
            tf.keras.applications.EfficientNetB0(
                weights=None, 
                include_top=False, 
                input_shape=(224, 224, 3)
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            # THIS IS THE MISSING PIECE: The 256-neuron hidden layer
            tf.keras.layers.Dense(256, activation='relu'), 
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.build((None, 224, 224, 3))
        # Load weights - no skip_mismatch needed now because the shapes match!
        model.load_weights(model_path)
        return model
# --- ANALYSIS HELPERS ---
def analyze_quality(img_np, confidence):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = np.sum(brown_mask > 0) / (img_np.shape[0] * img_np.shape[1])
    
    if brown_ratio > 0.15: return f"⚠️ Low Quality ({brown_ratio:.1%})"
    if confidence < 0.60: return "❓ Average"
    return "✅ Fresh"

# --- UI LAYOUT ---
st.set_page_config(page_title="AI Food Lab", layout="wide")

# Custom CSS to keep things compact
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stMetric { background: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🍔 AI Food Lab")

model = load_my_model()
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Main Dashboard Columns
    col_img, col_res = st.columns([1, 1.5])
    
    with col_img:
        # Resize display image so it doesn't take the whole screen
        st.image(img, caption="Scanned Item", width=400)

    with col_res:
        # Prediction
        img_rgb=img.convert('RGB')
        img_input = img_rgb.resize((224, 224))
        img_arr = image.img_to_array(img_input)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = tf.keras.applications.efficientnet.preprocess_input(img_arr)
        
        preds = model.predict(img_arr, verbose=0)
        food_name = my_10_classes[np.argmax(preds)]
        conf = np.max(preds)
        quality = analyze_quality(np.array(img), conf)
        if "Low Quality" in quality:
            st.error("🚨 WARNING: This food item shows significant discoloration. Consumption is not recommended.")
        else:
            st.success("🍏 This item looks fresh and safe to eat!")
        
        # Results Section
        st.subheader(f"Detection: {food_name.title().replace('_', ' ')}")
        
        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{conf*100:.1f}%")
        m2.metric("Freshness", quality)
        
        st.write("---")
        st.subheader("Nutritional Stats (100g)")
        info = food_info.get(food_name, {})
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Cals", f"{info['calories']}")
        n2.metric("Prot", info['protein'])
        n3.metric("Fat", info['fat'])
        n4.metric("Carb", info['carbs'])

        import pandas as pd
        
        # Strip the 'g' and convert to float for the chart
        chart_data = pd.DataFrame({
            'Nutrient': ['Protein', 'Fat', 'Carbs'],
            'Amount': [
                float(info['protein'].replace('g','')), 
                float(info['fat'].replace('g','')), 
                float(info['carbs'].replace('g',''))
            ]
        })
        st.vega_lite_chart(chart_data, {
            'mark': {'type': 'arc', 'innerRadius': 50},
            'encoding': {
                'theta': {'field': 'Amount', 'type': 'quantitative'},
                'color': {'field': 'Nutrient', 'type': 'nominal'},
            },
        }, use_container_width=True)

else:
    st.write("👈 Upload a photo in the sidebar to begin analysis.")