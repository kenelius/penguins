import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, auc)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Iris Species Classification Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Functions
@st.cache_resource
def load_models():
    """Load pre-trained models and artifacts."""
    try:
        models = {
            "Logistic Regression": joblib.load('model_lr.pkl'),
            "K-Nearest Neighbors": joblib.load('model_knn.pkl'),
            "Random Forest": joblib.load('model_rf.pkl'),
            "XGBoost": joblib.load('model_xgb.pkl')
        }
        le = joblib.load('label_encoder.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return models, le, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()

def get_recommendation(probabilities, class_names):
    """Generate recommendation based on prediction probabilities."""
    pred_idx = np.argmax(probabilities)
    pred_prob = probabilities[pred_idx] * 100
    
    if pred_prob <= 30:
        level, color, advice = "Low Confidence", "#ff4b4b", "The model is uncertain. Please verify input data."
    elif 31 <= pred_prob <= 50:
        level, color, advice = "Moderate Confidence", "#ffa500", "Prediction is tentative."
    elif 51 <= pred_prob <= 70:
        level, color, advice = "High Confidence", "#ffff00", "Likely correct prediction."
    else:
        level, color, advice = "Very High Confidence", "#00cc00", "Strong prediction."

    return class_names[pred_idx], pred_prob, level, color, advice

# Main App
def main():
    st.sidebar.title("Navigation")
    menu = ["Home", "Prediction", "About"]
    choice = st.sidebar.radio("Go to", menu)
    
    # Load models and data
    models, le, feature_names = load_models()
    data = pd.read_csv("IRIS.csv")
    class_names = le.classes_
    
    if choice == "Home":
        st.title("🌸 Iris Species Classification Dashboard")
        st.markdown("Welcome to the Iris Classification System.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(data))
        col2.metric("Features", len(feature_names))
        col3.metric("Classes", len(class_names))
        
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

    elif choice == "Prediction":
        st.title("🔮 Make a Prediction")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_model_name = st.selectbox("Select Model", list(models.keys()))
            sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
            sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
            petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
            petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
            predict_btn = st.button("Predict Species", type="primary")
            
        with col2:
            if predict_btn:
                input_data = pd.DataFrame({
                    'sepal_length': [sepal_length], 'sepal_width': [sepal_width],
                    'petal_length': [petal_length], 'petal_width': [petal_width]
                })
                
                model = models[selected_model_name]
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                predicted_species = le.inverse_transform([prediction])[0]
                
                pred_class, pred_prob, conf_level, color, advice = get_recommendation(probabilities, class_names)
                
                st.subheader("Prediction Result")
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; border-left: 5px solid {color};">
                    <h3 style="margin:0;">{predicted_species}</h3>
                    <p style="margin:5px 0;"><strong>Confidence:</strong> {pred_prob:.2f}%</p>
                    <p style="margin:5px 0; color: {color};"><strong>Status:</strong> {conf_level}</p>
                </div>
                """, unsafe_allow_html=True)
                
                prob_df = pd.DataFrame({'Species': class_names, 'Probability': probabilities})
                st.bar_chart(prob_df.set_index('Species'))

    elif choice == "About":
        st.title("️ About")
        st.markdown("This dashboard classifies Iris flowers using Machine Learning.")

if __name__ == '__main__':
    main()
