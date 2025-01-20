import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle

# Load the model
@st.cache_resource  # Cache the model for better performance
def load_trained_model():
    return load_model('model.h5')

model = load_trained_model()

# Load the class indices
@st.cache_resource
def load_class_indices():
    with open('class_indices.pkl', 'rb') as f:
        return pickle.load(f)

class_indices = load_class_indices()

# Precautions dictionary for each disease
disease_precautions = {
    "Apple___Apple_scab": "Use fungicides and remove infected leaves. Maintain dry conditions during the growing season.",
    "Apple___Black_rot": "Prune and destroy infected plant parts. Apply appropriate fungicides.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees or treat them with fungicides. Use rust-resistant apple varieties.",
    "Tomato___Late_blight": "Avoid overhead watering. Use disease-resistant varieties and apply copper-based fungicides.",
    "Potato___Early_blight": "Ensure proper crop rotation. Remove and destroy infected foliage.",
    "Strawberry___Leaf_scorch": "Avoid overhead irrigation. Remove and destroy infected leaves. Apply fungicides if necessary.",
    # Add precautions for other diseases...
    "Tomato___healthy": "No precautions needed. Keep up good agricultural practices!"
}

# Plant emojis dictionary
plant_emojis = {
    "Apple": "🍎",
    "Blueberry": "🫐",
    "Cherry (including sour)": "🍒",
    "Tomato": "🍅",
    "Peach": "🍑",
    "Potato": "🥔",
    "Strawberry": "🍓",
    "Orange": "🍊",
    "Raspberry": "🍇",
    "Corn (maize)": "🌽",
}

# Extract unique plant names from class indices
def get_unique_plants(class_indices):
    plant_names = set()
    for disease in class_indices.values():
        plant_name = disease.split("___")[0].replace("_", " ")  # Extract plant name
        plant_names.add(plant_name)
    return sorted(plant_names)

supported_plants = get_unique_plants(class_indices)

# Define preprocessing for the image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)  # Resize image
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize
    return image_array

# App Layout
st.title("🌱 Plant Disease Detection Web App")
st.markdown("Upload an image of a plant leaf, and this app will predict the disease (if any).")

# New Section: Supported Plants
st.markdown("## 🌿 Supported Plants")
st.write("The model can analyze diseases in the following plants:")

# Display plants in a 2 x 7 matrix format with emojis
columns = st.columns(2)  # Create 2 columns
plants_per_column = 7  # Number of plants per column

# Divide plants into two groups and display
for i, plant in enumerate(supported_plants):
    col_index = i // plants_per_column  # Determine column (0 or 1)
    emoji = plant_emojis.get(plant, "🌱")  # Default emoji if not found
    with columns[col_index]:
        st.write(f"{emoji} {plant}")

# Upload Image Section
st.markdown("---")
uploaded_file = st.file_uploader("Choose an image of a plant leaf (JPG, JPEG, or PNG format)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image with reduced size
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image (Resized for Prediction)", width=200)  # Reduced size

    # Add a button for processing
    if st.button("🔍 Detect Disease"):
        st.write("Processing the image...")
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices[predicted_class_index]
        prediction_confidence = np.max(predictions)

        # Display the result
        st.write(f"### Predicted Class: `{predicted_class_name}`")
        st.write(f"### Confidence: `{prediction_confidence * 100:.2f}%`")

        # Display precautions if available
        if predicted_class_name in disease_precautions:
            st.markdown(f"### Precautions: {disease_precautions[predicted_class_name]}")
        else:
            st.markdown("### Precautions: Information not available.")

        # Add additional guidance
        st.markdown(
            """
            ### Next Steps:
            - If the plant is diseased, consider applying appropriate remedies.
            - If it's healthy, keep up the good work! 🌿
            """
        )
else:
    st.info("Upload an image to start the prediction process.")

# Footer Section
st.markdown("---")
# st.markdown("👨‍💻 Developed by [Prashant Kumar ](https://www.linkedin.com/in/prashant-kumar-62b76024a/) | 🌟 Powered by TensorFlow and Streamlit")
