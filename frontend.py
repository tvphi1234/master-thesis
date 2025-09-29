import streamlit as st
import matplotlib.cm as cm

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import CLASS_NAMES
from utils import load_model, model_predict, visualize_grad_cam


# Configuration
MODEL_PATH = "./models/xception_best_20250826.pth"

# Load model
model = load_model(model_name="xception",
                   num_classes=2,
                   model_path=MODEL_PATH,
                   is_train=False)

# Chọn layer cuối cùng của backbone (tùy mô hình, với Xception thường là 'block14_sepconv2')
target_layers = [model.act4]

# Grad-CAM
grad_cam = GradCAM(model=model, target_layers=target_layers)

# Streamlit UI
st.set_page_config(
    page_title="Cancer Classification",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Cancer Classification System")
st.write("Upload a medical image to classify as **Benign** or **Cancer**")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info(
    "This model uses Xception architecture to classify medical images. "
    "Use Grad-CAM visualization to understand which regions the model focuses on. "
    "Please consult with medical professionals for actual diagnosis."
)

st.sidebar.header("Features")
st.sidebar.markdown(
    "• **Classification**: Benign vs Cancer detection\n"
    "• **Grad-CAM**: Visual explanation of model decisions\n"
    "• **Confidence scores**: Probability for each class"
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a medical image for classification"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Input Image", use_container_width=True)

    with col2:
        st.subheader("Classification Result")

        # Add a button to trigger prediction
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                try:

                    # Make prediction
                    predicted_class, confidence, probabilities = model_predict(
                        model, image)
                    predicted_label = CLASS_NAMES[predicted_class]

                    # Display results
                    if predicted_class == 1:  # Cancer
                        st.error(f"⚠️ **{predicted_label}**")
                    else:  # Benign
                        st.success(f"✅ **{predicted_label}**")

                    # Additional info
                    st.info(
                        f"**Benign Probability:** {probabilities[0]:.2%}\n\n"
                        f"**Cancer Probability:** {probabilities[1]:.2%}"
                    )

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.error(
                        "Please make sure the model file exists and is properly trained.")

    # Add Grad-CAM visualization section
    if uploaded_file is not None and st.button("🔥 Generate Grad-CAM Visualization", type="secondary"):
        with st.spinner("Generating Grad-CAM visualization..."):
            try:

                # Make prediction
                predicted_class, confidence, probabilities = model_predict(
                    model, image)
                predicted_label = CLASS_NAMES[predicted_class]
                targets = [ClassifierOutputTarget(1)]

                # Generate Grad-CAM
                gradcam_image, heatmap = visualize_grad_cam(
                    image, grad_cam, targets)

                if gradcam_image is not None:
                    st.subheader("🔥 Grad-CAM Visualization")
                    st.write(
                        f"Highlighting regions important for **{predicted_label}** classification")

                    # Display original, heatmap, and overlay in three columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Original Image**")
                        st.image(image, use_container_width=True)
                    with col2:
                        st.write("**Heatmap**")
                        # Display heatmap using matplotlib colormap
                        # Remove alpha channel
                        heatmap_colored = cm.jet(heatmap)[:, :, :3]
                        st.image(heatmap_colored, use_container_width=True)
                    with col3:
                        st.write("**Grad-CAM Overlay**")
                        st.image(gradcam_image, use_container_width=True)

                    # Add explanation
                    st.info(
                        "🔍 **Grad-CAM Explanation:**\n\n"
                        "• **Red/Warm colors**: Regions that strongly support the predicted class\n"
                        "• **Blue/Cool colors**: Regions that have less influence on the prediction\n"
                        "• **Overlay**: Shows the heatmap overlaid on the original image\n"
                        "• This visualization helps understand what parts of the image the model focuses on"
                    )

            except Exception as e:
                st.error(f"Error generating Grad-CAM: {str(e)}")
                st.error(
                    "Please make sure pytorch-grad-cam is installed: `pip install grad-cam`")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<small>⚠️ This is for research purposes only. Always consult medical professionals for diagnosis.</small>"
    "</div>",
    unsafe_allow_html=True
)
