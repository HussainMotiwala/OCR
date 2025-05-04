import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import easyocr
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch  # Import torch to check GPU availability

# Set page configuration
st.set_page_config(
    page_title="Handwritten Text Recognition",
    page_icon="📝",
    layout="wide"
)

# Create a custom temp directory for uploaded files
TEMP_DIR = tempfile.mkdtemp()

def check_gpu():
    """Check if GPU is available and return its name"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        return True, gpu_names
    else:
        return False, []

def initialize_ocr(lang, use_gpu=False):
    """Initialize EasyOCR with the appropriate language"""
    lang_map = {
        'English': ['en'],
        'Hindi': ['hi'],
        'English & Hindi': ['en', 'hi']  # EasyOCR can handle multiple languages directly
    }
    languages = lang_map.get(lang, ['en'])
    
    # Initialize EasyOCR reader with selected languages
    reader = easyocr.Reader(
        languages,
        gpu=use_gpu
    )
    return reader

def process_image(image_path, lang='English', use_gpu=False):
    """Process the image with EasyOCR"""
    reader = initialize_ocr(lang, use_gpu)
    
    # Read image
    image = cv2.imread(image_path)
    
    # Preprocess the image for better OCR results
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Run OCR on processed image - EasyOCR takes the image directly
    # EasyOCR can use the original image or preprocessed image
    # Let's try the preprocessed image for better results
    result = reader.readtext(thresh)
    
    # Format results - EasyOCR returns [bbox, text, confidence]
    text_results = []
    for (bbox, text, confidence) in result:
        text_results.append({
            'text': text,
            'confidence': float(confidence),
            'position': bbox  # EasyOCR returns [top-left, top-right, bottom-right, bottom-left]
        })
    
    return image, text_results

def draw_bounding_boxes(image, results):
    """Draw bounding boxes on the image"""
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_rgb)
    
    # Draw each bounding box
    for item in results:
        position = np.array(item['position'])
        
        # Create a polygon patch
        polygon = patches.Polygon(
            position, 
            closed=True,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        ax.add_patch(polygon)
        
        # Add text - EasyOCR's bounding boxes are in a different format than PaddleOCR
        # We'll calculate the top-center position differently
        cx = (position[0][0] + position[1][0]) / 2  # Average of top-left and top-right x
        cy = min(position[0][1], position[1][1]) - 10  # Y position of the top edge minus some padding
        
        ax.text(
            cx, cy, 
            f"{item['text']} ({item['confidence']:.2f})",
            color='blue',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    ax.set_axis_off()
    return fig

def main():
    st.title("Handwritten Text Recognition with EasyOCR")
    
    # Sidebar for language selection
    st.sidebar.header("Settings")
    language = st.sidebar.selectbox(
        "Select OCR Language",
        ["English", "Hindi", "English & Hindi"]
    )
    
    # Check GPU availability
    gpu_available, gpu_names = check_gpu()
    
    # GPU option in sidebar with GPU info
    if gpu_available:
        st.sidebar.success(f"✅ GPU Available: {', '.join(gpu_names)}")
        use_gpu = st.sidebar.checkbox("Use GPU for faster processing", value=True)
        if use_gpu:
            st.sidebar.info(f"Currently using: {gpu_names[0]}")
    else:
        st.sidebar.warning("❌ No GPU detected - falling back to CPU")
        use_gpu = False
    
    # File uploader for multiple images
    uploaded_files = st.file_uploader(
        "Upload handwritten document images (max 10 files)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Check if number of files exceeds limit
        if len(uploaded_files) > 10:
            st.error("You can only upload up to 10 files at once.")
            return
        
        # Process each uploaded file
        processed_images = []
        
        with st.spinner(f"Processing images using {'GPU' if use_gpu else 'CPU'}... This may take a moment as EasyOCR loads its models."):
            for uploaded_file in uploaded_files:
                # Save uploaded file to temp directory
                file_path = os.path.join(TEMP_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the image
                image, results = process_image(file_path, language, use_gpu)
                
                processed_images.append({
                    'name': uploaded_file.name,
                    'image': image,
                    'results': results
                })
        
        # Create tabs for each image
        if processed_images:
            tabs = st.tabs([img['name'] for img in processed_images])
            
            for i, tab in enumerate(tabs):
                with tab:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Image with Bounding Boxes")
                        fig = draw_bounding_boxes(processed_images[i]['image'], processed_images[i]['results'])
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Extracted Text")
                        if processed_images[i]['results']:
                            # Create a dataframe for displaying results
                            df = pd.DataFrame([
                                {
                                    'Text': item['text'],
                                    'Confidence': f"{item['confidence']:.2f}"
                                }
                                for item in processed_images[i]['results']
                            ])
                            st.dataframe(df, use_container_width=True)
                            
                            # Add a download button for the text
                            csv = df.to_csv(index=False)
                            filename = processed_images[i]['name'].split('.')[0] + '_text.csv'
                            st.download_button(
                                label="Download Text as CSV",
                                data=csv,
                                file_name=filename,
                                mime="text/csv"
                            )
                        else:
                            st.info("No text detected in this image.")

if __name__ == "__main__":
    main()