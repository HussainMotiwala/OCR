import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set page configuration
st.set_page_config(
    page_title="Handwritten Text Recognition",
    page_icon="📝",
    layout="wide"
)

# Create a custom temp directory for uploaded files
TEMP_DIR = tempfile.mkdtemp()

def initialize_ocr(lang):
    """Initialize PaddleOCR with the appropriate language"""
    lang_map = {
        'English': 'en',
        'Hindi': 'devanagari',
        'English & Hindi': 'ch'  # Use Chinese model for multilingual
    }
    paddle_lang = lang_map.get(lang, 'en')
    
    ocr = PaddleOCR(
        use_angle_cls=True, 
        lang=paddle_lang,
        show_log=False  # Disable log output in streamlit
    )
    return ocr

def process_image(image_path, lang='English'):
    """Process the image with PaddleOCR"""
    ocr = initialize_ocr(lang)
    
    # Read image
    image = cv2.imread(image_path)
    
    # Preprocess the image for better OCR results
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Run OCR on processed image
    result = ocr.ocr(thresh, cls=True)
    
    # Format results
    text_results = []
    if result:
        for line in result:
            for word_info in line:
                coordinates, (text, confidence) = word_info
                text_results.append({
                    'text': text,
                    'confidence': float(confidence),
                    'position': coordinates
                })
    
    return image, thresh, text_results

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
        
        # Add text
        cx = position[:, 0].mean()
        cy = position[:, 1].min() - 10
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
    st.title("Handwritten Text Recognition with PaddleOCR")
    
    # Sidebar for language selection
    st.sidebar.header("Settings")
    language = st.sidebar.selectbox(
        "Select OCR Language",
        ["English", "Hindi", "English & Hindi"]
    )
    
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
        
        with st.spinner("Processing images..."):
            for uploaded_file in uploaded_files:
                # Save uploaded file to temp directory
                file_path = os.path.join(TEMP_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the image
                original_image, processed_image, results = process_image(file_path, language)
                
                processed_images.append({
                    'name': uploaded_file.name,
                    'original_image': original_image,
                    'processed_image': processed_image,
                    'results': results
                })
        
        # Create tabs for each image
        if processed_images:
            tabs = st.tabs([img['name'] for img in processed_images])
            
            for i, tab in enumerate(tabs):
                with tab:
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Create three tabs for each image
                        image_tabs = st.tabs(["Original", "Processed", "Detected with Bounding Boxes"])
                        
                        # Tab 1: Original Image
                        with image_tabs[0]:
                            st.image(
                                cv2.cvtColor(processed_images[i]['original_image'], cv2.COLOR_BGR2RGB),
                                caption="Original Image",
                                use_container_width=True
                            )
                        
                        # Tab 2: Processed Image
                        with image_tabs[1]:
                            # Convert to RGB for display if it's a grayscale image
                            if len(processed_images[i]['processed_image'].shape) == 2:
                                display_image = cv2.cvtColor(processed_images[i]['processed_image'], cv2.COLOR_GRAY2RGB)
                            else:
                                display_image = cv2.cvtColor(processed_images[i]['processed_image'], cv2.COLOR_BGR2RGB)
                                
                            st.image(
                                display_image,
                                caption="Processed Image (After Thresholding)",
                                use_container_width=True
                            )
                        
                        # Tab 3: Detected with Bounding Boxes
                        with image_tabs[2]:
                            fig = draw_bounding_boxes(processed_images[i]['original_image'], processed_images[i]['results'])
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