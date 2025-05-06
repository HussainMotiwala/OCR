import streamlit as st
import os
import cv2
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
import pandas as pd
from paddleocr import PaddleOCR
import tempfile
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdf2image  # Added for PDF conversion

# Set page configuration
st.set_page_config(
    page_title="Text Recognition",
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

def convert_pdf_to_images(pdf_path):
    """Convert PDF file to a list of images"""
    try:
        # Convert PDF to list of PIL Images with error handling
        images = pdf2image.convert_from_path(pdf_path)
        
        # Return empty list if no images were converted
        if not images:
            st.error(f"Could not convert PDF to images: {pdf_path}")
            return []
        
        # Convert PIL Images to OpenCV format
        opencv_images = []
        for i, img in enumerate(images):
            # Convert PIL Image to numpy array
            np_image = np.array(img)
            # Convert RGB to BGR (OpenCV format)
            opencv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            opencv_images.append(opencv_image)
        
        return opencv_images
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        # Return empty list so the application doesn't crash
        return []

def process_image(image, lang='English'):
    """Process the image with PaddleOCR"""
    ocr = initialize_ocr(lang)
    
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
    if result and isinstance(result, list) and len(result) > 0:
        for line in result:
            if line:  # Check if line is not None and not empty
                for word_info in line:
                    if word_info and len(word_info) == 2:  # Ensure word_info has expected structure
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

def process_file(file_path, language):
    """Process a file based on its type (image or PDF)"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        # Convert PDF to images and process each page
        images = convert_pdf_to_images(file_path)
        
        # If no images were converted, return empty result
        if not images:
            return []
        
        results = []
        for i, image in enumerate(images):
            try:
                original, processed, ocr_results = process_image(image, language)
                results.append({
                    'name': f"Page {i+1}",
                    'original_image': original,
                    'processed_image': processed,
                    'results': ocr_results
                })
            except Exception as e:
                st.error(f"Error processing page {i+1}: {str(e)}")
                # Continue with next page instead of crashing
                continue
                
        return results
    else:
        # Process as a single image
        try:
            # Read image with error handling
            img = cv2.imread(file_path)
            if img is None:
                st.error(f"Could not read image file: {file_path}")
                return []
                
            original, processed, ocr_results = process_image(img, language)
            return [{
                'name': os.path.basename(file_path),
                'original_image': original,
                'processed_image': processed,
                'results': ocr_results
            }]
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return []

def main():
    st.title("Text Recognition with PaddleOCR")
    
    # Sidebar for language selection
    st.sidebar.header("Settings")
    language = st.sidebar.selectbox(
        "Select OCR Language",
        ["English", "Hindi", "English & Hindi"]
    )
    
    # File uploader for multiple files including PDFs
    uploaded_files = st.file_uploader(
        "Upload document images or PDFs (max 10 files)",
        type=["jpg", "jpeg", "png", "pdf"],  # Added PDF as supported type
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Check if number of files exceeds limit
        if len(uploaded_files) > 10:
            st.error("You can only upload up to 10 files at once.")
            return
        
        # Process each uploaded file
        all_processed_items = []
        
        with st.spinner("Processing files..."):
            # Keep track of processed filenames to avoid duplicates
            processed_filenames = set()
            
            for uploaded_file in uploaded_files:
                try:
                    # Check if this filename has already been processed
                    if uploaded_file.name in processed_filenames:
                        # Create a unique filename by adding a timestamp
                        base_name, extension = os.path.splitext(uploaded_file.name)
                        unique_filename = f"{base_name}_{int(time.time())}_{len(processed_filenames)}{extension}"
                        st.info(f"Duplicate file detected. Processing {uploaded_file.name} as {unique_filename}")
                        file_path = os.path.join(TEMP_DIR, unique_filename)
                    else:
                        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
                        processed_filenames.add(uploaded_file.name)
                    
                    # Save uploaded file to temp directory
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the file (image or PDF)
                    processed_items = process_file(file_path, language)
                    if processed_items:  # Only extend if we got valid results
                        all_processed_items.extend(processed_items)
                    else:
                        st.warning(f"No results were generated for {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")        
        # Create tabs for each processed item (image or PDF page)
        if all_processed_items:
            tabs = st.tabs([item['name'] for item in all_processed_items])
            
            for i, tab in enumerate(tabs):
                with tab:
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Create three tabs for each image
                        image_tabs = st.tabs(["Original", "Processed", "Detected with Bounding Boxes"])
                        
                        # Tab 1: Original Image
                        with image_tabs[0]:
                            st.image(
                                cv2.cvtColor(all_processed_items[i]['original_image'], cv2.COLOR_BGR2RGB),
                                caption="Original Image",
                                use_container_width=True
                            )
                        
                        # Tab 2: Processed Image
                        with image_tabs[1]:
                            # Convert to RGB for display if it's a grayscale image
                            if len(all_processed_items[i]['processed_image'].shape) == 2:
                                display_image = cv2.cvtColor(all_processed_items[i]['processed_image'], cv2.COLOR_GRAY2RGB)
                            else:
                                display_image = cv2.cvtColor(all_processed_items[i]['processed_image'], cv2.COLOR_BGR2RGB)
                                
                            st.image(
                                display_image,
                                caption="Processed Image (After Thresholding)",
                                use_container_width=True
                            )
                        
                        # Tab 3: Detected with Bounding Boxes
                        with image_tabs[2]:
                            fig = draw_bounding_boxes(all_processed_items[i]['original_image'], all_processed_items[i]['results'])
                            st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Extracted Text")
                        if all_processed_items[i]['results']:
                            # Create a dataframe for displaying results
                            df = pd.DataFrame([
                                {
                                    'Text': item['text'],
                                    'Confidence': f"{item['confidence']:.2f}"
                                }
                                for item in all_processed_items[i]['results']
                            ])
                            st.dataframe(df, use_container_width=True)
                            
                            # Add a download button for the text with a unique key
                            csv = df.to_csv(index=False)
                            filename = all_processed_items[i]['name'].split('.')[0] + '_text.csv'
                            st.download_button(
                                label="Download Text as CSV",
                                data=csv,
                                file_name=filename,
                                mime="text/csv",
                                key=f"download_{i}"  # Adding a unique key for each download button
                            )
                        else:
                            st.info("No text detected in this image.")

    # Add instructions in the sidebar
    with st.sidebar.expander("Instructions", expanded=False):
        st.markdown("""
        ### How to use:
        1. Select the language for OCR recognition
        2. Upload images or PDF files containing text
        3. View the extracted text and download as CSV
        
        ### Supported file types:
        - Images: JPG, JPEG, PNG
        - Documents: PDF
        
        ### Notes:
        - For PDFs, each page is processed separately
        - Maximum 10 files can be uploaded at once
        """)

if __name__ == "__main__":
    main()
