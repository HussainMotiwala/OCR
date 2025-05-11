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

def initialize_ocr(lang, model_type="Default"):
    """Initialize PaddleOCR with the appropriate language and model"""
    lang_map = {
        'English': 'en',
        'Hindi': 'devanagari',
        'English & Hindi': 'ch'  # Use Chinese model for multilingual
    }
    paddle_lang = lang_map.get(lang, 'en')
    
    # Model configuration based on selection
    if model_type == "PP-OCRv3":
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=paddle_lang,
            show_log=False,
            use_pd_optimize=True,
            rec_algorithm="SVTR_LCNet"
        )
    else:
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

def preprocess_none(image):
    """No preprocessing, return original image"""
    return image

def preprocess_basic(image):
    """Basic preprocessing: grayscale and thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2  # Changed from THRESH_BINARY_INV to THRESH_BINARY
    )
    return thresh

def preprocess_enhanced(image):
    """Enhanced preprocessing for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh

def preprocess_advanced(image):
    """Advanced preprocessing for text"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Apply bilateral filter to preserve edges while removing noise
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to close small gaps
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def preprocess_super_resolution(image):
    """Super resolution processing for low-quality images"""
    # Resize image to 2x using INTER_CUBIC
    h, w = image.shape[:2]
    enhanced = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    
    # Now apply standard preprocessing to the enlarged image
    processed = preprocess_enhanced(enhanced)
    
    return processed

def deskew_image(image):
    """Deskew the image to straighten text lines"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply threshold to get black text on white background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find all non-zero points
    coords = np.column_stack(np.where(thresh > 0))
    
    # Find the minimum area rectangle that contains all white pixels
    rect = cv2.minAreaRect(coords)
    angle = rect[2]
    
    # Adjust angle for proper rotation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate the image to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), 
                             flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def apply_selected_preprocessing(image, method):
    """Apply the selected preprocessing method to the image"""
    if method == "None":
        return preprocess_none(image)
    elif method == "Basic":
        return preprocess_basic(image)
    elif method == "Enhanced":
        return preprocess_enhanced(image)
    elif method == "Advanced":
        return preprocess_advanced(image)
    elif method == "Super Resolution":
        return preprocess_super_resolution(image)
    elif method == "Deskewed":
        return deskew_image(image)
    else:
        return image  # Default fallback

def process_image(image, lang='English', preprocessing_method="Basic", model_type="Default"):
    """Process the image with PaddleOCR"""
    ocr = initialize_ocr(lang, model_type)
    
    # Apply selected preprocessing
    processed_image = apply_selected_preprocessing(image, preprocessing_method)
    
    # Run OCR on processed image
    result = ocr.ocr(processed_image, cls=True)
    
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
    
    return image, processed_image, text_results

def post_process_text(text_results, language):
    """Post-process OCR results based on language"""
    processed_results = []
    
    for item in text_results:
        text = item['text']
        
        # Basic text cleaning
        text = text.strip()
        
        # Language-specific corrections
        if language == 'English':
            # Remove non-alphanumeric characters except spaces and punctuation
            text = re.sub(r'[^\w\s\.\,\?\!\-\']', '', text)
            
            # Correct common OCR errors
            text = text.replace('0', 'O').replace('1', 'I') if len(text) == 1 else text
            
        elif language == 'Hindi':
            # Hindi-specific corrections here
            pass
        
        # Update the result
        item['text'] = text
        processed_results.append(item)
        
    return processed_results

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

def group_text_into_sentences(text_results):
    """Group detected text into sentences based on position"""
    if not text_results:
        return []
        
    # Sort results by y-coordinate (top to bottom)
    sorted_by_y = sorted(text_results, key=lambda x: np.mean([p[1] for p in x['position']]))
    
    # Group by approximate y position (same line)
    lines = []
    current_line = [sorted_by_y[0]]
    current_y = np.mean([p[1] for p in sorted_by_y[0]['position']])
    
    for item in sorted_by_y[1:]:
        y = np.mean([p[1] for p in item['position']])
        # Threshold for same line - may need adjustment
        if abs(y - current_y) < 20:  
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
            current_y = y
    
    if current_line:
        lines.append(current_line)
    
    # Sort each line by x-coordinate (left to right)
    for i, line in enumerate(lines):
        lines[i] = sorted(line, key=lambda x: np.mean([p[0] for p in x['position']]))
    
    # Join text in each line
    sentences = []
    for line in lines:
        sentence = ' '.join([item['text'] for item in line])
        combined_confidence = sum([item['confidence'] for item in line]) / len(line)
        sentences.append({
            'text': sentence,
            'confidence': combined_confidence
        })
    
    return sentences

def process_file(file_path, language, preprocessing_methods, model_type):
    """Process a file with multiple preprocessing methods"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        # Convert PDF to images and process each page
        images = convert_pdf_to_images(file_path)
        
        # If no images were converted, return empty result
        if not images:
            return []
        
        results = []
        for i, image in enumerate(images):
            page_results = []
            
            for method in preprocessing_methods:
                try:
                    original, processed, ocr_results = process_image(
                        image, language, method, model_type
                    )
                    
                    # Post-process the text
                    processed_results = post_process_text(ocr_results, language)
                    
                    # Group text into sentences
                    sentences = group_text_into_sentences(processed_results)
                    
                    page_results.append({
                        'name': f"Page {i+1} - {method}",
                        'original_image': original,
                        'processed_image': processed,
                        'results': processed_results,
                        'sentences': sentences,
                        'method': method
                    })
                except Exception as e:
                    st.error(f"Error processing page {i+1} with {method}: {str(e)}")
                    # Continue with next method instead of crashing
                    continue
            
            if page_results:
                results.extend(page_results)
                
        return results
    else:
        # Process as a single image
        try:
            # Read image with error handling
            img = cv2.imread(file_path)
            if img is None:
                st.error(f"Could not read image file: {file_path}")
                return []
            
            image_results = []
            
            for method in preprocessing_methods:
                try:
                    original, processed, ocr_results = process_image(
                        img, language, method, model_type
                    )
                    
                    # Post-process the text
                    processed_results = post_process_text(ocr_results, language)
                    
                    # Group text into sentences
                    sentences = group_text_into_sentences(processed_results)
                    
                    image_results.append({
                        'name': f"{os.path.basename(file_path)} - {method}",
                        'original_image': original,
                        'processed_image': processed,
                        'results': processed_results,
                        'sentences': sentences,
                        'method': method
                    })
                except Exception as e:
                    st.error(f"Error processing {file_path} with {method}: {str(e)}")
                    # Continue with next method instead of crashing
                    continue
            
            return image_results
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return []

def main():
    st.title("Advanced Text Recognition with PaddleOCR")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # Language selection
    language = st.sidebar.selectbox(
        "Select OCR Language",
        ["English", "Hindi", "English & Hindi"]
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Default", "PP-OCRv3"]
    )
    
    # Preprocessing method selection
    preprocessing_methods = st.sidebar.multiselect(
        "Select Preprocessing Methods (For Comparison)",
        ["None", "Basic", "Enhanced", "Advanced", "Super Resolution", "Deskewed"],
        default=["None", "Basic", "Advanced"]
    )
    
    # Confidence threshold
    min_confidence = st.sidebar.slider(
        "Minimum Confidence Threshold", 
        0.0, 1.0, 0.3, 0.05
    )
    
    # File uploader for multiple files including PDFs
    uploaded_files = st.file_uploader(
        "Upload document images or PDFs (max 10 files)",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Check if number of files exceeds limit
        if len(uploaded_files) > 10:
            st.error("You can only upload up to 10 files at once.")
            return
        
        # Ensure at least one preprocessing method is selected
        if not preprocessing_methods:
            st.warning("Please select at least one preprocessing method.")
            return
        
        # Process each uploaded file
        all_processed_items = []
        
        with st.spinner("Processing files..."):
            # Progress bar
            progress_bar = st.progress(0)
            
            # Keep track of processed filenames to avoid duplicates
            processed_filenames = set()
            
            for file_idx, uploaded_file in enumerate(uploaded_files):
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
                    
                    # Process the file with multiple preprocessing methods
                    processed_items = process_file(file_path, language, preprocessing_methods, model_type)
                    
                    if processed_items:  # Only extend if we got valid results
                        all_processed_items.extend(processed_items)
                    else:
                        st.warning(f"No results were generated for {uploaded_file.name}")
                        
                    # Update progress
                    progress = (file_idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Clear progress when done
            progress_bar.empty()
        
        # Display results if any
        if all_processed_items:
            # Group items by original file/page
            grouped_items = {}
            for item in all_processed_items:
                # Extract the base name without the method
                base_name = item['name'].split(' - ')[0]
                if base_name not in grouped_items:
                    grouped_items[base_name] = []
                grouped_items[base_name].append(item)
            
            # Create tabs for each original file/page
            file_tabs = st.tabs(list(grouped_items.keys()))
            
            for file_idx, (file_name, items) in enumerate(grouped_items.items()):
                with file_tabs[file_idx]:
                    # Display the original image first
                    st.subheader("Original Image")
                    st.image(
                        cv2.cvtColor(items[0]['original_image'], cv2.COLOR_BGR2RGB),
                        use_container_width=True
                    )
                    
                    # Create tabs for each preprocessing method
                    method_tabs = st.tabs([item['method'] for item in items])
                    
                    for method_idx, item in enumerate(items):
                        with method_tabs[method_idx]:
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # Display processed image
                                if len(item['processed_image'].shape) == 2:
                                    display_image = cv2.cvtColor(item['processed_image'], cv2.COLOR_GRAY2RGB)
                                else:
                                    display_image = cv2.cvtColor(item['processed_image'], cv2.COLOR_BGR2RGB)
                                
                                st.image(
                                    display_image,
                                    caption=f"Processed with {item['method']}",
                                    use_container_width=True
                                )
                                
                                # Display image with bounding boxes
                                st.subheader("Detected Text")
                                fig = draw_bounding_boxes(item['original_image'], item['results'])
                                st.pyplot(fig)
                            
                            with col2:
                                # Display raw OCR results
                                st.subheader("Raw Text Results")
                                if item['results']:
                                    # Filter by confidence threshold
                                    filtered_results = [
                                        result for result in item['results'] 
                                        if result['confidence'] >= min_confidence
                                    ]
                                    
                                    if filtered_results:
                                        df = pd.DataFrame([
                                            {
                                                'Text': result['text'],
                                                'Confidence': f"{result['confidence']:.2f}"
                                            }
                                            for result in filtered_results
                                        ])
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Download button for raw text
                                        csv = df.to_csv(index=False)
                                        filename = f"{file_name}_{item['method']}_raw.csv"
                                        st.download_button(
                                            label="Download Raw Text as CSV",
                                            data=csv,
                                            file_name=filename,
                                            mime="text/csv",
                                            key=f"download_raw_{file_idx}_{method_idx}"
                                        )
                                    else:
                                        st.info("No text detected above confidence threshold.")
                                else:
                                    st.info("No text detected in this image.")
                                
                                # Display grouped sentences
                                st.subheader("Structured Text")
                                if item['sentences']:
                                    # Filter by confidence threshold
                                    filtered_sentences = [
                                        sentence for sentence in item['sentences'] 
                                        if sentence['confidence'] >= min_confidence
                                    ]
                                    
                                    if filtered_sentences:
                                        sentences_df = pd.DataFrame([
                                            {
                                                'Sentence': sentence['text'],
                                                'Confidence': f"{sentence['confidence']:.2f}"
                                            }
                                            for sentence in filtered_sentences
                                        ])
                                        st.dataframe(sentences_df, use_container_width=True)
                                        
                                        # Display full text
                                        st.subheader("Full Text")
                                        full_text = "\n".join([s['text'] for s in filtered_sentences])
                                        st.text_area(
                                            "Extracted Text", 
                                            full_text, 
                                            height=200,
                                            key=f"text_{file_idx}_{method_idx}"
                                        )
                                        
                                        # Download button for sentences
                                        sentences_csv = sentences_df.to_csv(index=False)
                                        sentences_filename = f"{file_name}_{item['method']}_sentences.csv"
                                        st.download_button(
                                            label="Download Structured Text as CSV",
                                            data=sentences_csv,
                                            file_name=sentences_filename,
                                            mime="text/csv",
                                            key=f"download_sent_{file_idx}_{method_idx}"
                                        )
                                        
                                        # Download button for full text
                                        st.download_button(
                                            label="Download Full Text",
                                            data=full_text,
                                            file_name=f"{file_name}_{item['method']}_fulltext.txt",
                                            mime="text/plain",
                                            key=f"download_full_{file_idx}_{method_idx}"
                                        )
                                    else:
                                        st.info("No sentences detected above confidence threshold.")
                                else:
                                    st.info("No structured text detected.")
                            
                            # Add comparison section for method performance
                            st.subheader("Method Performance")
                            if item['results']:
                                avg_confidence = sum(r['confidence'] for r in item['results']) / len(item['results']) if item['results'] else 0
                                text_count = len(item['results'])
                                char_count = sum(len(r['text']) for r in item['results'])
                                
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                with metrics_col1:
                                    st.metric("Text Blocks", text_count)
                                with metrics_col2:
                                    st.metric("Characters", char_count)
                                with metrics_col3:
                                    st.metric("Avg. Confidence", f"{avg_confidence:.2f}")
                            
    # Add instructions in the sidebar
    with st.sidebar.expander("Instructions", expanded=False):
        st.markdown("""
        ### How to use:
        1. Select the language for OCR recognition
        2. Choose the model type (Default or PP-OCRv3)
        3. Select multiple preprocessing methods to compare
        4. Set the minimum confidence threshold
        5. Upload images or PDF files containing text
        6. Compare results across different preprocessing methods
        
        ### Preprocessing Methods:
        - **None**: Original image without preprocessing
        - **Basic**: Simple grayscale and thresholding
        - **Enhanced**: Noise reduction + contrast enhancement + thresholding
        - **Advanced**: Complex pipeline with bilateral filtering and morphology
        - **Super Resolution**: Upscaling for low-quality images
        - **Deskewed**: Rotation correction for tilted text
        
        ### Supported file types:
        - Images: JPG, JPEG, PNG
        - Documents: PDF
        
        ### Notes:
        - For PDFs, each page is processed separately
        - Maximum 10 files can be uploaded at once
        - Different preprocessing methods work better for different document types
        """)

if __name__ == "__main__":
    main()
