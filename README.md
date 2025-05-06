# OCR
OCR using PaddleOCR

Features

Multi-language OCR Support: Process documents in English, Hindi, or both languages simultaneously
Batch Processing: Upload and process up to 10 images at once
Advanced Image Processing: Automatic thresholding to improve handwriting recognition accuracy
Interactive Visualization: View original images, preprocessed versions, and text detection with bounding boxes
Data Export: Download extracted text as CSV files for further analysis

Requirements

Python 3.7+
PaddleOCR
OpenCV
Streamlit
Pandas
Matplotlib
Pillow

Installation

Clone this repository:
bashgit clone https://github.com/HussainMotiwala/OCR.git
cd OCR

Create and activate a virtual environment (recommended):
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required packages:
bashpip install -r requirements.txt

Install PaddleOCR and dependencies:
bashpip install paddlepaddle paddleocr


Usage

Start the Streamlit application:
bashstreamlit run app.py

Open your web browser and navigate to:
http://localhost:8501

Use the application:

Select your preferred OCR language from the sidebar
Upload handwritten document images (JPG, JPEG, or PNG format)
View the extracted text and visualizations in the tabbed interface
Download the results as CSV files if needed



How It Works
The application processes handwritten documents through the following pipeline:

Image Upload: Process multiple handwritten document images through the Streamlit interface
Preprocessing: Convert images to grayscale and apply adaptive thresholding to improve text visibility
OCR Processing: Run PaddleOCR on the preprocessed images with the selected language model
Visualization: Display the original image, processed image, and text detection with bounding boxes
Results Display: Present extracted text with confidence scores in a tabular format
Export: Download the extracted text as CSV files for further analysis

Code Structure

app.py: Main application file containing the Streamlit UI and processing logic

initialize_ocr(): Configures PaddleOCR with the appropriate language model
process_image(): Handles image preprocessing and OCR processing
draw_bounding_boxes(): Creates visualizations with text detection bounding boxes
main(): Orchestrates the Streamlit UI and processing workflow



Customization
Language Support
The application supports English, Hindi, and multilingual (English & Hindi) text recognition. To add support for additional languages:

Update the lang_map dictionary in the initialize_ocr() function:
pythonlang_map = {
    'English': 'en',
    'Hindi': 'devanagari',
    'English & Hindi': 'ch',  # Use Chinese model for multilingual
    'New Language': 'appropriate_paddleocr_code'  # Add your new language
}

Add the new language option to the sidebar selectbox:
pythonlanguage = st.sidebar.selectbox(
    "Select OCR Language",
    ["English", "Hindi", "English & Hindi", "New Language"]
)


Image Preprocessing
To adjust the image preprocessing parameters for better OCR results with specific document types:

Modify the thresholding parameters in the process_image() function:
pythonthresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, block_size, C
)

Increase block_size (default: 11) for documents with larger text
Adjust C (default: 2) to control thresholding sensitivity



Performance Tips

For better OCR accuracy with handwritten text:

Ensure good lighting and contrast in the source images
Use the appropriate language model for your documents
Experiment with different preprocessing parameters


For faster processing:

Resize large images before uploading
Process fewer images at a time for quicker results



License
MIT License
Credits
This application leverages several powerful open-source libraries:

PaddleOCR - Optical Character Recognition toolkit
Streamlit - Web application framework
OpenCV - Computer vision and image processing
Pandas - Data manipulation and analysis
Matplotlib - Data visualization

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

