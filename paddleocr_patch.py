import numpy as np
import os
import sys

# Apply patches to PaddleOCR modules
def patch_paddleocr():
    # Find the db_postprocess.py file in your environment
    site_packages_path = None
    for path in sys.path:
        if 'site-packages' in path:
            site_packages_path = path
            break
    
    if site_packages_path:
        db_postprocess_path = os.path.join(site_packages_path, 'paddleocr', 'ppocr', 'postprocess', 'db_postprocess.py')
        
        if os.path.exists(db_postprocess_path):
            with open(db_postprocess_path, 'r') as file:
                content = file.read()
            
            # Replace deprecated np.int with int
            content = content.replace('np.int', 'int')
            
            with open(db_postprocess_path, 'w') as file:
                file.write(content)
            
            print("Applied patches to PaddleOCR successfully.")
        else:
            print(f"Could not find {db_postprocess_path}")
    else:
        print("Could not find site-packages directory.")

if __name__ == "__main__":
    patch_paddleocr()
