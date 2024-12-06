import zipfile
import os

def extract_zip(zip_path, extract_to):
    # Ensure the destination folder exists
    os.makedirs(extract_to, exist_ok=True)
    
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all the contents to the specified folder
        zip_ref.extractall(extract_to)
    print(f"Extracted '{zip_path}' to '{extract_to}'")
