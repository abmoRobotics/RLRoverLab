import os
import sys
import time
import gdown
import zipfile

def main():
    # Get the script directory (repo root)
    script_dir = os.path.dirname(__file__)

    # Define the download URL and zip file path
    url = "https://drive.google.com/file/d/1VXFTD2OgHcsQL_ifO81AzD2HDkA98h93/view?usp=sharing"
    zip_filename = "terrain_assets.zip"
    zip_path = os.path.join(script_dir, zip_filename)

    # Download the zip file
    try:
        gdown.download(url, output=zip_path, quiet=False)
    except TypeError as e:
        # Exception raised: older gdown version needs fuzzy parameter
        # Fallback for older gdown versions that require fuzzy parameter
        print(f"Error: {e}. Retrying with fuzzy=True for other gdown versions...")
        gdown.download(url, output=zip_path, quiet=False, fuzzy=True)

    # Unzip to repo root
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(script_dir)

    # Delete the zip file
    os.remove(zip_path)

if __name__ == "__main__":
    main()