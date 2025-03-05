import os
import time
import gdown
import zipfile

# Get the script directory (repo root)
script_dir = os.path.dirname(__file__)

# First download entry (only process this one)
url = "https://drive.google.com/file/d/1VXFTD2OgHcsQL_ifO81AzD2HDkA98h93/view?usp=sharing"
zip_filename = "terrain_assets.zip"
zip_path = os.path.join(script_dir, zip_filename)

# Download the zip file
gdown.download(url, output=zip_path, quiet=False, fuzzy=True)

# Unzip to repo root
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(script_dir)

# Delete the zip file
os.remove(zip_path)