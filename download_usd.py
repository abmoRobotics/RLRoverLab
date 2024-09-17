import os
import time

import gdown


def download_google_drive_folder(url, target_path):
    os.makedirs(absolute_path, exist_ok=True)
    os.chdir(absolute_path)
    gdown.download_folder(url, quiet=False, use_cookies=False)


downloads = [
    ("https://drive.google.com/drive/folders/1AILZGie2vZBGXWITrE5IPTy9UoWsTZ6a?usp=sharing",
     "rover_envs/assets/terrains/debug"),
    ("https://drive.google.com/drive/folders/1r75L4dbtflN301RXLy8J_CHHBNqutPgE?usp=sharing",
     "rover_envs/assets/terrains/mars/terrain1"),
    ("https://drive.google.com/drive/folders/1DFLI9TooqGj-PBeFKvDGSP1b7B1zzPD8?usp=sharing",
     "rover_envs/assets/robots/aau_rover_simple"),
]
script_dir = os.path.dirname(__file__)
for url, path in downloads:
    absolute_path = os.path.join(script_dir, path)
    download_google_drive_folder(url, absolute_path)

    # Time to sleep to avoid getting blocked by Google Drive
    time.sleep(10)
