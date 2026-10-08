"""
This Python script scans all `.tif` raster images in a given source folder,
detects images that contain any completely black pixel (0 or very close to 0),
and moves them to a specified 'trash' folder.

🎯 Purpose:
- Remove `.tif` files that contain completely black or nearly black pixels, which carry no meaning
  during training in image processing or machine learning models, from the dataset.
- Improve data quality and reduce the risk of faulty segmentation.

📥 Inputs:
- `source_dir`: Source folder to be scanned (`.tif` files including subfolders)
  - Example: `output_tiles/terrestrial`
- RGB or multi-band images in `.tif` format

📤 Outputs:
- `.tif` files that contain near-black pixels are moved to the `trash_dir` folder
  - Example: `output_tiles/to_delete`
- The number of moved files is printed on the command line

🔁 Processing Steps:
1. All `.tif` files under the given `source_dir` directory are listed with `glob.glob()` (including subfolders).
2. Each `.tif` file is read using `rasterio`.
3. With the expression `numpy.any(data < 5)`, if any pixel value is less than 5 (close to black), the file is considered invalid.
4. The invalid file is moved to the `trash_dir` folder with `shutil.move()`.
5. At the end, the number of moved files is reported to the user.

"""


import os
import glob
import shutil
import rasterio
import numpy as np
from tqdm import tqdm

# Folder paths
source_dir = r"path\to\input"
trash_dir = os.path.join(r"path\to\output")
os.makedirs(trash_dir, exist_ok=True)

# Scan the .tif files
tif_files = glob.glob(os.path.join(source_dir, "**", "*.tif"), recursive=True)
moved_count = 0

for tif_path in tqdm(tif_files, desc="Searching for images with black pixels", unit="file"):
    try:
        with rasterio.open(tif_path) as src:
            data = src.read()

        # Move the file if at least one pixel is fully black (0)
        if np.any(data < 5):
            shutil.move(tif_path, os.path.join(trash_dir, os.path.basename(tif_path)))
            moved_count += 1
            #print(f"{os.path.basename(tif_path)} moved: At least one black pixel was found.")

    except Exception as e:
        print(f"⚠️ An error occurred: {tif_path} | {e}")

if moved_count == 0:
    print("⚠️ No images containing black pixels were found.")
else:
    print(f"✅ {moved_count} files moved to the trash folder.")