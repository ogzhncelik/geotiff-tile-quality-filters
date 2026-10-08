"""
This Python script runs the 256x256 `.tif` image files in a given folder through
a quality control filter, identifies faulty or corrupted ones, and
moves them to another folder ("to_delete").

🎯 Purpose:
- Automatically detect quality issues in the image dataset
- Remove faulty, incomplete or corrupted images before model training

📥 Input:
- `source_dir`: All `.tif` files in the main folder (including subfolders)
- The size and content structure of each `.tif` file are checked

📤 Output:
- The detected corrupted or unsuitable `.tif` files are moved to another location inside
  `trash_dir`, preserving the same folder structure (the `to_delete` folder)

🧪 Applied Quality Checks (in order):
1. **NaN check**: If any pixel contains NaN, the file is considered corrupted.
2. **Nodata check**: If there is even a single pixel whose value == 0 at the same position in all
   3 RGB bands, the file contains nodata and is rejected.
3. **Image size check**: If it is not 256x256, it is considered faulty.
4. **Flat image check**: If the standard deviation of the whole image is < 1, it is treated as a
   completely single-colored (empty/corrupted) tile.
5. **Error cases**: If the file cannot be opened or another error occurs, it is also moved to the
   to_delete folder.

🔧 Technical Features:
- A progress bar is shown with `tqdm`
- Raster reading and analysis are done with `rasterio` and `numpy`
- Ctrl+C stop support with `signal` → the log is written in every case

"""


import os
import glob
import shutil
import signal
import datetime
import rasterio
import numpy as np
from tqdm import tqdm


# Settings
source_dir = r"path\to\input"
trash_dir = os.path.join(r"path\to\output",
                         os.path.basename(source_dir.rstrip("\\/")))
os.makedirs(trash_dir, exist_ok=True)

tif_files = glob.glob(os.path.join(source_dir, "**", "*.tif"), recursive=True)

log_file = os.path.join(os.path.dirname(trash_dir), "filter_log.txt")

expected_width = 256
expected_height = 256


def move_with_structure(file_path):
    relative_path = os.path.relpath(file_path, source_dir)
    destination_path = os.path.join(trash_dir, relative_path)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(file_path, destination_path)


def process_single_tif(tif_path):
    reason = ""
    detail = ""
    should_move = False

    try:
        with rasterio.open(tif_path) as src:
            data = src.read()  # shape: (bands, height, width)
            width, height = src.width, src.height

            # 1. NaN check
            if np.any(np.isnan(data)):
                should_move = True
                reason = "nan_pixel"
                nan_count = int(np.sum(np.isnan(data)))
                detail = f"nan_pixel_count={nan_count}"

            # 2. Nodata check — pixels that are 0 at the same position in all 3 bands
            if not should_move:
                nodata_mask = np.all(data == 0, axis=0)  # shape: (height, width)
                if np.any(nodata_mask):
                    should_move = True
                    reason = "nodata"
                    nodata_count = int(np.sum(nodata_mask))
                    nodata_ratio = nodata_count / (height * width)
                    detail = f"nodata_pixels={nodata_count},nodata_ratio={nodata_ratio:.4f}"

            # 3. Size check
            if not should_move and (width != expected_width or height != expected_height):
                should_move = True
                reason = "size_error"
                detail = f"{width}x{height}"

            # 4. Flat image check (completely single-colored / empty tile)
            if not should_move and np.std(data) < 1:
                should_move = True
                reason = "flat_image"
                detail = f"std={np.std(data):.4f},min={int(np.min(data))},max={int(np.max(data))}"

            # 5. Dark image check (nearly black tile)
            if not should_move and np.max(data) < 30:
                should_move = True
                reason = "dark_image"
                detail = f"max={int(np.max(data))},std={np.std(data):.4f}"

    except Exception as e:
        should_move = True
        reason = "exception"
        detail = str(e)

    if should_move:
        move_with_structure(tif_path)
        return (1, tif_path, reason, detail)

    return (0, tif_path, "clean", "")


if __name__ == '__main__':
    results = []
    stop_flag = False

    def handle_signal(sig, frame):
        global stop_flag
        stop_flag = True
        print("\n⚠️ Stop signal received, writing the log...")

    signal.signal(signal.SIGINT, handle_signal)

    for tif_path in tqdm(tif_files, total=len(tif_files)):
        if stop_flag:
            break
        r = process_single_tif(tif_path)
        results.append(r)

    # Write the log (delete it if it exists, then recreate it)
    if os.path.exists(log_file):
        os.remove(log_file)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Run time: {datetime.datetime.now()}\n")
        f.write(f"Processed: {len(results)}/{len(tif_files)}\n")
        f.write(f"\n📌 Column Descriptions:\n")
        f.write(f"   nan_pixel_count : Number of NaN pixels in the image\n")
        f.write(f"   nodata_pixels   : Number of pixels that are 0 at the same position in all 3 RGB bands\n")
        f.write(f"   nodata_ratio    : Ratio of the nodata pixel count to the total pixel count\n")
        f.write(f"   std             : Standard deviation of all pixel values in the image\n")
        f.write(f"{'=' * 60}\n")
        for r in results:
            durum = "REJECTED" if r[0] == 1 else "CLEAN"
            f.write(f"[{durum}] {os.path.basename(r[1])} | {r[2]} | {r[3]}\n")

    moved_count = sum(r[0] for r in results)
    print(f"📄 Log file: {log_file} ({len(results)}/{len(tif_files)} files processed)")

    if moved_count == 0:
        print("⚠️ No data to delete was found.")
    else:
        print(f"✅ A total of {moved_count} files were moved to the to_delete folder: {trash_dir}")