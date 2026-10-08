"""
This Python script runs the 256x256 `.tif` image files in a given folder (e.g. "terrestrial") through a quality control filter, identifies faulty or corrupted ones, and moves them to another folder ("to_delete").

🎯 Purpose:
- Automatically detect quality issues in the image dataset
- Remove faulty, incomplete or corrupted images before model training

📥 Input:
- `source_dir`: All `.tif` files in the main folder (e.g. terrestrial), including subfolders
- The size and content structure of each `.tif` file are checked

📤 Output:
- The detected corrupted or unsuitable `.tif` files are moved to another location inside `trash_dir`, preserving the same folder structure (the `to_delete` folder)

🧪 Applied Quality Checks:
1. **Black window detection**: 32x32 windows are scanned inside the image. If there are windows where more than 3% of the pixels are very dark (pixel value < 5), the file is considered corrupted.
2. **Image size check**: If it is not 256x256, it is considered faulty.
3. **Corner check**: If the mean value of the pixels at the four corners of the image is very low (e.g. < 15), the file is considered cropped.
4. **Error cases**: If the file cannot be opened or another error occurs, it is also moved to the to_delete folder.

🔧 Technical Features:
- All files are processed in parallel with `multiprocessing.Pool` → runs fast
- A progress bar is shown with `tqdm`
- Raster reading and analysis are done with `rasterio` and `numpy`

"""

import os
import glob
import shutil
import rasterio
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# Settings
source_dir = r"path\to\input"
trash_dir = os.path.join(r"path\to\output",
                         os.path.basename(source_dir.rstrip("\\/")))
os.makedirs(trash_dir, exist_ok=True)

tif_files = glob.glob(os.path.join(source_dir, "**", "*.tif"), recursive=True)

log_file = os.path.join(os.path.dirname(trash_dir), "filter_log.txt")

expected_width = 256
expected_height = 256
corner_threshold = 15  # if below this value, consider it "cropped"
corner_size = 1  # corner window size

def move_with_structure(file_path):
    relative_path = os.path.relpath(file_path, source_dir)
    destination_path = os.path.join(trash_dir, relative_path)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(file_path, destination_path)

def process_single_tif(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            data = src.read()
            width, height = src.width, src.height

            # Black window detection
            window_size = 32
            threshold_ratio = 0.03
            for i in range(0, data.shape[1], window_size):
                for j in range(0, data.shape[2], window_size):
                    window = data[:, i:i + window_size, j:j + window_size]
                    if window.shape[1] != window_size or window.shape[2] != window_size:
                        continue
                    black_ratio = np.mean(window < 5)
                    if black_ratio > threshold_ratio:
                        move_with_structure(tif_path)
                        return (1, tif_path, "black_window", f"i={i},j={j},black_ratio={black_ratio:.4f}")

            # Size check
            if width != expected_width or height != expected_height:
                move_with_structure(tif_path)
                return (1, tif_path, "size_error", f"{width}x{height}")

            # Corner check
            ul = data[:, :corner_size, :corner_size]
            ur = data[:, :corner_size, -corner_size:]
            ll = data[:, -corner_size:, :corner_size]
            lr = data[:, -corner_size:, -corner_size:]
            corners = [ul, ur, ll, lr]
            corner_means = [np.mean(corner) for corner in corners]
            if any(mean < corner_threshold for mean in corner_means):
                move_with_structure(tif_path)
                return (1, tif_path, "corner_error", f"UL={corner_means[0]:.1f},UR={corner_means[1]:.1f},LL={corner_means[2]:.1f},LR={corner_means[3]:.1f}")

        return (0, tif_path, "clean", "")
    except Exception as e:
        move_with_structure(tif_path)
        return (1, tif_path, "exception", str(e))

if __name__ == '__main__':
    # Test: try to open the first file
    test_file = tif_files[0]
    try:
        with open(test_file, "rb") as f:
            print(f"✅ File can be opened: {test_file}")
    except Exception as e:
        print(f"❌ File cannot be opened: {e}")

    try:
        with rasterio.open(test_file) as src:
            print(f"✅ Can be opened with rasterio: {src.width}x{src.height}")
    except Exception as e:
        print(f"❌ Cannot be opened with rasterio: {e}")

    results = []
    try:
        with Pool(cpu_count()) as pool:
            for r in tqdm(pool.imap(process_single_tif, tif_files), total=len(tif_files)):
                results.append(r)
    except KeyboardInterrupt:
        print("\n⚠️ Program stopped by the user.")
    finally:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"Run time: {__import__('datetime').datetime.now()}\n")
                f.write(f"Processed: {len(results)}/{len(tif_files)}\n")
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
        except Exception as e2:
            print(f"❌ Could not write the log: {e2}")

        moved_count = sum(r[0] for r in results)
        print(f"📄 Log file: {log_file} ({len(results)}/{len(tif_files)} files processed)")

        if moved_count == 0:
            print("⚠️ No data to delete was found.")
        else:
            print(f"✅ A total of {moved_count} files were moved to the to_delete folder: {trash_dir}")