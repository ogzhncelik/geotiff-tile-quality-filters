# geotiff-tile-quality-filters

Quality-control scripts that scan folders of GeoTIFF tiles, detect corrupted or unusable ones and move them to a separate folder before model training.
Each script is standalone: set the paths at the top and run it. Rejected tiles are moved (not deleted), so nothing is lost.

## Scripts

| Script | What it checks |
|---|---|
| `geotiff_tile_filter_any_black_pixel.py` | Strictest filter: rejects a tile if even a single pixel is near-black (value < 5). |
| `geotiff_tile_filter_black_regions.py` | Scans the tile in 32×32 windows and rejects it if any window has more than 3% dark pixels. Also checks tile size (256×256) and dark corners. Runs in parallel and writes a log file. |
| `geotiff_tile_filter_nodata_nan.py` | Rejects tiles with NaN pixels, nodata pixels (0 in all bands), wrong size, flat single-color content or near-black content. Writes a detailed log and can be stopped safely with Ctrl+C. |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python geotiff_tile_filter_nodata_nan.py
```

Edit the input/output paths at the top of each script before running it.
The folder structure of the input is preserved in the output folder.

## License

MIT
