import os
import cv2
import numpy as np
from tkinter import filedialog, Tk
from glob import glob
import csv

def run_analysis():
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select fluorescence image folder")
    if not folder:
        print("Folder not selected")
        return

    output_dir = os.path.join(folder, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)

    tif_files = sorted(glob(os.path.join(folder, "*.tif")))
    if not tif_files:
        print("No .tif files found")
        return

    filenames, pixel_intensity = [], []

    for fpath in tif_files:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image: {fname}")
            continue

        nonzero_pixels = img[img > 0]
        intensity = np.sum(nonzero_pixels)
        pixel_intensity.append(intensity)
        filenames.append(fname)

    output_file = os.path.join(output_dir, "intensity_sum.csv")
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "PixelSum"])
        for fname, intensity in zip(filenames, pixel_intensity):
            writer.writerow([fname, intensity])

    print("Analysis results saved to:", output_file)

if __name__ == "__main__":
    run_analysis()
