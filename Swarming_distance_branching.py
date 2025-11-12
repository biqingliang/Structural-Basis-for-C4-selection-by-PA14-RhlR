import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import skeletonize
import pandas as pd
import tkinter as tk
import skimage
from scipy import ndimage
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.morphology import skeletonize, dilation, erosion, square
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, disk, opening

def branching_analysis(binary_image, pre_filter_size=3):
    binary_image = opening(binary_image, disk(1))
    skeleton = skeletonize(binary_image)

    struct = ndimage.generate_binary_structure(2, 2)
    neighbors = ndimage.convolve(skeleton.astype(int), struct.astype(int)) - 1
    branch_points = (neighbors > 2) & skeleton
    num_branch_points = np.sum(branch_points)
    skeleton_length = np.sum(skeleton)
    labeled, num_segments = ndimage.label(skeleton & ~branch_points)
    segment_lengths = [np.sum(labeled == i) for i in range(1, num_segments + 1) if np.sum(labeled == i) > 10]
    return {
        'num_branch_points': num_branch_points,
        'skeleton_length': skeleton_length,
        'branching_ratio': num_branch_points / max(1, skeleton_length),
        'avg_segment_length': np.mean(segment_lengths) if segment_lengths else 0,
        'skeleton': skeleton,
        'branch_points': branch_points
    }

def get_center_point_interactive(binary_mask):
    root = tk.Tk()
    root.title("Select Center Point")
    root.geometry("800x600")
    tk.Label(root, text="Click to set the center point of the swarming plate.").pack(pady=10)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(binary_mask, cmap='binary')
    ax.set_title("Click to set center point")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    center_point = [None, None]
    center_dot = None
    def on_click(event):
        nonlocal center_dot
        if event.xdata is not None and event.ydata is not None:
            center_point[0] = int(event.xdata)
            center_point[1] = int(event.ydata)
            if center_dot is not None:
                center_dot.remove()
            center_dot = ax.plot(center_point[0], center_point[1], 'ro', markersize=10)[0]
            canvas.draw()
    fig.canvas.mpl_connect('button_press_event', on_click)
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    status_label = tk.Label(root, text="")
    status_label.pack(pady=5)
    def on_confirm():
        root.quit()
    def on_save():
        if center_point[0] is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], parent=root)
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(f"Center point: ({center_point[0]}, {center_point[1]})")
                status_label.config(text=f"Saved to: {os.path.basename(save_path)}")
    tk.Button(button_frame, text="Confirm", command=on_confirm).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Save", command=on_save).pack(side=tk.LEFT, padx=10)
    root.mainloop()
    plt.close(fig)
    root.destroy()
    return tuple(center_point)

def measure_swarming_distance(binary_mask, center_point=None):
    if center_point is None:
        center_point = get_center_point_interactive(binary_mask)
    eroded = ndimage.binary_erosion(binary_mask)
    boundary = binary_mask & ~eroded
    boundary_y, boundary_x = np.where(boundary)
    distances = [np.sqrt((x - center_point[0])**2 + (y - center_point[1])**2) for x, y in zip(boundary_x, boundary_y)]
    max_distance = max(distances) if distances else 0
    return max_distance, distances, center_point

def visualize_swarming_distance(binary_mask, center_point, max_distance, sample_name, output_dir):
    distance_dir = os.path.join(output_dir, "swarming_distance")
    os.makedirs(distance_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(binary_mask, cmap='binary')
    plt.plot(center_point[0], center_point[1], 'ro', markersize=10, label='Center')
    circle = plt.Circle(center_point, max_distance, fill=False, edgecolor='red', linestyle='--', linewidth=2, label=f'Max Radius: {max_distance:.1f} px')
    plt.gca().add_patch(circle)
    eroded = ndimage.binary_erosion(binary_mask)
    boundary = binary_mask & ~eroded
    boundary_y, boundary_x = np.where(boundary)
    distances = [np.sqrt((x - center_point[0])**2 + (y - center_point[1])**2) for x, y in zip(boundary_x, boundary_y)]
    if distances:
        max_idx = np.argmax(distances)
        max_x, max_y = boundary_x[max_idx], boundary_y[max_idx]
        plt.plot([center_point[0], max_x], [center_point[1], max_y], 'g-', linewidth=2, label='Max Distance')
    plt.title(f'{sample_name}: Maximum Swarming Distance = {max_distance:.1f} pixels')
    plt.legend(loc='best')
    plt.axis('on')
    plt.tight_layout()
    plt.savefig(os.path.join(distance_dir, f"{sample_name}_swarming_distance.png"), dpi=300)
    plt.close()

def batch_measure_swarming_distances(input_files, output_dir):
    results = []
    for i, npy_file in enumerate(input_files):
        sample_name = os.path.basename(npy_file).replace(".npy", "")
        print(f"Processing: {sample_name} ({i+1}/{len(input_files)})")
        try:
            binary_mask = np.load(npy_file).astype(bool)
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("Center Point Selection", f"Select center for sample: {sample_name}")
            root.destroy()
            max_distance, distances, center_point = measure_swarming_distance(binary_mask)
            visualize_swarming_distance(binary_mask, center_point, max_distance, sample_name, output_dir)
            branching = branching_analysis(binary_mask)
            skeleton = branching['skeleton']
            branch_points = branching['branch_points']
            branching_dir = os.path.join(output_dir, "branching")
            os.makedirs(branching_dir, exist_ok=True)
            dilated_skeleton = ndimage.binary_dilation(skeleton, iterations=1)
            dilated_branch_points = ndimage.binary_dilation(branch_points, iterations=2)
            overlay = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
            overlay[dilated_skeleton] = [135, 206, 250]
            overlay[dilated_branch_points] = [255, 0, 0]
            plt.figure(figsize=(10, 8))
            plt.imshow(overlay)
            plt.title(f'{sample_name}: Skeleton with Branch Points (n={branching["num_branch_points"]})')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(branching_dir, f"{sample_name}_branching.png"), dpi=300)
            plt.close()
            results.append({
                'sample': sample_name,
                'num_branch_points': branching['num_branch_points'],
                'branching_ratio': branching['branching_ratio'],
                'avg_segment_length': branching['avg_segment_length'],
                'max_swarming_distance_px': max_distance,
                'center_x': center_point[0],
                'center_y': center_point[1]
            })
        except Exception as e:
            print(f"Error processing {sample_name}: {str(e)}")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "branching_and_distance_results.csv"), index=False)
    print("Analysis complete.")

class BacterialAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Swarming Distance and Branching Analysis")
        self.root.geometry("600x300")
        self.setup_gui()

    def setup_gui(self):
        ttk.Label(self.root, text="Input Directory (.npy files):").pack(pady=5)
        self.input_dir_var = tk.StringVar()
        ttk.Entry(self.root, textvariable=self.input_dir_var, width=60).pack()
        ttk.Button(self.root, text="Browse Input", command=self.browse_input_dir).pack(pady=5)
        ttk.Label(self.root, text="Output Directory:").pack(pady=5)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(self.root, textvariable=self.output_dir_var, width=60).pack()
        ttk.Button(self.root, text="Browse Output", command=self.browse_output_dir).pack(pady=5)
        ttk.Button(self.root, text="Run Batch Analysis", command=self.run_analysis).pack(pady=10)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var).pack()

    def browse_input_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir_var.set(directory)

    def browse_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)

    def run_analysis(self):
        input_dir = self.input_dir_var.get()
        output_dir = self.output_dir_var.get()
        if not input_dir or not output_dir:
            self.status_var.set("Error: Select both input and output directories")
            return
        input_files = glob.glob(os.path.join(input_dir, "*.npy"))
        if not input_files:
            self.status_var.set("No .npy files found in input directory.")
            return
        self.status_var.set("Running analysis...")
        self.root.update()
        os.makedirs(output_dir, exist_ok=True)
        batch_measure_swarming_distances(input_files, output_dir)
        self.status_var.set("Analysis complete.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BacterialAnalysisGUI(root)
    root.mainloop()
