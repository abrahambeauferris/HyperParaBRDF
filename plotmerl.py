import struct
import numpy as np
import matplotlib.pyplot as plt
import os

def read_merl_file(file_path):
    """
    Read a MERL-like binary file and return a numpy array of shape [theta_h, theta_d, phi_d, 3].
    
    MERL file format:
        - The first 12 bytes are three 32-bit integers (theta_h, theta_d, phi_d).
        - Followed by (theta_h * theta_d * phi_d * 3) float64 values for RGB channels.
    """
    with open(file_path, 'rb') as f:
        # Read the resolution
        header = f.read(12)
        theta_h, theta_d, phi_d = struct.unpack('3i', header)
        
        # Read the entire BRDF data
        total_values = theta_h * theta_d * phi_d * 3  # 3 channels (R, G, B)
        data = struct.unpack(f'{total_values}d', f.read(8*total_values))
    
    # Convert to numpy array and reshape
    brdf_data = np.array(data, dtype=np.float64)
    brdf_data = brdf_data.reshape((theta_h, theta_d, phi_d, 3))
    return brdf_data

def plot_merl_slices(merl_data, output_dir, file_tag="merl", phi_d_slices=(0, 45, 90)):
    """
    For each specified phi_d slice, save:
      1) A single figure showing the combined RGB map
      2) A single figure for each R, G, and B channel
    
    Each slice is of shape [theta_h, theta_d, 3].
    """
    os.makedirs(output_dir, exist_ok=True)
    
    theta_h_res, theta_d_res, phi_d_res, channels = merl_data.shape
    print(f"  -> MERL data shape for {file_tag} = {merl_data.shape}")
    
    for slice_idx in phi_d_slices:
        # Check bounds
        if slice_idx < 0 or slice_idx >= phi_d_res:
            print(f"  -> WARNING: phi_d index {slice_idx} is out of range [0, {phi_d_res-1}] for {file_tag}")
            continue
        
        # Extract the 2D slice: shape = (theta_h, theta_d, 3)
        slice_rgb = merl_data[:, :, slice_idx, :]
        
        # 1) Plot combined RGB
        plt.figure()
        plt.imshow(slice_rgb)
        plt.title(f"{file_tag} | phi_d={slice_idx} | Combined RGB")
        plt.axis("off")
        
        combined_filename = os.path.join(output_dir, f"{file_tag}_phi_{slice_idx}_combinedRGB.png")
        plt.savefig(combined_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2) Plot separate R, G, B channels
        channel_names = ["R", "G", "B"]
        for ch_idx, ch_name in enumerate(channel_names):
            channel_data = slice_rgb[:, :, ch_idx]
            
            plt.figure()
            plt.imshow(channel_data, cmap="inferno")
            plt.title(f"{file_tag} | phi_d={slice_idx} | {ch_name} channel")
            plt.axis("off")
            
            ch_filename = os.path.join(output_dir, f"{file_tag}_phi_{slice_idx}_{ch_name}.png")
            plt.savefig(ch_filename, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"  -> Plots for {file_tag} saved in: {output_dir}")

def main():
    """
    Recursively process all .binary and .fullbin files in a given directory.
    Usage:
        python batch_plot_merl_slices_recursive.py <directory>
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python batch_plot_merl_slices_recursive.py <directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"ERROR: '{input_dir}' is not a valid directory.")
        sys.exit(1)
    
    # Collect all .binary or .fullbin files (recursively)
    merl_files = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            # Check extension
            if filename.lower().endswith(".binary") or filename.lower().endswith(".fullbin"):
                full_path = os.path.join(root, filename)
                merl_files.append(full_path)

    if not merl_files:
        print(f"No .binary or .fullbin files found in '{input_dir}'")
        return
    
    print(f"Found {len(merl_files)} MERL file(s) to process in '{input_dir}' (recursively).")

    # Create a top-level 'plots' folder in the input directory
    plots_root = os.path.join(input_dir, "plots")
    os.makedirs(plots_root, exist_ok=True)
    
    # For each MERL file, read & plot slices
    for file_path in merl_files:
        print(f"\nProcessing file: {file_path}")
        
        # Read data
        merl_data = read_merl_file(file_path)
        
        # Figure out relative path (for storing in a mirror subfolder under "plots")
        rel_path = os.path.relpath(file_path, input_dir)
        # Remove file extension to define the subfolder name
        base_name, ext = os.path.splitext(rel_path)
        
        # The subdirectory for plots for this file:
        #   e.g., "plots/subdirA/subdirB/filename"
        output_subdir = os.path.join(plots_root, base_name)
        os.makedirs(output_subdir, exist_ok=True)
        
        # We label plots by the file's base name
        tag_for_plots = os.path.basename(base_name)  # just the final part
        
        # Choose whichever slices you want
        phi_d_slices = [0, 45, 90]
        
        plot_merl_slices(
            merl_data,
            output_subdir,
            file_tag=tag_for_plots,
            phi_d_slices=phi_d_slices
        )

if __name__ == "__main__":
    main()