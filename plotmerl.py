#!/usr/bin/env python3
import os
import sys
import numpy as np
import imageio
from PIL import Image

def read_merl_brdf(filename):
    """
    Reads a MERL binary BRDF file.
    Returns (dims, brdf) where brdf has shape (nThetaH, nThetaD, nPhiD, 3).
    Typically, dims = (90, 90, 180).
    """
    with open(filename, 'rb') as f:
        dims = np.fromfile(f, dtype=np.int32, count=3)
        nvals = dims[0] * dims[1] * dims[2] * 3
        data = np.fromfile(f, dtype=np.float64, count=nvals)
        brdf = data.reshape((dims[0], dims[1], dims[2], 3))
    return dims, brdf

def create_slice_image(brdf, phi):
    """
    For a given MERL brdf array (shape: (nThetaH, nThetaD, nPhiD, 3)) and a φ index,
    extracts the slice and returns a PIL Image of the slice.
    Orientation:
      - Horizontal axis: θₕ (0° at left, 90° at right)
      - Vertical axis: θ_d (0° at bottom, 90° at top)
    """
    # Extract slice at φ index: shape (nThetaH, nThetaD, 3)
    slice_img = brdf[:, :, phi, :]
    # Transpose so that columns = θₕ and rows = θ_d → shape becomes (nThetaD, nThetaH, 3)
    transposed = np.transpose(slice_img, (1, 0, 2))
    # Flip vertically so that bottom corresponds to θ_d = 0° and top to θ_d = 90°
    oriented = np.flipud(transposed)
    # Normalize the slice to the [0, 255] range for display
    min_val = oriented.min()
    max_val = oriented.max()
    norm = (oriented - min_val) / (max_val - min_val + 1e-8) * 255
    norm = norm.astype(np.uint8)
    # Convert to PIL Image
    image = Image.fromarray(norm, mode='RGB')
    return image

def process_merl_file(input_file, output_dir, slice_list):
    """
    Processes a single MERL .binary file by extracting and saving PNG images for the specified φ indices.
    The output image filenames are based on the original binary file's base name and the φ value.
    """
    dims, brdf = read_merl_brdf(input_file)
    base = os.path.splitext(os.path.basename(input_file))[0]
    for phi in slice_list:
        if phi < 0 or phi >= dims[2]:
            print(f"Warning: φ index {phi} is out of range (0 to {dims[2]-1}) in {input_file}. Skipping.")
            continue
        image = create_slice_image(brdf, phi)
        out_filename = f"{base}_phi_{phi}.png"
        out_path = os.path.join(output_dir, out_filename)
        os.makedirs(output_dir, exist_ok=True)
        image.save(out_path)
        print(f"Saved {out_path}")

def main():
    """
    Usage:
      python generate_key_slices.py <input_dir> <output_dir> [<slices>]
      
    <slices> is an optional comma-separated list of φ indices (e.g., "60,90,120").
    If omitted, the default is "60,90,120".
    
    The script processes every .binary file in <input_dir> (recursively) and writes the resulting PNG images to <output_dir>.
    """
    if len(sys.argv) < 3:
        print("Usage: python generate_key_slices.py <input_dir> <output_dir> [<slices>]")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    slices_arg = sys.argv[3] if len(sys.argv) > 3 else "60,90,120"
    try:
        slice_list = [int(s.strip()) for s in slices_arg.split(",")]
    except Exception as e:
        print("Error parsing slices. Please supply a comma-separated list of integers.")
        sys.exit(1)
    
    # Recursively process every .binary file in input_dir
    for dirpath, _, filenames in os.walk(input_dir):
        for fname in filenames:
            if fname.lower().endswith(".binary"):
                full_path = os.path.join(dirpath, fname)
                print(f"Processing {full_path}")
                process_merl_file(full_path, output_dir, slice_list)

if __name__ == "__main__":
    main()