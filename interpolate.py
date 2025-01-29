#!/usr/bin/env python3

import os
import shutil

# Adjust these paths to fit your directory structure
SOURCE_DIR = "data/nano"        # Where your original .brdf.binary files live
DEST_DIR = "data/nano"          # Where we'll put the dummy copies (could be the same as SOURCE_DIR)

# Define 5 "interpolation" (thickness, doping) combos
INTERPOLATIONS = [
    (180,  80),
    (190,  90),
    (210, 110),
    (220,  95),
    (160,  70),
]

# Define 5 "extrapolation" combos
EXTRAPOLATIONS = [
    (260,  80),
    (270,  50),
    (250, 180),
    (300,  60),
    (130,  40),
]

# 1) Pick a single source file or multiple source files to copy
#    For simplicity, let's pick the first .brdf.binary file in SOURCE_DIR
def find_any_brdf_binary(path):
    for fname in os.listdir(path):
        if fname.endswith(".brdf.binary"):
            return os.path.join(path, fname)
    return None

def generate_filename(thickness, doping):
    """
    Returns something like: nano_XYZBRDF160nmD80nm.brdf.binary
    so the code can parse thickness=160, doping=80 from the filename.
    """
    return f"nano_XYZBRDF{thickness}nmD{doping}nm.brdf.binary"

def create_dummy_files(source_brdf, combos, label="interp"):
    """
    Copies a source BRDF for each (thickness, doping) in combos,
    saving to DEST_DIR with a name that encodes those parameters.
    """
    for (thick, dope) in combos:
        new_name = generate_filename(thick, dope)
        new_path = os.path.join(DEST_DIR, new_name)

        print(f"Creating dummy {label}: {new_path}")
        shutil.copy2(source_brdf, new_path)  # Copy the file's contents verbatim

def main():
    # Locate a valid .brdf.binary file to copy
    src_file = find_any_brdf_binary(SOURCE_DIR)
    if not src_file:
        print(f"No .brdf.binary files found in {SOURCE_DIR}. Exiting.")
        return
    
    # 2) Create 5 interpolations
    create_dummy_files(src_file, INTERPOLATIONS, label="interp")
    
    # 3) Create 5 extrapolations
    create_dummy_files(src_file, EXTRAPOLATIONS, label="extrap")

if __name__ == "__main__":
    main()