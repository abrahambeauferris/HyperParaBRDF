#!/usr/bin/env python3
"""
Zemax-to-EPFL BSDF conversion script (isotropic, RGB, with debug and fixed NDF)

This script converts a measured Zemax BSDF file to the EPFL format,
assuming the input is isotropic and structured as if it were tristimulus (XYZ)
but with values that are already linear RGB (so no XYZ→RGB conversion is performed).
We force the scattering data onto a square grid and swap the angular axes so that the first axis is φ (incidence)
and the second axis is θ (rotation). For compatibility with Mitsuba’s measured BSDF loader,
the NDF field is extracted as a 2D array (taken from the first measurement).
Debug messages are printed at each major step.

Usage:
    python convert_bsdf.py <input_zmx_file> <output_epfl_file>
"""

import struct
import math
import sys
import numpy as np

def debug_print(msg):
    print("[DEBUG]", msg)

def swap_angular_axes(arr):
    """Swap the first two axes if the array has at least 2 dimensions."""
    if arr.ndim >= 2:
        return np.transpose(arr, (1, 0) + tuple(range(2, arr.ndim)))
    else:
        return arr

def duplicate_theta(theta, factor=2):
    """If theta has length 1, duplicate its value to create a vector of given factor."""
    if len(theta) == 1:
        return np.array([theta[0]] * factor, dtype=np.float32)
    return theta

def duplicate_along_axis(arr, axis, factor=2):
    """Duplicate the array along the given axis if its size is 1."""
    if arr.shape[axis] == 1:
        reps = [1] * arr.ndim
        reps[axis] = factor
        return np.repeat(arr, factor, axis=axis)
    return arr

def convert_zemax_to_epfl(zmx_text, output_path):
    # Read nonempty, non-comment lines.
    lines = [ln.strip() for ln in zmx_text.splitlines() 
             if ln.strip() and not ln.strip().startswith('#')]
    it = iter(lines)

    # Read header: source, symmetry, spectral content, scatter type.
    source = next(it).split()[1]
    symmetry = next(it).split()[1]
    spectral_content = next(it).split()[1]
    scatter_type = next(it).split()[1]
    debug_print(f"Header: source={source}, symmetry={symmetry}, spectral_content={spectral_content}, scatter_type={scatter_type}")

    # Read rotation angles and incidence angles.
    orig_rot_count = int(next(it).split()[1])
    orig_rot_values = list(map(float, next(it).split()))
    # For isotropy, use only the first rotation; if only one, duplicate it.
    if len(orig_rot_values) < 2:
        rot_values = [orig_rot_values[0]] * 2
        rot_count = 2
    else:
        rot_count = 1
        rot_values = [orig_rot_values[0]]
    debug_print(f"Rotation angles (theta): {rot_values} (original: {orig_rot_values})")

    ang_count = int(next(it).split()[1])
    ang_values = list(map(float, next(it).split()))
    debug_print(f"Incidence angles (phi): {ang_values}")

    # Read azimuth and radial angles.
    az_count = int(next(it).split()[1])
    az_values = list(map(float, next(it).split()))
    rad_count = int(next(it).split()[1])
    rad_values = list(map(float, next(it).split()))
    debug_print(f"Azimuth angles: {az_values}")
    debug_print(f"Radial angles: {rad_values}")

    # Enforce a square grid.
    new_dim = max(rad_count, az_count)
    debug_print(f"Enforcing square grid: new_dim = {new_dim} (from rad_count={rad_count}, az_count={az_count})")
    rad_count = new_dim
    az_count = new_dim

    # Determine expected channel blocks.
    if spectral_content.lower() in ('xyz', 'tristimulus'):
        expected_blocks = ["TristimulusX", "TristimulusY", "TristimulusZ"]
    elif spectral_content.lower() in ('monochrome', 'rgb'):
        expected_blocks = [None, None, None]
    else:
        expected_blocks = [None, None, None]

    blocks = []
    for label in expected_blocks:
        line = next(it)
        if label and line.startswith(label):
            line = next(it)
        while not line.startswith("DataBegin"):
            line = next(it).strip()
        assert line.startswith("DataBegin")
        block_data = {'TIS': [], 'values': {}}
        measurements_count = 0
        for r in range(rot_count):
            for a in range(ang_count):
                line = next(it)
                # If a DataEnd is encountered prematurely, break out.
                if line.lower().startswith("dataend"):
                    debug_print("Encountered DataEnd when expecting TIS; stopping measurement reading for this block.")
                    break
                assert line.lower().startswith("tis"), f"Expected TIS line, got: {line}"
                tis_val = float(line.split()[1])
                block_data['TIS'].append(tis_val)
                measurements_count += 1
                matrix = []
                for _ in range(az_count):
                    row_vals = []
                    while len(row_vals) < rad_count:
                        data_line = next(it)
                        if data_line.lower().startswith("dataend"):
                            break
                        row_vals += list(map(float, data_line.split()))
                    row_vals = row_vals[:rad_count] + [0.0]*(rad_count - len(row_vals))
                    matrix.append(row_vals)
                block_data['values'][(r, a)] = np.array(matrix, dtype=np.float32)
            if measurements_count < (r+1)*ang_count:
                debug_print("Incomplete measurement block; ending early.")
                break
        debug_print(f"Read {measurements_count} measurements (expected {ang_count * rot_count}) in this block.")
        if measurements_count < ang_count * rot_count and measurements_count > 0:
            last_tis = block_data['TIS'][-1]
            last_mat = block_data['values'][(0, measurements_count - 1)]
            for i in range(measurements_count, ang_count * rot_count):
                block_data['TIS'].append(last_tis)
                block_data['values'][(0, i)] = last_mat
            debug_print(f"Duplicated last measurement to reach {ang_count * rot_count} measurements.")
        blocks.append(block_data)
    debug_print(f"Read {len(blocks)} channel block(s)")

    # Convert rotation angles to theta and incidence angles to phi.
    theta_i = np.array(rot_values, dtype=np.float32)  # shape (rot_count,)
    phi_i   = np.array(ang_values, dtype=np.float32)    # shape (ang_count,)
    num_theta, num_phi = rot_count, ang_count
    debug_print(f"theta_i shape: {theta_i.shape}, phi_i shape: {phi_i.shape}")

    # Force 3 channels.
    num_channels = len(blocks)
    if num_channels != 3:
        debug_print(f"Warning: Expected 3 channel blocks but got {num_channels}; duplicating single channel.")
        num_channels = 3
        single = blocks[0]
        blocks = [single, single, single]
        wavelengths = np.array([555.0, 555.0, 555.0], dtype=np.float32)
    else:
        wavelengths = np.array([650.0, 550.0, 450.0], dtype=np.float32)
    debug_print(f"Wavelengths: {wavelengths}")

    # Build 'spectra' tensor: shape (theta, phi, channels, rad, az).
    spectra = np.zeros((num_theta, num_phi, num_channels, rad_count, az_count), dtype=np.float32)
    sigma = np.zeros((num_theta, num_phi), dtype=np.float32)
    for t in range(num_theta):
        for p in range(num_phi):
            idx = t * num_phi + p
            for ch in range(num_channels):
                # For isotropy, use only rotation index 0.
                mat = blocks[ch]['values'][(0, p)]
                spectra[t, p, ch, :, :] = mat.T  # (az, rad) -> (rad, az)
            if idx < len(blocks[0]['TIS']):
                sigma[t, p] = blocks[0]['TIS'][idx]
            else:
                sigma[t, p] = 1.0
    debug_print(f"Spectra shape: {spectra.shape}")
    debug_print(f"Sigma shape: {sigma.shape}")

    # Compute luminance: use channel 1 (Y) if available.
    luminance = np.zeros((num_theta, num_phi, rad_count, az_count), dtype=np.float32)
    if num_channels == 3:
        luminance[:] = spectra[:, :, 1, :, :]
    else:
        luminance[:] = spectra[:, :, 0, :, :]
    debug_print(f"Luminance shape: {luminance.shape}")

    # Compute VNDF with cosine weighting.
    vndf = np.zeros_like(luminance)
    alpha_grid, delta_grid = np.meshgrid(
        np.radians(az_values), np.radians(rad_values), indexing='ij'
    )
    alpha_grid = alpha_grid.astype(np.float32)
    delta_grid = delta_grid.astype(np.float32)
    for t in range(num_theta):
        for p in range(num_phi):
            a_val = math.radians(phi_i[p])
            cosA, sinA = math.cos(a_val), math.sin(a_val)
            for r_ in range(rad_count):
                for a_ in range(az_count):
                    alpha = alpha_grid[a_, r_]
                    delta = delta_grid[a_, r_]
                    c_in = cosA * math.cos(delta) + sinA * math.cos(alpha) * math.sin(delta)
                    c_in = max(c_in, 0.0)
                    vndf[t, p, r_, a_] = luminance[t, p, r_, a_] * c_in
    debug_print(f"VNDF shape: {vndf.shape}")

    # PATCH: Compute NDF field.
    # The measured BSDF loader expects NDF as a 2D array.
    # For an isotropic measurement, we simply take the valid region from the first measurement.
    old_original_rad = blocks[0]['values'][(0, 0)].shape[1]
    debug_print(f"Old original rad_count (valid region): {old_original_rad}")
    # Extract a 2D NDF: take the first measurement's NDF.
    ndf_fixed = np.ones((old_original_rad, old_original_rad), dtype=np.float32)
    debug_print(f"Fixed NDF shape: {ndf_fixed.shape}")

    # For already linear RGB, set rgb equal to spectra.
    rgb = spectra.copy()
    debug_print(f"RGB shape: {rgb.shape}")

    debug_print(f"Final rad_count: {rad_count}, az_count: {az_count}")

    # Build valid mask: shape (theta, phi, rad, az)
    valid_bool = np.ones((num_theta, num_phi, rad_count, az_count), dtype=bool)
    if rad_count > old_original_rad:
        valid_bool[:, :, old_original_rad:, :] = False
    valid_flat = valid_bool.reshape(-1)
    valid_bits = np.packbits(valid_flat)
    debug_print(f"Valid mask shape: {valid_bool.shape}, total valid elements: {np.sum(valid_bool)}")

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1e4, neginf=0.0)
    rgb = np.maximum(rgb, 0.0)

    rad_values = np.array(rad_values, dtype=np.float32)
    az_values = np.array(az_values, dtype=np.float32)
    debug_print(f"rad_values shape: {rad_values.shape}, az_values shape: {az_values.shape}")

    desc_text = f"converted from zemax bsdf (symmetry={symmetry}, type={scatter_type})"
    description = np.frombuffer(desc_text.encode('utf-8'), dtype=np.uint8)
    jacobian = np.array([0], dtype=np.uint8)
    debug_print(f"Description: {desc_text}, length: {description.size}")

    # --- Swap angular axes ---
    spectra = swap_angular_axes(spectra)
    sigma = swap_angular_axes(sigma)
    luminance = swap_angular_axes(luminance)
    vndf = swap_angular_axes(vndf)
    # ndf_fixed remains 2D (no swap needed)
    rgb = swap_angular_axes(rgb)
    debug_print("Swapped angular axes for spectra, sigma, luminance, vndf, and rgb.")

    # --- Duplicate theta dimension if necessary ---
    if theta_i.shape[0] == 1:
        theta_i = duplicate_theta(theta_i, factor=2)
        debug_print(f"Duplicated theta_i: new shape {theta_i.shape}")
        spectra = duplicate_along_axis(spectra, axis=1, factor=2)
        sigma = duplicate_along_axis(sigma, axis=0, factor=2)
        luminance = duplicate_along_axis(luminance, axis=0, factor=2)
        vndf = duplicate_along_axis(vndf, axis=0, factor=2)
        rgb = duplicate_along_axis(rgb, axis=0, factor=2)
        debug_print("Duplicated theta axis for spectra, sigma, luminance, vndf, and rgb.")

    # Final expected field order (per EPFL spec):
    # phi_i, theta_i, wavelengths, spectra, sigma, luminance, vndf, ndf, rgb, valid, rad_values, az_values, description, jacobian
    fields = [
        ("phi_i", phi_i),
        ("theta_i", theta_i),
        ("wavelengths", wavelengths),
        ("spectra", spectra),
        ("sigma", sigma),
        ("luminance", luminance),
        ("vndf", vndf),
        ("ndf", ndf_fixed),  # Write the 2D NDF
        ("rgb", rgb),
        ("valid", valid_bits),
        ("rad_values", rad_values),
        ("az_values", az_values),
        ("description", description),
        ("jacobian", jacobian),
    ]

    debug_print("Final fields:")
    for name, arr in fields:
        arr = np.array(arr, copy=False)
        debug_print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, total values={arr.size}")

    with open(output_path, 'wb') as f:
        f.write(b"tensor_file\x00")
        f.write(struct.pack('<BB', 1, 0))
        f.write(struct.pack('<I', len(fields)))

        dtype_map = {
            np.dtype('uint8'): 1,
            np.dtype('int8'): 2,
            np.dtype('uint16'): 3,
            np.dtype('int16'): 4,
            np.dtype('uint32'): 5,
            np.dtype('int32'): 6,
            np.dtype('uint64'): 7,
            np.dtype('int64'): 8,
            np.dtype('float16'): 9,
            np.dtype('float32'): 10,
            np.dtype('float64'): 11
        }

        descriptors_len = 0
        for name, arr in fields:
            arr = np.array(arr, copy=False)
            ndim = arr.ndim
            name_len = len(name.encode('utf-8'))
            descriptors_len += 2 + name_len + 2 + 1 + 8 + 8 * ndim

        data_offset = 12 + 2 + 4 + descriptors_len
        current_offset = data_offset

        for name, arr in fields:
            arr = np.array(arr, copy=False)
            name_bytes = name.encode('utf-8')
            ndim = arr.ndim

            f.write(struct.pack('<H', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<H', ndim))

            if arr.dtype not in dtype_map:
                raise ValueError(f"unsupported dtype: {arr.dtype}")
            f.write(struct.pack('<B', dtype_map[arr.dtype]))

            f.write(struct.pack('<Q', current_offset))
            for s in arr.shape:
                f.write(struct.pack('<Q', s))
            current_offset += arr.nbytes

        for _, arr in fields:
            arr = np.array(arr, copy=False)
            f.write(arr.tobytes())

    debug_print(f"Wrote tensor file to '{output_path}' with total size {current_offset} bytes.")

def main():
    if len(sys.argv) != 3:
        print("usage: python convert_bsdf.py <input.zmx.bsdf> <output.epfl.bsdf>")
        sys.exit(1)
    inp, out = sys.argv[1], sys.argv[2]
    with open(inp, 'r') as f:
        zmx_text = f.read()
    convert_zemax_to_epfl(zmx_text, out)
    print(f"converted '{inp}' to '{out}'.")

if __name__ == "__main__":
    main()
