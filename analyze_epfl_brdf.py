#!/usr/bin/env python3

import sys
import numpy as np
import mitsuba as mi

def analyze_epfl_brdf(file_path):
    """
    Loads an EPFL .brdf file (or .epfl.brdf) via Mitsuba's TensorFile,
    then verifies the presence, shape, and data ranges of standard fields
    (phi_i, theta_i, spectra/rgb, sigma, luminance, vndf, ndf, valid, etc.).
    Prints extremely detailed feedback for diagnosing "dotty" or incorrect
    measured BSDF render results.
    """
    print(f"\n=== Analyzing EPFL BRDF file: '{file_path}' ===")
    try:
        tf = mi.util.TensorFile(file_path, mode='r')
    except Exception as e:
        print(f"[ERROR] Could not open '{file_path}': {e}")
        return

    # List all fields present
    fields_present = tf.fields()
    print(f"\nFound {len(fields_present)} fields in '{file_path}':")
    for f in fields_present:
        print("  -", f)

    # The typical fields expected by Mitsuba's measured plugin
    expected_fields = [
        "phi_i", "theta_i", "wavelengths", "spectra",
        "sigma", "luminance", "vndf", "ndf", "rgb",
        "valid", "rad_values", "az_values", "description", "jacobian"
    ]

    # Check presence of each standard field
    missing = []
    for ef in expected_fields:
        if ef not in fields_present:
            missing.append(ef)
    if missing:
        print("\n[WARNING] Missing standard field(s):", missing)
    else:
        print("\nAll standard fields appear to be present.")

    # Helper to fetch a field or None if missing
    def get_field(name):
        if name in fields_present:
            return tf.field(name)
        return None

    # We'll gather shape info for phi_i, theta_i, rad_values, az_values
    # to check consistency in 4D fields
    phi_i_field = get_field("phi_i")
    theta_i_field = get_field("theta_i")
    rad_values_field = get_field("rad_values")
    az_values_field = get_field("az_values")

    # We'll store dimension sizes
    num_phi = None
    num_theta = None
    rad_count = None
    az_count = None

    # We'll define a helper to print numeric array stats
    def print_array_stats(name, arr):
        """Print shape, dtype, min, max for a numeric array."""
        arr_np = np.array(arr, copy=False)
        print(f"  {name}: shape={arr_np.shape}, dtype={arr_np.dtype}, ",
              f"min={arr_np.min():.4g}, max={arr_np.max():.4g}, ",
              f"size={arr_np.size}")

    # 1) phi_i
    if phi_i_field is not None:
        arr = phi_i_field.data
        print("\n--- phi_i ---")
        print_array_stats("phi_i", arr)
        if arr.ndim == 1:
            num_phi = arr.shape[0]
            print(f"  => Interpreted as having {num_phi} incidence angles.")
        else:
            print("[WARNING] phi_i is not 1D. Mitsuba expects shape (N,).")
    else:
        print("\n[WARNING] phi_i field missing; can't verify incidence angles.")

    # 2) theta_i
    if theta_i_field is not None:
        arr = theta_i_field.data
        print("\n--- theta_i ---")
        print_array_stats("theta_i", arr)
        if arr.ndim == 1:
            num_theta = arr.shape[0]
            print(f"  => Interpreted as having {num_theta} rotation angles.")
        else:
            print("[WARNING] theta_i is not 1D. Mitsuba expects shape (M,).")
    else:
        print("\n[WARNING] theta_i field missing; can't verify rotation angles.")

    # 3) rad_values
    if rad_values_field is not None:
        arr = rad_values_field.data
        print("\n--- rad_values ---")
        print_array_stats("rad_values", arr)
        if arr.ndim == 1:
            rad_count = arr.shape[0]
            print(f"  => Interpreted as rad_count={rad_count}.")
        else:
            print("[WARNING] rad_values is not 1D.")
    else:
        print("\n[WARNING] rad_values field missing; can't verify radial dimension.")

    # 4) az_values
    if az_values_field is not None:
        arr = az_values_field.data
        print("\n--- az_values ---")
        print_array_stats("az_values", arr)
        if arr.ndim == 1:
            az_count = arr.shape[0]
            print(f"  => Interpreted as az_count={az_count}.")
        else:
            print("[WARNING] az_values is not 1D.")
    else:
        print("\n[WARNING] az_values field missing; can't verify azimuth dimension.")

    # We'll define a function to check 4D shape => (phi, theta, rad, az)
    def check_4d_field(name):
        field = get_field(name)
        if not field:
            return
        arr = field.data
        arr_np = np.array(arr, copy=False)
        print(f"\n--- {name} ---")
        print_array_stats(name, arr_np)
        if arr_np.ndim != 4:
            print(f"[WARNING] {name} has {arr_np.ndim}D shape; Mitsuba expects 4D (phi, theta, rad, az).")
            return
        # Check if it matches (num_phi, num_theta, rad_count, az_count)
        shape_ok = True
        if num_phi is not None and arr_np.shape[0] != num_phi:
            print(f"[WARNING] {name} shape[0]={arr_np.shape[0]}, but phi_i={num_phi}. Mismatch!")
            shape_ok = False
        if num_theta is not None and arr_np.shape[1] != num_theta:
            print(f"[WARNING] {name} shape[1]={arr_np.shape[1]}, but theta_i={num_theta}. Mismatch!")
            shape_ok = False
        if rad_count is not None and arr_np.shape[2] != rad_count:
            print(f"[WARNING] {name} shape[2]={arr_np.shape[2]}, but rad_count={rad_count}. Mismatch!")
            shape_ok = False
        if az_count is not None and arr_np.shape[3] != az_count:
            print(f"[WARNING] {name} shape[3]={arr_np.shape[3]}, but az_count={az_count}. Mismatch!")
            shape_ok = False
        if shape_ok:
            print(f"  => {name} shape matches (phi, theta, rad, az) as expected.")

    # We'll define a function to check 5D shape => (phi, theta, channels, rad, az)
    def check_5d_field(name):
        field = get_field(name)
        if not field:
            return
        arr = field.data
        arr_np = np.array(arr, copy=False)
        print(f"\n--- {name} ---")
        print_array_stats(name, arr_np)
        if arr_np.ndim != 5:
            print(f"[WARNING] {name} has {arr_np.ndim}D shape; Mitsuba often expects 5D for spectral data (phi,theta,channels,rad,az).")
            return
        # Check if it matches (phi, theta, ???, rad, az)
        shape_ok = True
        if num_phi is not None and arr_np.shape[0] != num_phi:
            print(f"[WARNING] {name} shape[0]={arr_np.shape[0]}, but phi_i={num_phi}. Mismatch!")
            shape_ok = False
        if num_theta is not None and arr_np.shape[1] != num_theta:
            print(f"[WARNING] {name} shape[1]={arr_np.shape[1]}, but theta_i={num_theta}. Mismatch!")
            shape_ok = False
        if rad_count is not None and arr_np.shape[3] != rad_count:
            print(f"[WARNING] {name} shape[3]={arr_np.shape[3]}, but rad_count={rad_count}. Mismatch!")
            shape_ok = False
        if az_count is not None and arr_np.shape[4] != az_count:
            print(f"[WARNING] {name} shape[4]={arr_np.shape[4]}, but az_count={az_count}. Mismatch!")
            shape_ok = False
        if shape_ok:
            print(f"  => {name} shape matches (phi,theta,channels,rad,az) as expected.")

    # 5) Check "sigma" => shape (phi, theta)
    sigma_field = get_field("sigma")
    if sigma_field:
        arr = sigma_field.data
        arr_np = np.array(arr, copy=False)
        print("\n--- sigma ---")
        print_array_stats("sigma", arr_np)
        if arr_np.ndim != 2:
            print("[WARNING] sigma should be 2D: (phi,theta).")
        else:
            if num_phi is not None and arr_np.shape[0] != num_phi:
                print(f"[WARNING] sigma shape[0]={arr_np.shape[0]}, but phi_i={num_phi}. Mismatch!")
            if num_theta is not None and arr_np.shape[1] != num_theta:
                print(f"[WARNING] sigma shape[1]={arr_np.shape[1]}, but theta_i={num_theta}. Mismatch!")

    # 6) Check 4D fields: vndf, ndf, luminance
    check_4d_field("vndf")
    check_4d_field("ndf")
    check_4d_field("luminance")

    # 7) Check 5D fields: spectra, rgb
    check_5d_field("spectra")
    check_5d_field("rgb")

    # 8) Check "valid" bitmask
    valid_field = get_field("valid")
    if valid_field:
        valid_data = np.array(valid_field.data, copy=False)
        print("\n--- valid ---")
        print(f"  shape={valid_data.shape}, dtype={valid_data.dtype}, size={valid_data.size} bytes")

        # Attempt to count how many bits are set
        bits = np.unpackbits(valid_data)
        total_bits = bits.size
        true_bits = bits.sum()
        print(f"  => Unpacked bits: {total_bits} total bits, #true bits={true_bits}")

        # If we have (phi,theta,rad,az) known, see if it matches
        if num_phi and num_theta and rad_count and az_count:
            needed_bits = num_phi * num_theta * rad_count * az_count
            if needed_bits != total_bits:
                print(f"[WARNING] valid bitmask has {total_bits} bits, but (phi*theta*rad*az)={needed_bits}.")
                leftover = total_bits - needed_bits
                if leftover > 0:
                    print("  => Possibly leftover bits for Mitsuba padding. That can cause brdf-loader's visualize script to fail unless truncated.")
            else:
                print(f"  => The bitmask size exactly matches (phi*theta*rad*az) = {needed_bits} bits.")
    else:
        print("\n[WARNING] valid field missing; can't check coverage bitmask.")

    # 9) If there's a "description" or "jacobian", we just print stats
    desc_field = get_field("description")
    if desc_field:
        arr = desc_field.data
        arr_np = np.array(arr, copy=False)
        print("\n--- description ---")
        print_array_stats("description", arr_np)
        try:
            text = arr_np.tobytes().decode('utf-8', errors='replace')
            print(f"  => Decoded text: '{text}'")
        except:
            print("  => Could not decode as UTF-8 text.")
    else:
        print("\nNo 'description' field found.")

    jac_field = get_field("jacobian")
    if jac_field:
        arr = jac_field.data
        arr_np = np.array(arr, copy=False)
        print("\n--- jacobian ---")
        print_array_stats("jacobian", arr_np)
    else:
        print("\nNo 'jacobian' field found.")

    print("\n=== Analysis complete. If you see [WARNING] messages above, investigate those. ===\n")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <epfl.brdf file>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_epfl_brdf(file_path)

if __name__ == "__main__":
    main()
