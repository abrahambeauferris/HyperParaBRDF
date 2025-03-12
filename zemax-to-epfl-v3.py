import struct
import math
import sys
import numpy as np

def convert_zemax_to_epfl(zmx_text, output_path):

    # first, read each line but skip empties and ones starting with '#'
    lines = [ln.strip() for ln in zmx_text.splitlines()
             if ln.strip() and not ln.strip().startswith('#')]
    it = iter(lines)

    # zemax header lines we expect: source, symmetry, spectrum type, scatter type
    source = next(it).split()[1]
    symmetry = next(it).split()[1]
    spectral_content = next(it).split()[1]
    scatter_type = next(it).split()[1]

    # next, lines for rotation angles and incidence angles
    rot_count = int(next(it).split()[1])
    rot_values = list(map(float, next(it).split()))

    ang_count = int(next(it).split()[1])
    ang_values = list(map(float, next(it).split()))

    # then we have how many az and radial angles plus their actual angle lists
    az_count = int(next(it).split()[1])
    az_values = list(map(float, next(it).split()))
    rad_count = int(next(it).split()[1])
    rad_values = list(map(float, next(it).split()))

    # figure out how many channels blocks we should read (xyz if 3, else 1)
    if spectral_content.lower() in ('xyz', 'tristimulus'):
        expected_blocks = ["TristimulusX", "TristimulusY", "TristimulusZ"]
    else:
        expected_blocks = [None]

    blocks = []
    for label in expected_blocks:
        line = next(it)
        if label and line.startswith(label):
            line = next(it)
        while not line.startswith("DataBegin"):
            line = next(it).strip()
        assert line.startswith("DataBegin")

        block_data = {'TIS': [], 'values': {}}
        for r in range(rot_count):
            for a in range(ang_count):
                tis_line = next(it)
                assert tis_line.lower().startswith("tis")
                tis_val = float(tis_line.split()[1])
                block_data['TIS'].append(tis_val)

                matrix = []
                for _ in range(az_count):
                    row_vals = []
                    while len(row_vals) < rad_count:
                        data_line = next(it)
                        if data_line.lower().startswith("dataend"):
                            break
                        row_vals += list(map(float, data_line.split()))
                    row_vals = row_vals[:rad_count] + [0.0]*(max(0, rad_count - len(row_vals)))
                    matrix.append(row_vals)

                block_data['values'][(r, a)] = np.array(matrix, dtype=np.float32)
        blocks.append(block_data)

    # interpret rotation angles as theta, incidence angles as phi
    theta_i = np.array(rot_values, dtype=np.float32)
    phi_i   = np.array(ang_values, dtype=np.float32)
    num_theta, num_phi = rot_count, ang_count

    # if we have 3 channels => xyz, else 1 channel
    num_channels = len(blocks)
    if num_channels == 3:
        wavelengths = np.array([650.0, 550.0, 450.0], dtype=np.float32)
    else:
        wavelengths = np.array([555.0], dtype=np.float32)

    # build 'spectra' => shape (theta, phi, channels, rad, az)
    spectra = np.zeros((num_theta, num_phi, num_channels, rad_count, az_count), dtype=np.float32)
    sigma = np.zeros((num_theta, num_phi), dtype=np.float32)

    for t in range(num_theta):
        for p in range(num_phi):
            idx = t*num_phi + p
            for ch in range(num_channels):
                mat = blocks[ch]['values'][(t, p)]
                # mat is shape (az_count, rad_count), we want (rad_count, az_count)
                mat = mat.T
                spectra[t,p,ch,:,:] = mat

            if idx < len(blocks[0]['TIS']):
                sigma[t,p] = blocks[0]['TIS'][idx]
            else:
                sigma[t,p] = 1.0

    # luminance => (theta, phi, rad, az), pick y channel if we have 3, else channel0
    luminance = np.zeros((num_theta, num_phi, rad_count, az_count), dtype=np.float32)
    if num_channels == 3:
        luminance[:] = spectra[:, :, 1, :, :]
    else:
        luminance[:] = spectra[:, :, 0, :, :]

    # vndf => same shape, we do cos factor
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
                    c_in = cosA*math.cos(delta) + sinA*math.cos(alpha)*math.sin(delta)
                    c_in = max(c_in, 0.0)
                    vndf[t,p,r_,a_] = luminance[t,p,r_,a_] * c_in

    # build rgb => shape (theta, phi, 3, rad, az)
    if num_channels == 3:
        xyz_to_rgb = np.array([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.2040,  1.0570]
        ], dtype=np.float32)
        X = spectra[:,:,0,:,:]
        Y = spectra[:,:,1,:,:]
        Z = spectra[:,:,2,:,:]

        R = xyz_to_rgb[0,0]*X + xyz_to_rgb[0,1]*Y + xyz_to_rgb[0,2]*Z
        G = xyz_to_rgb[1,0]*X + xyz_to_rgb[1,1]*Y + xyz_to_rgb[1,2]*Z
        B = xyz_to_rgb[2,0]*X + xyz_to_rgb[2,1]*Y + xyz_to_rgb[2,2]*Z

        rgb = np.zeros((num_theta, num_phi, 3, rad_count, az_count), dtype=np.float32)
        rgb[:, :, 0, :, :] = R
        rgb[:, :, 1, :, :] = G
        rgb[:, :, 2, :, :] = B
    else:
        c = spectra[:,:,0,:,:]
        rgb = np.zeros((num_theta, num_phi, 3, rad_count, az_count), dtype=np.float32)
        rgb[:, :, 0, :, :] = c
        rgb[:, :, 1, :, :] = c
        rgb[:, :, 2, :, :] = c

    # if total number of elements in (theta, phi, rad, az) not multiple of 8 => pad rad_count
    total_elems = num_theta * num_phi * rad_count * az_count
    if total_elems % 8 != 0:
        needed_bits = 8 - (total_elems % 8)
        per_col = num_theta * num_phi * az_count
        extra_cols = (needed_bits + per_col - 1)//per_col

        new_rad_count = rad_count + extra_cols
        new_total = num_theta * num_phi * new_rad_count * az_count
        if new_total % 8 != 0:
            raise ValueError("padding logic error: still not multiple of 8")

        if len(rad_values) > 1:
            step = rad_values[-1] - rad_values[-2]
            if step == 0:
                step = 5.0
        else:
            step = 5.0

        pads = []
        current = rad_values[-1]
        for _ in range(extra_cols):
            current += step
            pads.append(current)
        rad_values = list(rad_values) + pads

        new_spectra = np.zeros((num_theta, num_phi, num_channels, new_rad_count, az_count), dtype=np.float32)
        new_luminance = np.zeros((num_theta, num_phi, new_rad_count, az_count), dtype=np.float32)
        new_vndf = np.zeros((num_theta, num_phi, new_rad_count, az_count), dtype=np.float32)
        new_rgb = np.zeros((num_theta, num_phi, 3, new_rad_count, az_count), dtype=np.float32)

        new_spectra[:,:,:,:rad_count,:] = spectra
        new_luminance[:,:,:rad_count,:] = luminance
        new_vndf[:,:,:rad_count,:] = vndf
        new_rgb[:,:,:,:rad_count,:] = rgb

        rad_count = new_rad_count
        spectra = new_spectra
        luminance = new_luminance
        vndf = new_vndf
        rgb = new_rgb

    # build valid => shape (theta, phi, rad, az)
    valid_bool = np.ones((num_theta, num_phi, rad_count, az_count), dtype=bool)
    old_original_rad = blocks[0]['values'][(0,0)].shape[1]
    if rad_count> old_original_rad:
        valid_bool[:,:,old_original_rad:, :] = False

    valid_flat = valid_bool.reshape(-1)
    valid_bits = np.packbits(valid_flat)

    # clamp negative or inf in rgb
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1e4, neginf=0.0)
    rgb = np.maximum(rgb, 0.0)

    # convert rad_values, az_values to arrays
    rad_values = np.array(rad_values, dtype=np.float32)
    az_values = np.array(az_values, dtype=np.float32)

    # short text desc
    desc_text = f"converted from zemax bsdf (symmetry={symmetry}, type={scatter_type})"
    description = np.frombuffer(desc_text.encode('utf-8'), dtype=np.uint8)
    jacobian = np.array([0], dtype=np.uint8)

    # gather fields for the final .bsdf
    fields = [
        ("theta_i", theta_i),
        ("phi_i", phi_i),
        ("wavelengths", wavelengths),
        ("spectra", spectra),
        ("luminance", luminance),
        ("sigma", sigma),
        ("vndf", vndf),
        ("rgb", rgb),
        ("valid", valid_bits),
        ("rad_values", rad_values),
        ("az_values", az_values),
        ("description", description),
        ("jacobian", jacobian),
    ]

    # write epfl binary format
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
            descriptors_len += 2 + name_len + 2 + 1 + 8 + 8*ndim

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
