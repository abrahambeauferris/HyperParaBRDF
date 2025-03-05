import struct
import math
import sys
import numpy as np

def convert_zemax_to_epfl(zmx_text, output_path):
    # first we take each line from the zemax file
    # we ignore empty lines or lines that start with a comment symbol
    lines = [ln.strip() for ln in zmx_text.splitlines() if ln.strip() and not ln.strip().startswith('#')]
    it = iter(lines)

    # the file starts with info about source, symmetry, spectrum type, and brdf/btdf
    source = next(it).split()[1]
    symmetry = next(it).split()[1]
    spectral_content = next(it).split()[1]
    scatter_type = next(it).split()[1]

    # next we get the count of sample rotations and incidence angles plus their values
    rot_count = int(next(it).split()[1])
    rot_values = list(map(float, next(it).split()))
    ang_count = int(next(it).split()[1])
    ang_values = list(map(float, next(it).split()))

    # same for azimuth and radial angles, which describe how light scatters
    az_count = int(next(it).split()[1])
    az_values = list(map(float, next(it).split()))
    rad_count = int(next(it).split()[1])
    rad_values = list(map(float, next(it).split()))

    # if it's xyz or tristimulus, we'll expect three data blocks
    blocks = []
    if spectral_content.lower() in ('xyz', 'tristimulus'):
        expected_blocks = ["TristimulusX", "TristimulusY", "TristimulusZ"]
    else:
        expected_blocks = [None]  # just one block for monochrome

    # now we read each data block
    for label in expected_blocks:
        # if we see a label line (like "TristimulusX"), skip it
        line = next(it)
        if label and line.startswith(label):
            line = next(it)

        # we expect a "DataBegin" line to start the data section
        assert line.startswith("DataBegin")

        block_data = {'TIS': [], 'values': {}}

        # for each rotation and incidence angle, read tis and the grid of bsdf values
        for r in range(rot_count):
            for m in range(ang_count):
                # tis line tells us how much light is scattered overall
                tis_line = next(it)
                assert tis_line.lower().startswith("tis")
                tis_val = float(tis_line.split()[1])
                block_data['TIS'].append(tis_val)

                # now we read az_count rows, each with radial data, until we have enough entries
                matrix = []
                for i in range(az_count):
                    row_vals = []
                    while len(row_vals) < rad_count:
                        data_line = next(it)
                        if data_line.lower().startswith("dataend"):
                            break
                        row_vals += list(map(float, data_line.split()))
                    # pad row if it's short
                    row_vals = row_vals[:rad_count] + [0.0]*(max(0, rad_count - len(row_vals)))
                    matrix.append(row_vals)

                block_data['values'][(r, m)] = np.array(matrix, dtype=np.float32)

        blocks.append(block_data)
        # eventually we hit a "DataEnd" line but the loop logic mostly handles it

    # now we set up arrays for the epfl format, calling our rotation angles phi_i and incidence angles theta_i
    phi_i = np.array(rot_values, dtype=np.float32)
    theta_i = np.array(ang_values, dtype=np.float32)
    num_phi, num_theta = rot_count, ang_count

    # figure out how many channels we have (1 for mono, 3 for xyz), assign placeholder wavelengths
    num_channels = len(blocks)
    if num_channels == 3:
        wavelengths = np.array([650.0, 550.0, 450.0], dtype=np.float32)
    else:
        wavelengths = np.array([555.0], dtype=np.float32)

    # epfl format uses a square grid n x n, so let's figure out the needed size
    N = max(az_count, rad_count)

    # we might need to pad azimuth or radial arrays if they're shorter than n
    az_list = np.array(az_values, dtype=np.float64)
    rad_list = np.array(rad_values, dtype=np.float64)

    if az_count < N:
        step = az_list[-1] - az_list[-2] if len(az_list) > 1 else 0
        extra = np.arange(az_list[-1] + step, az_list[-1] + step*(N - az_count + 1), step)
        az_list = np.concatenate([az_list, extra[:N - az_count]])

    if rad_count < N:
        step = rad_list[-1] - rad_list[-2] if len(rad_list) > 1 else 0
        extra = np.arange(rad_list[-1] + step, rad_list[-1] + step*(N - rad_count + 1), step)
        rad_list = np.concatenate([rad_list, extra[:N - rad_count]])

    # set up empty arrays for our epfl fields
    spectra = np.zeros((num_phi, num_theta, num_channels, N, N), dtype=np.float32)
    luminance = np.zeros((num_phi, num_theta, N, N), dtype=np.float32)
    vndf = np.zeros((num_phi, num_theta, N, N), dtype=np.float32)
    ndf = np.zeros((num_phi, num_theta), dtype=np.float32)
    sigma = np.zeros((num_phi, num_theta), dtype=np.float32)

    # fill in the data from our zemax blocks
    for r in range(num_phi):
        for m in range(num_theta):
            for ch in range(num_channels):
                mat = blocks[ch]['values'][(r, m)]
                pad = np.zeros((N, N), dtype=np.float32)
                pad[:mat.shape[0], :mat.shape[1]] = mat
                spectra[r, m, ch, :, :] = pad

            # if we have xyz, let's use the y channel as luminance, or use the only channel if monochrome
            if num_channels == 3:
                luminance[r, m, :, :] = spectra[r, m, 1, :, :]
            else:
                luminance[r, m, :, :] = spectra[r, m, 0, :, :]

            # store tis in sigma, and set ndf to the specular peak (the first cell)
            idx = r * num_theta + m
            sigma[r, m] = blocks[0]['TIS'][idx] if idx < len(blocks[0]['TIS']) else 1.0
            ndf[r, m] = float(luminance[r, m, 0, 0])

    # approximate vndf by doing brdf * cos(theta_in)
    Alpha, Delta = np.meshgrid(np.radians(az_list), np.radians(rad_list), indexing='ij')
    for r in range(num_phi):
        for m in range(num_theta):
            A = math.radians(theta_i[m])
            cosA, sinA = math.cos(A), math.sin(A)
            cos_theta_in = cosA * np.cos(Delta) + sinA * np.cos(Alpha) * np.sin(Delta)
            cos_theta_in = np.clip(cos_theta_in, 0.0, 1.0)
            vndf[r, m, :, :] = luminance[r, m, :, :] * cos_theta_in

    # add a casual text note about where this file came from
    desc_text = f"converted from zemax bsdf (symmetry={symmetry}, type={scatter_type})"
    description = np.frombuffer(desc_text.encode('utf-8'), dtype=np.uint8)

    # the jacobian is just a flag, we'll say 0 for none
    jacobian = np.array([0], dtype=np.uint8)

    # now we write everything to a binary .bsdf file
    with open(output_path, 'wb') as f:
        # write our special header
        f.write(b"tensor_file\x00")
        f.write(struct.pack('<BB', 1, 0))

        # list all the fields we plan to write, then say how many there are
        fields = [
            ("theta_i", theta_i),
            ("phi_i", phi_i),
            ("ndf", ndf),
            ("sigma", sigma),
            ("vndf", vndf),
            ("luminance", luminance),
            ("spectra", spectra),
            ("wavelengths", wavelengths),
            ("description", description),
            ("jacobian", jacobian)
        ]
        f.write(struct.pack('<I', len(fields)))

        # figure out how big all the field descriptors will be so we know where data starts
        descriptors_len = 0
        for name, arr in fields:
            arr = np.array(arr, copy=False)
            name_bytes = name.encode('utf-8')
            ndim = arr.ndim
            descriptors_len += 2 + len(name_bytes) + 2 + 1 + 8 + 8*ndim

        # total offset is header size + descriptors
        data_offset = 12 + 2 + 4 + descriptors_len
        current_offset = data_offset

        # define numeric codes for each data type
        dtype_code = {
            np.dtype('uint8'): 1,
            np.dtype('uint16'): 2,
            np.dtype('uint32'): 3,
            np.dtype('uint64'): 4,
            np.dtype('int8'): 5,
            np.dtype('int16'): 6,
            np.dtype('int32'): 7,
            np.dtype('int64'): 8,
            np.dtype('float32'): 9,
            np.dtype('float64'): 10
        }

        # write each field's descriptor
        for name, arr in fields:
            arr = np.array(arr, copy=False)
            name_bytes = name.encode('utf-8')

            # field name length and the name
            f.write(struct.pack('<H', len(name_bytes)))
            f.write(name_bytes)

            # number of dimensions, data type, offset
            f.write(struct.pack('<H', arr.ndim))
            f.write(struct.pack('<B', dtype_code[arr.dtype]))
            f.write(struct.pack('<Q', current_offset))

            # store the size of each dimension
            for dim in arr.shape:
                f.write(struct.pack('<Q', dim))

            current_offset += arr.nbytes

        # finally write the raw data for each field
        for name, arr in fields:
            arr = np.array(arr, copy=False)
            f.write(arr.tobytes())

def main():
    if len(sys.argv) != 3:
        print("usage: python convert_bsdf.py <input.zmx.bsdf> <output.epfl.bsdf>")
        sys.exit(1)

    inp_file, out_file = sys.argv[1], sys.argv[2]
    with open(inp_file, 'r') as f_in:
        zmx_text = f_in.read()

    convert_zemax_to_epfl(zmx_text, out_file)
    print(f"converted '{inp_file}' to '{out_file}'.")

if __name__ == "__main__":
    main()