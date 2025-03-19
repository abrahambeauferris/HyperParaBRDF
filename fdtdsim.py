#!/usr/bin/env python3
"""
End-to-End Synthetic Nano Material Dataset Generation using MEEP

This script uses MEEP to simulate a nanohole array at normal incidence.
It measures the broadband reflectance over the wavelength range 400–700 nm,
forcing a common spectral resolution of 31 wavelengths (400, 410, ..., 700 nm)
and an angular resolution of 91 samples (0° to 90° in 1° increments).

It then exports each simulated BSDF in:
 • A MERL–like binary file (with header and 4D array: [1, nThetaD, 1, 3])
 • An EPFL–like CSV file (rows = angular samples, columns = spectral reflectance)

Adjust geometry, material models, and simulation parameters as needed.
"""

import meep as mp
import numpy as np
import struct
import json
import os
from scipy.interpolate import interp1d

# ============================
# 1. MEEP Simulation Function
# ============================

def simulate_spectral_bsdf_meep(params):
    """
    Simulate a nanohole array using MEEP FDTD and compute its reflectance spectrum.
    
    Parameters (in nm):
      - "film_thickness": film thickness (nm)
      - "period": lattice period (nm)
      - "diameter": hole diameter (nm)
      - "material": e.g. "Ag" (for silver; here a dummy metal model is used)
      
    Returns a dictionary containing:
      - "inc_angle_deg": incident angle (0° assumed)
      - "theta_out_deg": 91 angular samples from 0° to 90°
      - "wavelengths_nm": 31 wavelengths from 400 to 700 nm
      - "BRDF": a 2D array of shape (31, 91) with reflectance values
      - "params": the input parameters (for record)
    """
    # Convert nm to µm (MEEP default units)
    period = params['period'] / 1000.0         # µm
    film_thickness = params['film_thickness'] / 1000.0   # µm
    diameter = params['diameter'] / 1000.0       # µm
    
    resolution = 200  # pixels per µm
    
    # Define simulation cell: size in x,y equal to period; z covers film and extra space.
    cell_size = mp.Vector3(period, period, film_thickness + 1.0)
    
    # Define material: using a dummy metal model for silver ("Ag")
    if params['material'] == 'Ag':
        # Dummy Drude model parameters (placeholders)
        metal = mp.Medium(epsilon=1, E1=-10, E2=0)
    else:
        metal = mp.Medium(epsilon=1)
    
    # Geometry: a block (metal film) with a cylindrical hole (nanohole)
    geometry = [
        mp.Block(size=mp.Vector3(period, period, film_thickness),
                 center=mp.Vector3(0, 0, 0),
                 material=metal),
        mp.Cylinder(radius=diameter/2, height=mp.inf,
                    center=mp.Vector3(0, 0, 0),
                    material=mp.air)
    ]
    
    # Wavelength range: 400 nm to 700 nm => 0.4 µm to 0.7 µm.
    # Corresponding frequencies: f = 1/λ (µm^-1)
    fmin = 1/0.7   # ~1.4286
    fmax = 1/0.4   # ~2.5
    fcen = (fmin + fmax) / 2  # center frequency ~1.9643 µm^-1
    df = fmax - fmin         # frequency width ~1.0714 µm^-1
    
    # Use 31 frequency points so that we can sample the spectrum uniformly.
    nfreq = 31
    sources = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                           component=mp.Ey,
                           center=mp.Vector3(0, 0, film_thickness/2 + 0.5),
                           size=mp.Vector3(period, period, 0))]
    
    pml_layers = [mp.PML(1.0)]
    
    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        sources=sources,
                        boundary_layers=pml_layers,
                        resolution=resolution)
    
    # Add a flux region above the film to record reflected power.
    refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, film_thickness/2 + 0.75),
                             size=mp.Vector3(period, period, 0))
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    
    sim.run(until=200)
    
    # Retrieve flux spectrum (frequencies and fluxes).
    flux_freqs = np.array(mp.get_flux_freqs(refl))
    flux_data = np.array(mp.get_fluxes(refl))
    
    # Convert frequencies (µm^-1) to wavelengths (µm): λ = 1/f.
    wavelengths_sim = 1.0 / flux_freqs  # in µm
    # Convert to nm.
    wavelengths_sim_nm = wavelengths_sim * 1000
    
    # We desire a fixed spectral sampling resolution: wavelengths from 400 to 700 nm with 10 nm steps.
    target_wavelengths = np.arange(400, 701, 10)  # 31 values
    
    # Interpolate the measured reflectance to the target wavelengths.
    interp_func = interp1d(wavelengths_sim_nm, flux_data, kind='linear', fill_value="extrapolate")
    refl_target = interp_func(target_wavelengths)
    
    # For simplicity, assume isotropic reflectance (no angular variation);
    # replicate the spectrum over 91 angular samples (0° to 90° in 1° increments).
    theta_out = np.linspace(0, 90, 91)
    brdf = np.tile(refl_target[:, np.newaxis], (1, len(theta_out)))
    
    bsdf = {
        "inc_angle_deg": 0,
        "theta_out_deg": theta_out,
        "wavelengths_nm": target_wavelengths,
        "BRDF": brdf,
        "params": params
    }
    return bsdf

# ============================
# 2. Spectral Conversion and Export Functions
# ============================
# Define simplified CIE 1931 matching functions and D65 illuminant (400–700 nm, 10 nm steps).
cie_lambda = np.linspace(400, 700, 11)
cie_x = np.array([0.014, 0.134, 0.283, 0.348, 0.336, 0.290, 0.195, 0.095, 0.032, 0.004, 0.000])
cie_y = np.array([0.000, 0.004, 0.032, 0.095, 0.195, 0.290, 0.336, 0.348, 0.283, 0.134, 0.014])
cie_z = np.array([0.067, 0.238, 0.503, 0.654, 0.563, 0.348, 0.164, 0.061, 0.015, 0.002, 0.000])
d65 = np.array([82.75, 91.49, 93.43, 86.69, 104.87, 117.01, 117.81, 114.86, 115.92, 108.81, 105.35])

def spectral_to_rgb(wavelengths, values):
    """
    Convert a spectral reflectance (values at given wavelengths) into linear RGB.
    Interpolates the CIE functions and D65 values if the wavelength grid differs.
    """
    if len(wavelengths) != len(cie_lambda):
        cie_x_interp = np.interp(wavelengths, cie_lambda, cie_x)
        cie_y_interp = np.interp(wavelengths, cie_lambda, cie_y)
        cie_z_interp = np.interp(wavelengths, cie_lambda, cie_z)
        d65_interp   = np.interp(wavelengths, cie_lambda, d65)
    else:
        cie_x_interp = cie_x
        cie_y_interp = cie_y
        cie_z_interp = cie_z
        d65_interp   = d65

    X = np.dot(values * d65_interp, cie_x_interp)
    Y = np.dot(values * d65_interp, cie_y_interp)
    Z = np.dot(values * d65_interp, cie_z_interp)
    norm = np.dot(d65_interp, cie_y_interp)
    X /= norm; Y /= norm; Z /= norm
    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]])
    rgb = np.dot(M, np.array([X, Y, Z]))
    rgb = np.clip(rgb, 0, 1)
    return rgb

def export_merl(bsdf, filename):
    """
    Export a simulated BSDF into a MERL-like binary file.
    
    MERL format (simplified):
      - Header: three int32 values: nThetaH, nThetaD, nPhiD.
      - Followed by a float32 array of shape (nThetaH x nThetaD x nPhiD x 3)
    
    Here we assume:
      nThetaH = 1 (single incident direction),
      nThetaD = number of outgoing angular samples (91),
      nPhiD = 1 (assumed isotropic).
    """
    theta_out = bsdf["theta_out_deg"]
    wavelengths = bsdf["wavelengths_nm"]
    brdf_spec = bsdf["BRDF"]
    n_theta = len(theta_out)
    
    rgb_values = np.zeros((n_theta, 3), dtype=np.float32)
    for i in range(n_theta):
        spectrum = brdf_spec[:, i]
        rgb = spectral_to_rgb(wavelengths, spectrum)
        rgb_values[i, :] = rgb

    nThetaH = 1
    nThetaD = n_theta
    nPhiD = 1
    merl_array = np.zeros((nThetaH, nThetaD, nPhiD, 3), dtype=np.float32)
    for i in range(nThetaH):
        for j in range(nThetaD):
            for k in range(nPhiD):
                merl_array[i, j, k, :] = rgb_values[j, :]

    with open(filename, "wb") as f:
        f.write(struct.pack('3i', nThetaH, nThetaD, nPhiD))
        f.write(merl_array.tobytes())
    print(f"MERL file saved to: {filename}")

def export_epfl(bsdf, filename):
    """
    Export the spectral BSDF data to an EPFL-like CSV file.
    Each row corresponds to an outgoing angular sample, with columns for theta_out and reflectance values.
    """
    theta_out = bsdf["theta_out_deg"]
    wavelengths = bsdf["wavelengths_nm"]
    brdf_spec = bsdf["BRDF"]

    header = "theta_out_deg," + ",".join([f"{int(w)}nm" for w in wavelengths])
    lines = [header]
    for i, theta in enumerate(theta_out):
        spec_values = brdf_spec[:, i]
        line = f"{theta:.1f}," + ",".join([f"{val:.6f}" for val in spec_values])
        lines.append(line)
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"EPFL (CSV) file saved to: {filename}")

# ============================
# 3. Main: Generate Smart Nano Material Configurations and Export
# ============================
def main():
    out_dir = "synthetic_dataset"
    os.makedirs(out_dir, exist_ok=True)
    
    # Define a smart selection of nano material configurations.
    sample_configs = [
        {
            "structure": "nanohole",
            "film_thickness": 20,
            "period": 150,
            "diameter": 70,
            "material": "Ag"
        },
        {
            "structure": "nanohole",
            "film_thickness": 20,
            "period": 170,
            "diameter": 85,
            "material": "Ag"
        },
        {
            "structure": "nanohole",
            "film_thickness": 20,
            "period": 200,
            "diameter": 100,
            "material": "Ag"
        }
        # Additional configurations (e.g., nanodisk, hybrid) can be added here.
    ]
    
    for idx, config in enumerate(sample_configs):
        print(f"\nSimulating sample {idx+1} with parameters: {json.dumps(config)}")
        bsdf = simulate_spectral_bsdf_meep(config)
        
        # Export to MERL-like binary format.
        merl_filename = os.path.join(out_dir, f"sample_{idx+1}.merl")
        export_merl(bsdf, merl_filename)
        
        # Export to EPFL-like CSV format.
        epfl_filename = os.path.join(out_dir, f"sample_{idx+1}_epfl.csv")
        export_epfl(bsdf, epfl_filename)
        
    print("\nDataset generation complete.")

if __name__ == '__main__':
    main()
