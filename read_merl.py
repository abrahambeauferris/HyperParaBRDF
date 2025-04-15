#!/usr/bin/env python3

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def debug_print(msg, debug=False):
    if debug:
        print(msg)

###############################################################################
# 1) READ MERL (with endianness detection)
###############################################################################
def try_read_merl_brdf(filename, endian, debug=False):
    file_size = os.path.getsize(filename)
    debug_print(f"  [try_read_merl_brdf] Attempting endianness='{endian}', size={file_size}", debug)

    with open(filename, 'rb') as f:
        dims_raw = np.fromfile(f, dtype=f"{endian}i4", count=3)
        if len(dims_raw)<3:
            debug_print("  [try_read_merl_brdf] Could not read 3 dims", debug)
            return None
        nThetaH, nThetaD, nPhiD = map(int, dims_raw)
        debug_print(f"  [try_read_merl_brdf] dims=({nThetaH},{nThetaD},{nPhiD})", debug)

        if not(1<=nThetaH<=2000 and 1<=nThetaD<=2000 and 1<=nPhiD<=4000):
            debug_print("  [try_read_merl_brdf] Dims out of range", debug)
            return None

        expected = 12+(nThetaH*nThetaD*nPhiD*3*8)
        if expected!=file_size:
            debug_print(f"  [try_read_merl_brdf] Mismatch size. expected={expected}, got={file_size}", debug)
            return None

        total_vals = nThetaH*nThetaD*nPhiD*3
        brdf_data = np.fromfile(f, dtype=f"{endian}f8", count=total_vals)
        if len(brdf_data)<total_vals:
            debug_print("  [try_read_merl_brdf] Not enough data read", debug)
            return None

        brdf_data = brdf_data.reshape((nThetaH,nThetaD,nPhiD,3))
    return (np.array([nThetaH,nThetaD,nPhiD], dtype=int), brdf_data)

def robust_read_merl_brdf(filename, debug=False):
    debug_print(f"[robust_read_merl_brdf] reading {filename}", debug)
    # 1) Big-endian
    res = try_read_merl_brdf(filename, '>', debug)
    if res: 
        dims, brdf=res
        debug_print(f"  => big-endian OK, dims={dims.tolist()}", debug)
        return dims, brdf
    # 2) Little-endian
    res = try_read_merl_brdf(filename, '<', debug)
    if res:
        dims, brdf=res
        debug_print(f"  => little-endian OK, dims={dims.tolist()}", debug)
        return dims, brdf
    raise ValueError("Could not read MERL data as big- or little-endian")

###############################################################################
# 2) LINEAR WARP for θh
###############################################################################
def linear_theta_h_deg(h_index, nThetaH):
    """
    Maps integer h in [0..nThetaH-1] => θh in [0..90].
    This ensures we start at 0° and end at 90°,
    avoiding the ~6° offset from bin-centering.
    """
    return (h_index/(nThetaH-1))*90.0

###############################################################################
# 3) CREATE SINGLE SLICE with BILINEAR INTERPOLATION
###############################################################################
def create_single_slice_linearwarp(
    brdf, out_png,
    phi=90,
    mask_quarter_circle=True,
    debug=False
):
    nThetaH, nThetaD, nPhiD, _=brdf.shape
    debug_print(f"[create_single_slice_linearwarp] shape={brdf.shape}, phi={phi}, mask={mask_quarter_circle}", debug)

    # 1) arrays of h, d => "true" angles
    #    We'll do a linear mapping for θh => 0..90, and assume θd is 1° per index
    merl_h = np.arange(nThetaH) # 0..89
    merl_d = np.arange(nThetaD) # 0..89
    real_h_deg = np.array([linear_theta_h_deg(h, nThetaH) for h in merl_h])  # shape (nThetaH,)
    real_d_deg = merl_d.astype(np.float32) # 0..(nThetaD-1)

    debug_print(f"  [warp] θh range=({real_h_deg[0]:.2f}..{real_h_deg[-1]:.2f}), θd range=({real_d_deg[0]}..{real_d_deg[-1]})", debug)

    # 2) Extract slice => shape (nThetaH,nThetaD,3)
    slice_img = brdf[:,:,phi,:]
    debug_print(f"  [create_single_slice_linearwarp] slice_img={slice_img.shape}", debug)

    # 3) Uniform output grid => 91 steps => 0..90
    out_size=91
    out_h = np.linspace(0,90,out_size) # horizontal => 0..90
    out_d = np.linspace(0,90,out_size) # vertical => 0..90
    # We'll store final in shape (out_size, out_size,3),
    # row=0 => top => d=90, row=out_size-1 => bottom => d=0
    out_img = np.zeros((out_size,out_size,3),dtype=np.float32)

    # bilinear interpolation
    for i in range(out_size):
        for j in range(out_size):
            # oh => θh
            oh = out_h[j]
            # od => θd => top= i=0 => 90, bottom= i=out_size-1 => 0
            od = out_d[out_size-1-i]

            # find nearest neighbors in real_h_deg
            idx_h2 = np.searchsorted(real_h_deg, oh)
            idx_h1 = max(idx_h2-1,0)
            idx_h2 = min(idx_h2,nThetaH-1)
            dh = real_h_deg[idx_h2]-real_h_deg[idx_h1]
            frac_h=0
            if abs(dh)>1e-12:
                frac_h=(oh-real_h_deg[idx_h1])/(dh)

            # find nearest neighbors in real_d_deg
            idx_d2=np.searchsorted(real_d_deg, od)
            idx_d1=max(idx_d2-1,0)
            idx_d2=min(idx_d2,nThetaD-1)
            dd= real_d_deg[idx_d2]-real_d_deg[idx_d1]
            frac_d=0
            if abs(dd)>1e-12:
                frac_d=(od-real_d_deg[idx_d1])/(dd)

            c11= slice_img[idx_h1, idx_d1,:]
            c12= slice_img[idx_h1, idx_d2,:]
            c21= slice_img[idx_h2, idx_d1,:]
            c22= slice_img[idx_h2, idx_d2,:]
            c1 = c11*(1-frac_d)+ c12*(frac_d)
            c2 = c21*(1-frac_d)+ c22*(frac_d)
            c  = c1*(1-frac_h)+ c2*(frac_h)

            out_img[i,j,:]=c

    # 4) optional mask
    masked=0
    if mask_quarter_circle:
        for i in range(out_size):
            for j in range(out_size):
                oh= out_h[j]
                od= out_d[out_size-1-i]
                if (oh+od)>90:
                    out_img[i,j,:]=0
                    masked+=1
    debug_print(f"  [create_single_slice_linearwarp] masked={masked} px", debug)

    # 5) normalize
    mn= out_img.min()
    mx= out_img.max()
    debug_print(f"  [create_single_slice_linearwarp] min_val={mn}, max_val={mx}", debug)
    if abs(mx-mn)<1e-12:
        out_img[:]=0
    else:
        out_img= (out_img-mn)/(mx-mn)
        out_img*=255.0
        out_img= out_img.clip(0,255).astype(np.uint8)

    # 6) Convert to PIL
    out_pil= Image.fromarray(out_img, mode='RGB')
    w,h= out_pil.size
    debug_print(f"  [create_single_slice_linearwarp] final size=({w},{h})", debug)

    # margin-based canvas
    margin=60
    cw= w+2*margin
    ch= h+2*margin
    canvas= Image.new("RGB",(cw,ch),(0,0,0))
    canvas.paste(out_pil,(margin,margin))
    draw= ImageDraw.Draw(canvas)
    font= ImageFont.load_default()

    def text_size(txt):
        bbox= draw.textbbox((0,0),txt,font=font)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])

    # labels
    left_label= "Specular Peak\n(θh=0°)"
    lw,lh= text_size(left_label)
    left_x= (margin-lw)//2
    left_y= margin+(h-lh)//2
    draw.text((left_x,left_y), left_label, fill=(255,255,255), font=font)

    right_label= "θh=90°"
    rw,rh= text_size(right_label)
    right_x= margin+w+(margin-rw)//2
    right_y= margin+(h-rh)//2
    draw.text((right_x,right_y), right_label, fill=(255,255,255), font=font)

    top_label= "Fresnel Peak\n(θd=90°)"
    tw,th= text_size(top_label)
    top_x= margin+(w-tw)//2
    top_y= (margin-th)//2
    draw.text((top_x,top_y), top_label, fill=(255,255,255), font=font)

    bottom_label= "Retroreflection\n(θd=0°)"
    bw,bh= text_size(bottom_label)
    bottom_x= margin+(w-bw)//2
    bottom_y= margin+h+(margin-bh)//2
    draw.text((bottom_x,bottom_y), bottom_label, fill=(255,255,255), font=font)

    phi_label= f"φd= {phi}°"
    draw.text((5,5), phi_label, fill=(255,255,0), font=font)

    canvas.save(out_png)
    debug_print(f"  [create_single_slice_linearwarp] saved => {out_png}", debug)

###############################################################################
# 4) MAIN
###############################################################################
def main():
    """
    Usage:
      python single_slice_linearwarp_debug.py <merl_file> <output_png> [phi=90] [--debug]

    Example:
      python single_slice_linearwarp_debug.py blue-fabric.binary out.png 90 --debug
    """
    debug_flag=False
    args=sys.argv[1:]
    if "--debug" in args:
        debug_flag=True
        args.remove("--debug")

    if len(args)<2:
        print("Usage: python single_slice_linearwarp_debug.py <merl_file> <output_png> [phi=90] [--debug]")
        sys.exit(1)

    merl_file= args[0]
    out_png= args[1]
    phi=90
    if len(args)>2:
        phi= int(args[2])

    # read merl
    try:
        dims, brdf= robust_read_merl_brdf(merl_file, debug=debug_flag)
    except ValueError as e:
        print(f"Error reading: {e}")
        sys.exit(1)

    # create slice
    create_single_slice_linearwarp(brdf, out_png, phi=phi, mask_quarter_circle=True, debug=debug_flag)

if __name__=="__main__":
    main()