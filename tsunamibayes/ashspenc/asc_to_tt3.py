import numpy as np

def asc_to_tt3(asc_file, tt3_file):
    with open(asc_file, 'r') as f:
        # Read header
        header = {}
        for _ in range(6):
            key, value = f.readline().split()
            header[key.lower()] = float(value)

        ncols = int(header['ncols'])
        nrows = int(header['nrows'])
        nodata = header.get('nodata_value', -9999)

        # Load grid data
        data = np.loadtxt(f)
    
    # Flip vertically (ASC starts from upper-left; GeoClaw expects bottom-up)
    data = np.flipud(data)

    # Replace NODATA with 0 or another appropriate value
    data[data == nodata] = 0.0

    # Save to binary file in float32 format
    data.astype(np.float32).tofile(tt3_file)

    print(f"✅ Converted to {tt3_file} with shape ({nrows}, {ncols})")

    # Return info to use in setrun.py
    xlower = header['xllcorner']
    ylower = header['yllcorner']
    cellsize = header['cellsize']
    xupper = xlower + cellsize * ncols
    yupper = ylower + cellsize * nrows

    return {
        'nx': ncols,
        'ny': nrows,
        'xlower': xlower,
        'xupper': xupper,
        'ylower': ylower,
        'yupper': yupper
    }


if __name__ =='__main__':
    meta = asc_to_tt3(r'C:\Users\ashle\Documents\GitHub\tsunamibayes\tsunamibayes\ashspenc\GEBCO_28_Apr_2025_5086e97df180\gebco_2024_n10.0_s-2.0_w116.0_e129.0.asc', 'north_sulewesi_bathy.tt3')
