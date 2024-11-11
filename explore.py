import logging
from astropy.io import fits
import numpy as np

filename = 'data/Lightcurves/gen_20220830.lc'
with fits.open(filename) as hdul:
    table: fits.hdu.table.BinTableHDU = hdul.pop(1)
    # Rename column from RATE to FLUX
    print(table.columns)

    table.columns['RATE'].name = 'FLUX'
    data = np.array(table.data)
    print(table.columns)
    print(data['RATE'])
    # Create a new HDU list with the modified table
    new_hdul = fits.HDUList([fits.PrimaryHDU(), table])
    # Save the modified file
    new_hdul.writeto(filename, overwrite=True)
