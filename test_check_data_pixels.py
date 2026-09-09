from pathlib import Path
import numpy as np


filename = Path("pixel_data") / "260716_PV3" / "t2" / "pixels_f_gcamp_full.npy"
data = np.load(filename)

filename_meta = Path("pixel_data") / "260716_PV3" / "t2" / "pixels_meta_full.npz"
meta = np.load(filename_meta)
a = 1