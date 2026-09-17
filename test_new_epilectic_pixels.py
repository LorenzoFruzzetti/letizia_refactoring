from pathlib import Path

import numpy as np

from roi_editor import load_lambda_offset


filename_emo = Path("pixel_data\\260616_R4\\t1\\pixels_f_emo_full.npy")
filename_gcamp = Path("pixel_data\\260616_R4\\t1\\pixels_f_gcamp_full.npy")
filename_meta = Path("pixel_data\\260616_R4\\t1\\pixels_meta_full.npz")

meta = np.load(filename_meta, allow_pickle=True)

# The crop window is Bregma-relative, so Bregma is at a fixed index in every dump.
bregma_rc = (-int(meta["window_rel"][0]), -int(meta["window_rel"][2]))
lambda_rows = load_lambda_offset(str(meta["roi_set"]))
lambda_rc = (bregma_rc[0] + lambda_rows, bregma_rc[1])


def map_pixel_to_bregma_lambda(pixel):
    """(row, col) -> offset from Bregma in grid pixels, and in Bregma-Lambda units.

    +AP is posterior, +ML is right. The normalised pair (0 at Bregma, 1 at Lambda)
    is the one that is comparable across animals.
    """
    row, col = pixel
    ap = row - bregma_rc[0]
    ml = col - bregma_rc[1]
    return (ap, ml), (ap / lambda_rows, ml / lambda_rows)




print("bregma", bregma_rc, "lambda", lambda_rc, "offset", lambda_rows)
print(map_pixel_to_bregma_lambda(bregma_rc))
print(map_pixel_to_bregma_lambda(lambda_rc))
print(map_pixel_to_bregma_lambda((10, 70)))
