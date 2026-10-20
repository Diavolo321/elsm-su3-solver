"""Semi-automatic pixel extraction helper for Fig. 1 (SU(3), page 13).

This script deliberately does not claim a unique trace: several red solid curves
and the rising Polyakov-loop curve overlap. It records candidate red pixels and
calibrated coordinates so branch selection can be inspected rather than hidden.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE = Path('/mnt/data/user-cufeQyXpLl2fi9ofbeBJcrqZ/0d2fa795adee4f7c8199da1378e1363f/tmp/paperpage-13.png')
if not IMAGE.exists():
    raise FileNotFoundError(f'Place the source raster at {IMAGE} or edit IMAGE.')
im = np.asarray(Image.open(IMAGE).convert('RGB'))
# Axis calibration read from page-13 raster; uncertainty is several pixels due to
# antialiasing, line thickness, overlapping branches, and frame thickness.
x0, x1, y_top, y_bottom = 158, 475, 192, 420
rows=[]
for x in range(x0+1, x1):
    region=im[y_top:y_bottom+1, x, :]
    red=(region[:,0] > 145) & (region[:,1] < 105) & (region[:,2] < 105)
    for yrel in np.where(red)[0]:
        y=int(yrel+y_top)
        rows.append({'T_GeV':(x-x0)/(x1-x0)*0.5,
                     'condensate_normalized_candidate':1-(y-y_top)/(y_bottom-y_top),
                     'pixel_x':x,'pixel_y':y})
out=ROOT/'results'/'fig1_red_pixel_candidates.csv'
pd.DataFrame(rows).to_csv(out,index=False)
print(f'Wrote {len(rows)} red-pixel candidates to {out}')
print('Warning: candidates include multiple red branches; do not compute RMSE until the target branch is manually disambiguated.')
