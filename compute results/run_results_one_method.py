import os
from pathlib import Path
import mrcfile
from glob import glob
from registration_functions import (
    run_fbm, run_fbm_laguerre,
    run_fast_fbm_laguerre, precompute
)
from util_functions import save_arr_mrc

global_path = '/home/ubuntu/Data/cryo-em samples/synthetic_data_2023'
folder = 'small_trans'
model = 'model2096'
name = '_stack.mrc'
params_name = '_info.csv'
num = 2

path = Path(global_path) / folder / model / f'{num}{name}'

results_path = Path(global_path) / 'results'
os.makedirs(str(results_path), exist_ok=True)

method = 'fast_fbm_laguerre'
method_path = results_path / method
os.makedirs(str(method_path), exist_ok=True)
functions = {'fbm': run_fbm, 'fbm_laguerre': run_fbm_laguerre, 'fast_fbm_laguerre': run_fast_fbm_laguerre}
func = functions[method]

# image_radius = 100
# pixel_sampling = 1.
# com_offset_initial = 8
# lag_func_num=50
# lag_scale=5
# lag_num_dots = 1000
# center = 128, 128

image_radius = 120
pixel_sampling = 0.5
com_offset_initial = 10
lag_func_num=100
lag_scale=5
lag_num_dots = 2000
center = 128, 128

params = precompute()

method_path = results_path / method
os.makedirs(str(method_path), exist_ok=True)

opath = method_path / folder / model
os.makedirs(opath, exist_ok=True)
opath = method_path / folder / model / str(path).split('/')[-1]

with mrcfile.open(path) as mrc:
    seq = mrc.data
df, fbm_seq, fbm_seq_shift = func(seq, params)
save_arr_mrc(opath, fbm_seq)
df.to_csv(str(opath).replace('stack.mrc', 'info.csv'))

for snr_value in [0.1, 0.5, 1., 2.]:
    for data_name in glob(str(Path(global_path) / folder / model /
                              f'{snr_value}/{num}{name}')):
        with mrcfile.open(data_name) as mrc:
            seq = mrc.data
        df, fbm_seq, fbm_seq_shift = func(seq, params)
        opath = method_path / folder / model / str(snr_value)
        os.makedirs(str(opath), exist_ok=True)
        opath = method_path / folder / model / str(snr_value) / data_name.split('/')[-1]
        save_arr_mrc(opath, fbm_seq)
        df.to_csv(str(opath).replace('stack.mrc', 'info.csv'))
