import os
from pathlib import Path
import mrcfile
from glob import glob
from registration_functions import (
    run_fbm, run_fbm_laguerre,
    run_fast_fbm_laguerre, precompute_w_params
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

image_radius = 100
pixel_sampling = 1.
com_offset_initial = 8
lag_func_num=50
lag_scale=5
lag_num_dots = 1000
center = 128, 128

params = precompute_w_params(image_radius, pixel_sampling, com_offset_initial, lag_func_num,
                             lag_scale, compute_zeros=True)

for method, func in [('fbm', run_fbm), ('fbm_laguerre', run_fbm_laguerre),
                     ('fast_fbm_laguerre', run_fast_fbm_laguerre)]:

    method_path = results_path / method
    os.makedirs(str(method_path), exist_ok=True)

    opath = method_path / folder / model
    os.makedirs(opath, exist_ok=True)
    opath = method_path / folder / model / str(path).split('/')[-1]
    if not os.path.exists(opath):
        with mrcfile.open(path) as mrc:
            seq = mrc.data
        df, fbm_seq, fbm_seq_shift = func(seq, params)
        save_arr_mrc(opath, fbm_seq)
        df.to_csv(str(opath).replace('stack.mrc', 'info.csv'))

    for snr_value in [0.1, 0.5, 1., 2.]:
        data_name = str(Path(global_path) / folder / model / f'{snr_value}/{num}{name}')

        opath = method_path / folder / model / str(snr_value)
        os.makedirs(str(opath), exist_ok=True)
        opath = method_path / folder / model / str(snr_value) / data_name.split('/')[-1]

        if not os.path.exists(opath):
            with mrcfile.open(data_name) as mrc:
                seq = mrc.data
            df, fbm_seq, fbm_seq_shift = func(seq, params)

            save_arr_mrc(opath, fbm_seq)
            df.to_csv(str(opath).replace('stack.mrc', 'info.csv'))
