from dataset.utils import get_X_y, load_matlab_acquisition

def get_sehri_et_all_X_y(registers):
    raw_dir_path="raw_data/cwru"
    channels_columns=['DE']
    segment_length=1024
    load_acquisition_func=load_matlab_acquisition
    X, y = get_X_y(registers, raw_dir_path=raw_dir_path, channels_columns=channels_columns, segment_length=segment_length, load_acquisition_func=load_acquisition_func)
    return X, y

