from dataset.utils import get_X_y, load_matlab_acquisition

def get_sehri_et_al_X_y(registers):
    raw_dir_path="raw_data/cwru"
    channels_columns=['DE']
    segment_length=1024
    load_acquisition_func=load_matlab_acquisition
    X, y = get_X_y(registers, raw_dir_path=raw_dir_path, channels_columns=channels_columns, segment_length=segment_length, load_acquisition_func=load_acquisition_func)
    return X, y

# def single_channel_X_y_DE_FE_12k(combination, segment_length):
#     sample_rate = '12000'
#     X_y_DE = single_channel_X_y(combination, segment_length, sample_rate, 'Drive End', 'DE')
#     X_y_FE = single_channel_X_y(combination, segment_length, sample_rate, 'Fan End', 'FE')
#     list_of_X_y = merge_X_y_from_lists(X_y_DE, X_y_FE)
#     return list_of_X_y

def get_rauber_loca_et_al_X_y(registers):
    raw_dir_path="raw_data/cwru"
    channels_columns=['DE']
    segment_length=2048
    load_acquisition_func=load_matlab_acquisition
    X, y = get_X_y(registers, raw_dir_path=raw_dir_path, channels_columns=channels_columns, segment_length=segment_length, load_acquisition_func=load_acquisition_func)
    return X, y