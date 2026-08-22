from dataset.utils import filter_registers_by_key_value_sequence, get_acquisition_data, get_values_by_key, load_matlab_acquisition, prepare_segments_and_targets
import numpy as np
import librosa


MAP_LOAD_TO_RPM = {
    '0': 1797,
    '1': 1772,
    '2': 1750,
    '3': 1730,
}

def get_augmented_data(list_of_registers, experimental_setup, repetitions=1):
    X, y = [], []
    for _ in range(repetitions):
        X_aug, y_aug = augment_acquisition(list_of_registers, experimental_setup)
        X.append(X_aug)
        y.append(y_aug)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def augment_acquisition(list_of_registers, experimental_setup):
    conditions = get_values_by_key(list_of_registers, "condition")
    X, y, = [], []
    for condition in conditions:
        # X_agregated_sev, y_agregated_sev = aggregate_severity_acquisitions(list_of_registers, condition, experimental_setup)
        # if X_agregated_sev is None or y_agregated_sev is None:
        #     continue
        X_agregated_load, y_agregated_load = aggregate_load_acquisitions(list_of_registers, condition, experimental_setup)
        if X_agregated_load is None or y_agregated_load is None:
            continue
        # X_agregated = np.concatenate([X_agregated_sev, X_agregated_load], axis=0)
        # y_agregated = np.concatenate([y_agregated_sev, y_agregated_load], axis=0)
        X.append(X_agregated_load)
        y.append(y_agregated_load)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


# BY THE LOAD
def aggregate_load_acquisitions(
    list_of_registers,
    condition,
    experimental_setup
):
    X, y = [], []
    loads = (list(get_values_by_key(list_of_registers, "load")))
    for load in loads:
        condition_registers = filter_registers_by_key_value_sequence(list_of_registers, [("condition", [condition]), ("load", [load])])
        if len(condition_registers) <= 1:
            continue
        X_mixed, y_mixed = mix_severity_data(condition_registers, experimental_setup)
        X.append(X_mixed)
        y.append(y_mixed)
    if len(X) == 0 or len(y) == 0:
        return None, None
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X,y


def aggregate_severity_acquisitions(list_of_registers, condition, experimental_setup):
    X, y = [], []
    severities = (list(get_values_by_key(list_of_registers, "severity")))
    for severity in severities:
        if severity == '0':
            continue
        condition_registers = filter_registers_by_key_value_sequence(
            list_of_registers, 
            [("condition", [condition]), ("severity", [severity])]
        )
        if len(condition_registers) <= 1:
            continue
        X_mixed, y_mixed = mix_severity_data(condition_registers, experimental_setup)
        X.append(X_mixed)
        y.append(y_mixed)
    if len(X) == 0 or len(y) == 0:
        return None, None
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X,y


def mix_severity_data(condition_registers, experimental_setup):
    segment_length=experimental_setup["segment_length"]
    channels_columns=experimental_setup["channels_columns"]
    acquisitions = []
    for key_value in channels_columns.keys():
        key, value = key_value.split(":")
        value = eval(value)
        filtered_registers = filter_registers_by_key_value_sequence(condition_registers, [[key, value]])
        actual_channel_columns = channels_columns[key_value]
        for condition_register in filtered_registers:
            acquisition = load_original_acquisitions(condition_register, actual_channel_columns, experimental_setup)
            acquisitions.append(acquisition)
    X, y = mix_acquisitions(condition_registers, segment_length, acquisitions)
    return X, y


def align_acquisitions(acq_1, acq_2, load_1, load_2,fs=48_000):
    rpm_i = MAP_LOAD_TO_RPM.get(load_1)
    rpm_j = MAP_LOAD_TO_RPM.get(load_2)
    rpm_ref = (rpm_i + rpm_j) / 2
    acq_1_aligned = librosa.resample(
        np.asarray(acq_1).ravel(),
        orig_sr=fs,
        target_sr=fs * (rpm_ref / rpm_j)
    )
    acq_2_aligned = librosa.resample(
        np.asarray(acq_2).ravel(),
        orig_sr=fs,
        target_sr=fs * (rpm_ref / rpm_i)
    )
    min_len = min(len(acq_1_aligned), len(acq_2_aligned))
    return acq_1_aligned[:min_len].reshape(-1, 1), acq_2_aligned[:min_len].reshape(-1, 1)

def mix_acquisitions(condition_registers, segment_length, acquisitions):
    X, y = [], []
    for i in range(len(acquisitions)-1):
        for j in range(i+1, len(acquisitions)):
            # verify if the acquisitions are aligned
            load_i = condition_registers[i]["load"]
            load_j = condition_registers[j]["load"]
            if load_i == load_j:
                acq_i = acquisitions[i]
                acq_j = acquisitions[j]
            else:
                acq_i, acq_j = align_acquisitions(acquisitions[i], acquisitions[j], load_i, load_j)
            acq_i = acquisitions[i]
            acq_j = acquisitions[j]            
            mixed_acquisition = mix_two_acquisitions(acq_i, acq_j)
            X_mix, y_mix = prepare_segments_and_targets(segment_length=segment_length, register=condition_registers[i], acquisition=mixed_acquisition)
            X.append(X_mix)
            y.append(y_mix)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X,y


def load_original_acquisitions(condition_register, actual_channel_columns, experimental_setup):
    raw_dir_path=experimental_setup["raw_dir_path"]
    load_acquisition_func=eval(experimental_setup["load_acquisition_func"])
    acquisition = get_acquisition_data(raw_dir_path, actual_channel_columns, load_acquisition_func, condition_register)
    return acquisition


def mix_two_acquisitions(acq1, acq2):
    min_length = min(acq1.shape[0], acq2.shape[0])
    acq1, acq2 = acq1[:min_length], acq2[:min_length]
    xf1, xf2 = np.fft.rfft(acq1, axis=0), np.fft.rfft(acq2, axis=0)
    # xf_mix = (xf1 + xf2)
    mag = np.abs(xf1) + np.abs(xf2)
    phase = np.angle(xf1)
    xf_mix = mag * np.exp(1j * phase)
    return np.fft.irfft(xf_mix, n=max(acq1.shape[0], acq2.shape[0]), axis=0)

