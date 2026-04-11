import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sqlalchemy.util import defaultdict
from experiment.assesment import f1_macro, print_dict_of_scores, print_scores_list

from experiment.assesment import holdout
from dataset.utils import (
    read_registers_from_config,
    filter_registers_by_key_value_sequence,
    load_matlab_acquisition,
    extract_segments_and_targets,   # used for ORIGINAL (non-augmented) segments
    get_acquisition_data,          # loads full acquisition (with download fallback)
    split_acquisition,             # segments an acquisition into fixed-length windows
    target_array,                  # creates y array for a given label/length
)

segment_length = 1024

#            ### training -------------------------###   ### testing --###
papers_split = [(['0', '1', '2', '3'], ['0.007', '0.014']), (['0'], ['0.021'])]
# papers_split = [(['0', '1', '2', '3'], ['0.007', '0.014']), (['0', '1', '2', '3'], ['0.021'])]
# papers_split = [(['0', '1', '2', '3'], ['0.007', '0.021']), (['0', '1', '2', '3'], ['0.014'])]
# papers_split = [(['0', '1', '2', '3'], ['0.014', '0.021']), (['0', '1', '2', '3'], ['0.007'])]

# ---------------------------------------------------------------------
# Augmentation at ACQUISITION level (NOT segment level)
# Adapted rule (your request):
#   same load + same condition (class), but different severities.
#
# EXACT Mode 3 (paper):
#   - FFT EACH single sample/acquisition individually
#   - Random superposition in frequency domain
#   - IFFT back to time domain
#   - THEN segment
# ---------------------------------------------------------------------

def _rfft_acquisition(acq: np.ndarray) -> np.ndarray:
    """
    acq: (T, C) time-domain acquisition
    returns: (F, C) complex spectrum via rFFT along time axis
    """
    return np.fft.rfft(acq, axis=0)

def _irfft_acquisition(Xf: np.ndarray, n_time: int) -> np.ndarray:
    """
    Xf: (F, C) complex spectrum
    returns: (T, C) real acquisition via irFFT with length n_time
    """
    x = np.fft.irfft(Xf, n=n_time, axis=0)
    return x.astype(np.float32)

def _align_acquisitions(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensures both acquisitions have same time length by trimming to the minimum length.
    This is required to superpose spectra bin-to-bin.
    """
    T = min(a.shape[0], b.shape[0])
    return a[:T], b[:T], T


def _mix_two_acquisitions(acq_a: np.ndarray, acq_b: np.ndarray) -> np.ndarray:
    acq_a, acq_b, T = _align_acquisitions(acq_a, acq_b)
    Xf_a = _rfft_acquisition(acq_a)
    Xf_b = _rfft_acquisition(acq_b)
    Xf_mix = (Xf_a + Xf_b)
    acq_mix = _irfft_acquisition(Xf_mix, n_time=T)
    return acq_mix


# Same load + same condition (class), but different severities.
def _build_augmented_segments_from_registers(
    registers: list[dict],
    raw_dir_path: str,
    channels_columns: list[str],
    segment_length: int,
    load_acquisition_func,
    rng: np.random.Generator,
    mixes_per_pair: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
   
    # Cache full acquisitions:
    # groups[(load, condition)][severity] = [acq1, acq2, ...]
    groups: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}

    for reg in registers:
        ld = str(reg.get("load"))
        cond = str(reg.get("condition"))
        sev = str(reg.get("severity"))

        acq = get_acquisition_data(raw_dir_path, channels_columns, load_acquisition_func, reg)
        acq = acq.astype(np.float32)

        key = (ld, cond)
        groups.setdefault(key, {}).setdefault(sev, []).append(acq)
    
    X_aug_list: list[np.ndarray] = []
    y_aug_list: list[np.ndarray] = []

    for (ld, cond), sev_dict in groups.items():
        sevs = sorted(sev_dict.keys())
        if len(sevs) < 2:
            continue

        # Unique severity pairs
        for i in range(len(sevs)):
            for j in range(i + 1, len(sevs)):
                sev_a, sev_b = sevs[i], sevs[j]
                list_a = sev_dict.get(sev_a, [])
                list_b = sev_dict.get(sev_b, [])
                if not list_a or not list_b:
                    continue

                # Create mixed samples by random pairing of single acquisitions
                # A "natural" count is min(len_a, len_b); mixes_per_pair repeats that process.
                n_base = min(len(list_a), len(list_b))
                n_new = mixes_per_pair * n_base
                if n_new <= 0:
                    continue

                # Random pairing indices (with replacement if mixes_per_pair > 1)
                idx_a = rng.integers(0, len(list_a), size=n_new)
                idx_b = rng.integers(0, len(list_b), size=n_new)

                for ka, kb in zip(idx_a, idx_b):
                    acq_mix = _mix_two_acquisitions(list_a[ka], list_b[kb])

                    # Segment AFTER mixing (required by your request and matches the pipeline idea)
                    segs = split_acquisition(acq_mix, segment_length=segment_length)
                    if segs.shape[0] == 0:
                        continue

                    X_aug_list.append(segs)
                    y_aug_list.append(target_array(cond, segs.shape[0]))

    if not X_aug_list:
        return (
            np.empty((0, segment_length, 1), dtype=np.float32),
            np.empty((0,), dtype='U10')
        )

    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    return X_aug, y_aug


def get_X_y_augmented_acquisition_level(
    registers,
    raw_dir_path,
    channels_columns,
    segment_length,
    load_acquisition_func,
    rng: np.random.Generator,
    mixes_per_pair: int = 1,
    augment: bool = True,   # <--- NEW
):
    # ORIGINAL (always)
    X_list = []
    y_list = []

    if len(registers) > 0:
        for reg in registers:
            segs, targets = extract_segments_and_targets(
                raw_dir_path, channels_columns, segment_length, load_acquisition_func, reg
            )
            X_list.append(segs.astype(np.float32))
            y_list.append(targets)

    if not X_list:
        X_orig = np.empty((0, segment_length, 1), dtype=np.float32)
        y_orig = np.empty((0,), dtype="U10")
    else:
        X_orig = np.concatenate(X_list, axis=0)
        y_orig = np.concatenate(y_list, axis=0)

    # --- NO augmentation ---
    if not augment:
        return X_orig, y_orig

    # AUGMENTED (only if augment=True)
    X_aug, y_aug = _build_augmented_segments_from_registers(
        registers=registers,
        raw_dir_path=raw_dir_path,
        channels_columns=channels_columns,
        segment_length=segment_length,
        load_acquisition_func=load_acquisition_func,
        rng=rng,
        mixes_per_pair=mixes_per_pair,
    )

    # X_aug_ss, y_aug_ss = _build_augmented_segments_from_registers_same_severity(
    #     registers=registers,
    #     raw_dir_path=raw_dir_path,
    #     channels_columns=channels_columns,
    #     segment_length=segment_length,
    #     load_acquisition_func=load_acquisition_func,
    #     rng=rng,
    #     mixes_per_pair=mixes_per_pair,
    # )

    X_out = np.concatenate([X_orig, X_aug], axis=0)
    y_out = np.concatenate([y_orig, y_aug], axis=0)
    return X_out, y_out


def get_list_of_X_y_mode3_by_load(
    list_of_folds,
    raw_dir_path,
    channels_columns,
    segment_length,
    load_acquisition_func,
    seed: int = 0,
    mixes_per_pair: int = 1,
    test_fold_index: int | None = None,
    augment=False,
):
    """
    Returns list_of_X_y where:
      - training folds: augmented (Mode 3, acquisition-level)
      - test fold: NOT augmented
    """
    rng = np.random.default_rng(seed)
    list_of_X_y = []

    for fold_idx, fold in enumerate(list_of_folds):
        
        # debug
        counts = defaultdict(int)
        n_original = 0
        # Count original samples by class and severity
        for reg in fold:
            segs, _ = extract_segments_and_targets(
                raw_dir_path, channels_columns, segment_length, load_acquisition_func, reg
            )
            n_segs = len(segs)
            counts[(reg["condition"], reg["severity"])] += n_segs
            n_original += n_segs

        # Test fold untouched
        if test_fold_index is not None and fold_idx == test_fold_index:
            X_list = []
            y_list = []
            if len(fold) == 0:
                X = np.empty((0, segment_length, 1), dtype=np.float32)
                y = np.empty((0,), dtype='U10')
            else:
                for reg in fold:
                    segs, targets = extract_segments_and_targets(
                        raw_dir_path, channels_columns, segment_length, load_acquisition_func, reg
                    )
                    X_list.append(segs.astype(np.float32))
                    y_list.append(targets)

                X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, segment_length, 1), dtype=np.float32)
                y = np.concatenate(y_list, axis=0) if y_list else np.empty((0,), dtype='U10')
            
            # debug print
            print(f"\nFold {fold_idx} [TEST]")
            for (condition, severity), n in sorted(counts.items()):
                print(f"{condition} | severity {severity} -> {n} original samples")
            print(f"Augmented samples -> 0")
            
            list_of_X_y.append((X, y))
            continue

        # Training folds: augmented at acquisition-level
        X, y = get_X_y_augmented_acquisition_level(
            registers=fold,
            raw_dir_path=raw_dir_path,
            channels_columns=channels_columns,
            segment_length=segment_length,
            load_acquisition_func=load_acquisition_func,
            rng=rng,
            mixes_per_pair=mixes_per_pair,
            augment=augment
        )

        # debug print
        n_total = len(y)
        n_augmented = n_total - n_original
        print(f"\nFold {fold_idx} [TRAIN]")
        for (condition, severity), n in sorted(counts.items()):
            print(f"{condition} | severity {severity} -> {n} original samples")
        print(f"Augmented samples -> {n_augmented}")


        list_of_X_y.append((X, y))

    return list_of_X_y


# ---------------------------------------------------------------------
# Fold construction (same structure as sehri_et_al.py)
# ---------------------------------------------------------------------

def get_papers_split(loads, severities):
    sample_rate = "48000"
    config_file = "dataset/cwru/config.csv"
    prlzs = ['None', '6']

    registers = read_registers_from_config(config_file)
    filtered = filter_registers_by_key_value_sequence(
        registers,
        [
            ('sample_rate', [sample_rate]),
            ('load', loads),
            ('severity', severities),
            ('prlz', prlzs),
        ]
    )
    return filtered


def get_list_of_papers_splits():
    folds = []
    for loads, severities in papers_split:
        folds.append(get_papers_split(loads, severities))
    return folds

def print_samples_by_class_and_severity(list_of_folds):
    counts = {}
    for fold in list_of_folds:
        for reg in fold:
            condition = reg["condition"]
            severity = reg["severity"]
            key = (condition, severity)
            counts[key] = counts.get(key, 0) + 1
    for (condition, severity), n in sorted(counts.items()):
        print(f"{condition} | severity {severity} -> {n} samples")

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_papers_experiment_augmented(model, mixes_per_pair: int = 1, seed: int = 0, augment: bool = False):
    """
    Hold-out experiment with Mode 3 augmentation applied ONLY on training fold(s).
    Test fold is kept clean.
    Augmentation is performed at ACQUISITION level (FFT mix -> IFFT -> segment).
    """
    test_fold_index = 1
    list_of_folds = get_list_of_papers_splits()
    print_samples_by_class_and_severity(list_of_folds)
    list_of_X_y = get_list_of_X_y_mode3_by_load(
        list_of_folds,
        raw_dir_path="raw_data/cwru",
        channels_columns=['DE'],
        segment_length=segment_length,
        load_acquisition_func=load_matlab_acquisition,
        mixes_per_pair=mixes_per_pair,
        seed=seed,
        test_fold_index=test_fold_index,
        augment=augment
    )
    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    scores = holdout(model, list_of_X_y, test_fold_index=test_fold_index, list_of_metrics=list_of_metrics)
    print_dict_of_scores(scores)
