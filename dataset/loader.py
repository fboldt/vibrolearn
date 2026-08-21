# from dataset.utils import get_X_y, get_X_y_domains, load_matlab_acquisition
# from preprocessing.augmentation import get_agumented_data
# import numpy as np

from dataset.utils import (
    get_X_y,
    get_X_y_domains,
    load_matlab_acquisition,
    filter_registers_by_key_value_sequence,
    get_acquisition_data,
    prepare_segments_and_targets
)
import numpy as np


def vanilla(registers, experimental_setup, get_domains=True):
    raw_dir_path=experimental_setup["raw_dir_path"]
    channels_columns=experimental_setup["channels_columns"]
    segment_length=experimental_setup["segment_length"]
    load_acquisition_func=eval(experimental_setup["load_acquisition_func"])
    if get_domains:
        return get_X_y_domains(registers, 
                   raw_dir_path=raw_dir_path, 
                   channels_columns=channels_columns, 
                   segment_length=segment_length, 
                   load_acquisition_func=load_acquisition_func,
                   domain_key="severity")
    else:        
        return get_X_y(registers, 
                   raw_dir_path=raw_dir_path, 
                   channels_columns=channels_columns, 
                   segment_length=segment_length, 
                   load_acquisition_func=load_acquisition_func)
    




def mix_two_acquisitions(acq1, acq2):
    min_length = min(acq1.shape[0], acq2.shape[0])

    acq1 = acq1[:min_length]
    acq2 = acq2[:min_length]

    xf1 = np.fft.rfft(acq1, axis=0)
    xf2 = np.fft.rfft(acq2, axis=0)

    mag = np.abs(xf1) + np.abs(xf2)
    phase = np.angle(xf1)

    xf_mix = mag * np.exp(1j * phase)

    return np.fft.irfft(
        xf_mix,
        n=min_length,
        axis=0
    )


def augment_normal(registers, experimental_setup):
    raw_dir_path = experimental_setup["raw_dir_path"]
    channels_columns = experimental_setup["channels_columns"]
    segment_length = experimental_setup["segment_length"]
    load_acquisition_func = eval(
        experimental_setup["load_acquisition_func"]
    )

    # Seleciona apenas a classe Normal
    normal_registers = [
        register
        for register in registers
        if register["condition"] == "Normal"
    ]

    X_aug = []
    y_aug = []
    domains_aug = []

    for key_value, actual_channel_columns in channels_columns.items():

        key, value = key_value.split(":")
        value = eval(value)

        channel_registers = filter_registers_by_key_value_sequence(
            normal_registers,
            [[key, value]]
        )

        # Combina todos os pares de registros normais
        for i in range(len(channel_registers) - 1):
            for j in range(i + 1, len(channel_registers)):

                register_i = channel_registers[i]
                register_j = channel_registers[j]

                # A mistura ocorre somente entre cargas diferentes
                if register_i["load"] == register_j["load"]:
                    continue

                acq_i = get_acquisition_data(
                    raw_dir_path,
                    actual_channel_columns,
                    load_acquisition_func,
                    register_i
                )

                acq_j = get_acquisition_data(
                    raw_dir_path,
                    actual_channel_columns,
                    load_acquisition_func,
                    register_j
                )

                mixed_acquisition = mix_two_acquisitions(
                    acq_i,
                    acq_j
                )

                X_mix, y_mix = prepare_segments_and_targets(
                    segment_length=segment_length,
                    register=register_i,
                    acquisition=mixed_acquisition
                )

                X_aug.append(X_mix)
                y_aug.append(y_mix)

                # Como o domínio utilizado é severity
                domains_aug.append(
                    np.full(
                        len(y_mix),
                        register_i["severity"]
                    )
                )

    if len(X_aug) == 0:
        return None, None, None

    return (
        np.concatenate(X_aug, axis=0),
        np.concatenate(y_aug, axis=0),
        np.concatenate(domains_aug, axis=0)
    )


def augmented(registers, experimental_setup, get_domains=True):

    raw_dir_path = experimental_setup["raw_dir_path"]
    channels_columns = experimental_setup["channels_columns"]
    segment_length = experimental_setup["segment_length"]
    load_acquisition_func = eval(
        experimental_setup["load_acquisition_func"]
    )

    # Dados originais
    if get_domains:

        X, y, domains = get_X_y_domains(
            registers,
            raw_dir_path=raw_dir_path,
            channels_columns=channels_columns,
            segment_length=segment_length,
            load_acquisition_func=load_acquisition_func,
            domain_key="severity"
        )

        # Aumento apenas da classe Normal
        X_aug, y_aug, domains_aug = augment_normal(
            registers,
            experimental_setup
        )

        if X_aug is not None:
            X = np.concatenate([X, X_aug], axis=0)
            y = np.concatenate([y, y_aug], axis=0)
            domains = np.concatenate(
                [domains, domains_aug],
                axis=0
            )

        return X, y, domains

    else:

        X, y = get_X_y(
            registers,
            raw_dir_path=raw_dir_path,
            channels_columns=channels_columns,
            segment_length=segment_length,
            load_acquisition_func=load_acquisition_func
        )

        X_aug, y_aug, _ = augment_normal(
            registers,
            experimental_setup
        )

        if X_aug is not None:
            X = np.concatenate([X, X_aug], axis=0)
            y = np.concatenate([y, y_aug], axis=0)

        return X, y





# SPECTROGRAMAS

from pathlib import Path
import numpy as np


def spectrogram_vanilla(
    registers,
    experimental_setup
):

    spectrogram_dir = Path(
        experimental_setup.get(
            "spectrogram_dir",
            "spectrograms/cwru"
        )
    )

    X_list = []
    y_list = []

    missing = []


    for register in registers:

        # ----------------------------------------------------
        # Exemplo:
        #
        # acquisition_file = "97.mat"
        #
        # diretório:
        #
        # spectrograms/cwru/97/
        # ----------------------------------------------------

        acquisition_file = register[
            "acquisition_file"
        ]

        acquisition_id = Path(
            acquisition_file
        ).stem

        acquisition_dir = (
            spectrogram_dir
            / acquisition_id
        )


        if not acquisition_dir.exists():

            missing.append(
                str(acquisition_dir)
            )

            continue


        image_paths = sorted(
            acquisition_dir.glob("*.png")
        )


        if len(image_paths) == 0:

            missing.append(
                str(acquisition_dir)
            )

            continue


        # ----------------------------------------------------
        # Classe
        #
        # Mesmo campo utilizado pelo seu loader convencional.
        #
        # Ex:
        # Normal
        # Inner Race
        # Outer Race
        # Ball
        # ----------------------------------------------------

        target = register["condition"]


        X_list.extend(
            [
                str(image_path)
                for image_path
                in image_paths
            ]
        )

        y_list.extend(
            [target]
            * len(image_paths)
        )


    if missing:

        raise FileNotFoundError(
            "\nNão foram encontrados espectrogramas "
            "para os seguintes diretórios:\n"
            +
            "\n".join(missing)
        )


    return (
        np.asarray(
            X_list,
            dtype=object
        ),
        np.asarray(y_list)
    )