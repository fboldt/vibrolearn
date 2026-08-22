from ast import literal_eval
from pathlib import Path

import numpy as np

from matplotlib import colormaps
from PIL import Image
from scipy.signal import detrend, stft

from dataset.utils import (
    get_acquisition_data,
    load_matlab_acquisition
)

from preprocessing.data_adapter import (
    DataAdapter,
    PreparedData
)


class SpectrogramAdapter(DataAdapter):

    def __init__(
        self,
        segment_length=12000,
        normalization="rms",
        nperseg=1024,
        noverlap=896,
        nfft=2048,
        db_min=-100.0,
        db_max=0.0,
        image_size=(224, 224),
        cache_dir="cache/spectrograms"
    ):
        self.segment_length = segment_length

        self.normalization = normalization

        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft

        self.db_min = db_min
        self.db_max = db_max

        self.image_size = image_size

        self.cache_dir = Path(
            cache_dir
        )

    # ========================================================
    # INTERFACE
    # ========================================================

    def prepare(
        self,
        registers,
        experimental_setup,
        training=False
    ):
        X = []
        y = []

        for register in registers:

            image_paths = (
                self._prepare_acquisition(
                    register,
                    experimental_setup
                )
            )

            X.extend(image_paths)

            y.extend(
                [register["condition"]]
                * len(image_paths)
            )

        return PreparedData(
            X=np.asarray(
                X,
                dtype=object
            ),
            y=np.asarray(y)
        )

    # ========================================================
    # AQUISIÇÃO
    # ========================================================

    def _prepare_acquisition(
        self,
        register,
        experimental_setup
    ):
        acquisition_id = Path(
            register["acquisition_file"]
        ).stem

        output_dir = (
            self.cache_dir
            / acquisition_id
        )

        raw_dir_path = (
            experimental_setup[
                "raw_dir_path"
            ]
        )

        channel_columns = (
            self._resolve_channel_columns(
                register,
                experimental_setup
            )
        )

        acquisition = get_acquisition_data(
            raw_dir_path=raw_dir_path,
            channels_columns=channel_columns,
            load_acquisition_func=(
                load_matlab_acquisition
            ),
            register=register
        )

        signal = np.asarray(
            acquisition,
            dtype=np.float64
        ).squeeze()

        if signal.ndim != 1:
            raise ValueError(
                "SpectrogramAdapter currently "
                "expects a single vibration channel. "
                f"Received shape: {signal.shape}"
            )

        num_segments = (
            len(signal)
            // self.segment_length
        )

        # ----------------------------------------------------
        # Reutiliza cache quando completo
        # ----------------------------------------------------

        existing = []

        if output_dir.exists():
            existing = sorted(
                output_dir.glob("*.png")
            )

        if len(existing) == num_segments:
            return [
                str(path)
                for path in existing
            ]

        # ----------------------------------------------------
        # Geração
        # ----------------------------------------------------

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Remove cache incompleto.
        for path in output_dir.glob(
            "*.png"
        ):
            path.unlink()

        sample_rate = float(
            register["sample_rate"]
        )

        image_paths = []

        for index in range(num_segments):

            start = (
                index
                * self.segment_length
            )

            end = (
                start
                + self.segment_length
            )

            segment = signal[
                start:end
            ]

            image = self._create_spectrogram(
                segment,
                sample_rate
            )

            output_file = (
                output_dir
                / (
                    f"{acquisition_id}_"
                    f"{index:04d}.png"
                )
            )

            image.save(
                output_file
            )

            image_paths.append(
                str(output_file)
            )

        return image_paths

    # ========================================================
    # CANAL
    # ========================================================

    @staticmethod
    def _resolve_channel_columns(
        register,
        experimental_setup
    ):
        channels_config = (
            experimental_setup[
                "channels_columns"
            ]
        )

        for rule, columns in (
            channels_config.items()
        ):
            key, values = rule.split(
                ":",
                maxsplit=1
            )

            accepted_values = (
                literal_eval(values)
            )

            if (
                register.get(key)
                in accepted_values
            ):
                return columns

        raise ValueError(
            "No channel configuration found "
            f"for register "
            f"{register['acquisition_file']}."
        )

    # ========================================================
    # ESPECTROGRAMA
    # ========================================================

    def _create_spectrogram(
        self,
        segment,
        sample_rate
    ):
        segment = self._normalize(
            segment
        )

        segment = detrend(
            segment,
            type="linear"
        )

        _, _, zxx = stft(
            segment,
            fs=sample_rate,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            detrend=False,
            boundary=None,
            padded=False
        )

        power = (
            np.abs(zxx) ** 2
        )

        epsilon = (
            np.finfo(
                np.float32
            ).eps
        )

        spectrogram_db = (
            10.0
            * np.log10(
                power + epsilon
            )
        )

        spectrogram_db = np.clip(
            spectrogram_db,
            self.db_min,
            self.db_max
        )

        normalized = (
            spectrogram_db
            - self.db_min
        ) / (
            self.db_max
            - self.db_min
        )

        rgba = colormaps["jet"](
            normalized
        )

        rgb = (
            rgba[:, :, :3]
            * 255
        ).astype(
            np.uint8
        )

        image = Image.fromarray(
            rgb
        )

        return image.resize(
            self.image_size,
            Image.Resampling.BILINEAR
        )

    # ========================================================
    # NORMALIZAÇÃO
    # ========================================================

    def _normalize(
        self,
        segment
    ):
        segment = np.asarray(
            segment,
            dtype=np.float64
        )

        if self.normalization == "raw":
            return segment

        if self.normalization == "rms":

            rms = np.sqrt(
                np.mean(
                    segment ** 2
                )
            )

            if rms == 0:
                return segment

            return segment / rms

        if self.normalization == "zscore":

            mean = np.mean(segment)
            std = np.std(segment)

            if std == 0:
                return (
                    segment - mean
                )

            return (
                segment - mean
            ) / std

        raise ValueError(
            "normalization must be "
            "'raw', 'rms' or 'zscore'."
        )