import bisect
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.utils import (
    download_file_from_register,
    filter_registers_by_key_value_absence,
    filter_registers_by_key_value_sequence,
    get_channels_from_register,
    load_matlab_acquisition,
    read_registers_from_config,
)


@dataclass(frozen=True)
class _IndexedRegister:
    register_idx: int
    start: int
    end: int
    label: str
    channels: Tuple[str, ...]


def _to_key_value_sequence(filters: Optional[Dict[str, Sequence[str]]]) -> List[Tuple[str, Sequence[str]]]:
    if not filters:
        return []
    return [(key, values) for key, values in filters.items()]


def _channel_length(file_path: str, channel: str) -> int:
    # Fast path: inspect only metadata from the MAT file.
    try:
        import scipy.io  # local import to keep module import cheap

        for name, shape, _ in scipy.io.whosmat(file_path):
            if name == channel or channel in name:
                if not shape:
                    return 0
                return int(shape[0])
    except Exception:
        pass

    # Fallback: load only the requested channel.
    acquisition = load_matlab_acquisition(file_path, channels=[channel])
    return int(acquisition.shape[0])


class CWRUSegmentDataset(Dataset):
    def __init__(
        self,
        config_path: str = "dataset/cwru/config.csv",
        raw_dir_path: str = "raw_data/cwru",
        channels_columns: Sequence[str] = ("DE",),
        segment_length: int = 1024,
        label_column: str = "condition",
        include_filters: Optional[Dict[str, Sequence[str]]] = None,
        exclude_filters: Optional[Dict[str, Sequence[str]]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        channels_first: bool = True,
        dtype: torch.dtype = torch.float32,
        max_cached_files: int = 4,
    ) -> None:
        if segment_length <= 0:
            raise ValueError("segment_length must be greater than zero.")
        if max_cached_files <= 0:
            raise ValueError("max_cached_files must be greater than zero.")

        self.config_path = config_path
        self.raw_dir_path = raw_dir_path
        self.channels_columns = tuple(channels_columns)
        self.segment_length = segment_length
        self.label_column = label_column
        self.transform = transform
        self.target_transform = target_transform
        self.channels_first = channels_first
        self.dtype = dtype
        self.max_cached_files = max_cached_files

        registers = read_registers_from_config(config_path)
        include_seq = _to_key_value_sequence(include_filters)
        exclude_seq = _to_key_value_sequence(exclude_filters)
        if include_seq:
            registers = filter_registers_by_key_value_sequence(registers, include_seq)
        if exclude_seq:
            registers = filter_registers_by_key_value_absence(registers, exclude_seq)

        self._registers = registers
        self._indexed_registers, self._length = self._build_index()
        self._end_offsets = [item.end for item in self._indexed_registers]
        self.classes = sorted({item.label for item in self._indexed_registers})
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self._target_by_entry = [self.class_to_idx[item.label] for item in self._indexed_registers]
        self._cache: OrderedDict[Tuple[int, Tuple[str, ...]], np.ndarray] = OrderedDict()

    def _build_index(self) -> Tuple[List[_IndexedRegister], int]:
        indexed: List[_IndexedRegister] = []
        offset = 0
        for register_idx, register in enumerate(self._registers):
            label = register.get(self.label_column)
            if label is None:
                raise KeyError(f"Label column '{self.label_column}' not found in register.")

            channels = tuple(get_channels_from_register(self.channels_columns, register))
            if not channels or any(channel in ("None", "") for channel in channels):
                continue

            file_path = os.path.join(self.raw_dir_path, register["filename"])
            if not os.path.isfile(file_path):
                download_file_from_register(self.raw_dir_path, register)

            num_points = _channel_length(file_path, channels[0])
            num_segments = num_points // self.segment_length
            if num_segments == 0:
                continue

            next_offset = offset + num_segments
            indexed.append(
                _IndexedRegister(
                    register_idx=register_idx,
                    start=offset,
                    end=next_offset,
                    label=label,
                    channels=channels,
                )
            )
            offset = next_offset
        return indexed, offset

    def _load_acquisition(self, register_idx: int, channels: Tuple[str, ...]) -> np.ndarray:
        key = (register_idx, channels)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        register = self._registers[register_idx]
        file_path = os.path.join(self.raw_dir_path, register["filename"])
        acquisition = load_matlab_acquisition(file_path, list(channels))
        if acquisition.ndim == 1:
            acquisition = acquisition[:, np.newaxis]
        acquisition = np.asarray(acquisition)
        self._cache[key] = acquisition
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cached_files:
            self._cache.popitem(last=False)
        return acquisition

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of bounds for dataset with size {self._length}.")

        entry_pos = bisect.bisect_right(self._end_offsets, idx)
        entry = self._indexed_registers[entry_pos]
        local_idx = idx - entry.start

        acquisition = self._load_acquisition(entry.register_idx, entry.channels)
        start = local_idx * self.segment_length
        end = start + self.segment_length
        segment = acquisition[start:end]

        x = torch.as_tensor(segment, dtype=self.dtype)
        if self.channels_first:
            x = x.transpose(0, 1).contiguous()
        y = torch.tensor(self.class_to_idx[entry.label], dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


def create_cwru_dataloader(
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    dataset_kwargs: Optional[Dict] = None,
) -> DataLoader:
    dataset = CWRUSegmentDataset(**(dataset_kwargs or {}))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def create_cwru_dataloaders(
    train_filters: Dict[str, Sequence[str]],
    val_filters: Dict[str, Sequence[str]],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last_train: bool = False,
    common_dataset_kwargs: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader]:
    common_dataset_kwargs = common_dataset_kwargs or {}
    train_dataset = CWRUSegmentDataset(
        include_filters=train_filters,
        **common_dataset_kwargs,
    )
    val_dataset = CWRUSegmentDataset(
        include_filters=val_filters,
        **common_dataset_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
