from dataset.cwru.pytorch import CWRUSegmentDataset, create_cwru_dataloader

def check_pytorch_data_pipeline(batch_size: int = 8) -> None:
    dataset_kwargs = {
        "channels_columns": ("DE",),
        "segment_length": 1024,
    }
    dataset = CWRUSegmentDataset(**dataset_kwargs)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check data/config paths and filters.")

    sample_x, sample_y = dataset[0]
    loader = create_cwru_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        dataset_kwargs=dataset_kwargs,
    )
    batch_x, batch_y = next(iter(loader))

    print("PyTorch data pipeline check")
    print(f"- dataset_len: {len(dataset)}")
    print(f"- num_classes: {len(dataset.classes)}")
    print(f"- sample_x_shape: {tuple(sample_x.shape)}")
    print(f"- sample_y: {int(sample_y)}")
    print(f"- batch_x_shape: {tuple(batch_x.shape)}")
    print(f"- batch_y_shape: {tuple(batch_y.shape)}")
    print(f"- batch_y_unique: {batch_y.unique().tolist()}")
