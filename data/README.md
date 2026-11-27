# Datasets

The data files are not included in the repository because of their size. The code (`Final/ttamon/data.py`) expects the following layout in this folder. To use a different location, set the `TTAMON_DATA` environment variable.

```
data/
├── cifar10_train.parquet        # CIFAR-10 train (HuggingFace parquet; columns: img, label)
├── cifar10_test.parquet         # CIFAR-10 test
├── cifar10_c/
│   ├── <corruption>_s5.npy      # 15 corruptions, severity 5 (e.g. gaussian_noise_s5.npy)
│   └── labels_s5.npy
└── cifar10_1/
    ├── cifar10.1_v6_data.npy
    └── cifar10.1_v6_labels.npy
```

## Sources

* **CIFAR-10:** parquet files of the HuggingFace `uoft-cs/cifar10` dataset (`img`, `label` columns).
* **CIFAR-10-C (severity 5):** downloaded with `scripts/download_cifar10_c_curl.py` from an unofficial mirror (`huggingface.co/datasets/YashP54/cifar10-c`). Label/image alignment and severity level were checked with `scripts/verify_datasets.py`; agreement with the official checksum could not be verified.
* **CIFAR-10.1 (v6):** https://github.com/modestyachts/CIFAR-10.1 (`datasets/cifar10.1_v6_data.npy`, `datasets/cifar10.1_v6_labels.npy`).

The `data/cache/` folder is created automatically on the first run.
