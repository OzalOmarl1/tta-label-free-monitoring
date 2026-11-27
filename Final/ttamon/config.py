"""Deney tasarımı — tüm sabitler tek yerde (ön kayıt: PREREGISTRATION.md)."""

MODEL_SEEDS = [42, 43, 44, 45, 46]        # bağımsız eğitilmiş 5 kaynak model
MAIN_STREAM_SEED = 2026                    # tüm modeller AYNI akışı görür (model ve akış etkisi ayrıştırılır)
STREAM_ABLATION_SEEDS = [11, 12, 13]       # akış sırası varyansı (yalnızca model 42)
STREAM_ABLATION_MODEL = 42

# koşul adı -> (akış türü, batch size)
CONDITIONS = {
    "iid64": ("iid", 64),        # ana koşul: i.i.d. akış, N=64
    "sorted64": ("sorted", 64),  # i.i.d. olmayan (etiket-sıralı) akış, N=64
}
# SAPMA (PREREGISTRATION.md §Sapmalar): "iid16" (N=16) koşulu ve TENT lr=1e-4 ablasyonu, termal kısıtlar
# nedeniyle hiçbir sonuç görülmeden kapsamdan çıkarıldı.
DROPPED_CONDITIONS = {"iid16": ("iid", 16)}

MAIN_METHODS = ["Source", "BN-adapt", "TENT"]

# yöntem varyantı adı -> (temel yöntem, kwargs)
VARIANTS = {
    "Source": ("Source", {}),
    "BN-adapt": ("BN-adapt", {}),
    "TENT": ("TENT", {"lr": 1e-3, "steps": 1, "predict": "official"}),
    "TENT-after": ("TENT", {"lr": 1e-3, "steps": 1, "predict": "after_update"}),  # orijinal kodun davranışı
    "TENT-lr1e-4": ("TENT", {"lr": 1e-4, "steps": 1, "predict": "official"}),
    "TENT-lr1e-2": ("TENT", {"lr": 1e-2, "steps": 1, "predict": "official"}),
}
ABLATION_VARIANTS = ["TENT-after", "TENT-lr1e-2"]   # yalnızca iid64'te
STREAM_METHODS = ["BN-adapt", "TENT"]   # Source tahminleri akış sırasından bağımsızdır
