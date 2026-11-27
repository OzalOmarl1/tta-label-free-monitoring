MAE = |tahmin − gerçek doğruluk|, 16 kaymalı alan üzerinden; 5 model tohumu üzerinden ortalama ± std. src = kaynak modelin doğrulama kümesi referansı; matched = aynı TTA yöntemiyle işlenmiş doğrulama kümesi.

| koşul    | yöntem   | tahmin edici     | MAE (pp)     | yanlılık (pp)   |   Spearman ρ |
|:---------|:---------|:-----------------|:-------------|:----------------|-------------:|
| iid64    | Source   | AC               | 30.02 ± 0.54 | 30.02 ± 0.54    |        0.953 |
| iid64    | Source   | AC-TS (src)      | 22.58 ± 0.82 | 22.58 ± 0.82    |        0.95  |
| iid64    | Source   | AC-TS (matched)  | 22.58 ± 0.82 | 22.58 ± 0.82    |        0.95  |
| iid64    | Source   | DoC (src)        | 27.15 ± 0.70 | 27.15 ± 0.70    |        0.953 |
| iid64    | Source   | DoC (matched)    | 27.15 ± 0.70 | 27.15 ± 0.70    |        0.953 |
| iid64    | Source   | ATC-MC (src)     | 15.03 ± 1.04 | 15.03 ± 1.04    |        0.959 |
| iid64    | Source   | ATC-MC (matched) | 15.03 ± 1.04 | 15.03 ± 1.04    |        0.959 |
| iid64    | Source   | ATC-NE (src)     | 13.46 ± 1.41 | 13.46 ± 1.41    |        0.962 |
| iid64    | Source   | ATC-NE (matched) | 13.46 ± 1.41 | 13.46 ± 1.41    |        0.962 |
| iid64    | Source   | GDE              | 15.77 ± 0.37 | 15.77 ± 0.37    |        0.932 |
| iid64    | BN-adapt | AC               | 13.65 ± 0.45 | 13.65 ± 0.45    |        0.979 |
| iid64    | BN-adapt | AC-TS (src)      | 7.62 ± 0.40  | 7.62 ± 0.40     |        0.991 |
| iid64    | BN-adapt | AC-TS (matched)  | 7.55 ± 0.26  | 7.55 ± 0.26     |        0.991 |
| iid64    | BN-adapt | DoC (src)        | 10.78 ± 0.40 | 10.78 ± 0.40    |        0.979 |
| iid64    | BN-adapt | DoC (matched)    | 9.56 ± 0.38  | 9.56 ± 0.38     |        0.979 |
| iid64    | BN-adapt | ATC-MC (src)     | 3.57 ± 0.88  | 3.55 ± 0.93     |        0.991 |
| iid64    | BN-adapt | ATC-MC (matched) | 3.69 ± 0.59  | 3.68 ± 0.59     |        0.991 |
| iid64    | BN-adapt | ATC-NE (src)     | 2.58 ± 0.84  | 2.47 ± 1.00     |        0.988 |
| iid64    | BN-adapt | ATC-NE (matched) | 3.05 ± 0.60  | 3.03 ± 0.61     |        0.991 |
| iid64    | BN-adapt | GDE              | 1.64 ± 0.50  | 1.60 ± 0.54     |        0.979 |
| iid64    | TENT     | AC               | 13.79 ± 0.43 | 13.79 ± 0.43    |        0.924 |
| iid64    | TENT     | AC-TS (src)      | 9.78 ± 0.34  | 9.78 ± 0.34     |        0.924 |
| iid64    | TENT     | AC-TS (matched)  | 8.10 ± 0.27  | 8.10 ± 0.27     |        0.924 |
| iid64    | TENT     | DoC (src)        | 10.92 ± 0.32 | 10.92 ± 0.32    |        0.924 |
| iid64    | TENT     | DoC (matched)    | 9.21 ± 0.23  | 9.21 ± 0.23     |        0.924 |
| iid64    | TENT     | ATC-MC (src)     | 7.05 ± 0.87  | 7.05 ± 0.87     |        0.921 |
| iid64    | TENT     | ATC-MC (matched) | 4.56 ± 0.74  | 4.48 ± 0.64     |        0.924 |
| iid64    | TENT     | ATC-NE (src)     | 7.03 ± 0.92  | 7.03 ± 0.92     |        0.921 |
| iid64    | TENT     | ATC-NE (matched) | 4.51 ± 0.76  | 4.43 ± 0.69     |        0.921 |
| iid64    | TENT     | GDE              | 1.38 ± 0.42  | 1.32 ± 0.46     |        0.985 |
| sorted64 | Source   | AC               | 30.02 ± 0.54 | 30.02 ± 0.54    |        0.953 |
| sorted64 | Source   | AC-TS (src)      | 22.58 ± 0.82 | 22.58 ± 0.82    |        0.95  |
| sorted64 | Source   | AC-TS (matched)  | 22.58 ± 0.82 | 22.58 ± 0.82    |        0.95  |
| sorted64 | Source   | DoC (src)        | 27.15 ± 0.70 | 27.15 ± 0.70    |        0.953 |
| sorted64 | Source   | DoC (matched)    | 27.15 ± 0.70 | 27.15 ± 0.70    |        0.953 |
| sorted64 | Source   | ATC-MC (src)     | 15.03 ± 1.04 | 15.03 ± 1.04    |        0.959 |
| sorted64 | Source   | ATC-MC (matched) | 15.03 ± 1.04 | 15.03 ± 1.04    |        0.959 |
| sorted64 | Source   | ATC-NE (src)     | 13.46 ± 1.41 | 13.46 ± 1.41    |        0.962 |
| sorted64 | Source   | ATC-NE (matched) | 13.46 ± 1.41 | 13.46 ± 1.41    |        0.962 |
| sorted64 | Source   | GDE              | 15.77 ± 0.37 | 15.77 ± 0.37    |        0.932 |
| sorted64 | BN-adapt | AC               | 57.10 ± 0.44 | 57.10 ± 0.44    |        0.475 |
| sorted64 | BN-adapt | AC-TS (src)      | 46.17 ± 0.75 | 46.17 ± 0.75    |        0.389 |
| sorted64 | BN-adapt | AC-TS (matched)  | 46.05 ± 0.63 | 46.05 ± 0.63    |        0.389 |
| sorted64 | BN-adapt | DoC (src)        | 54.23 ± 0.54 | 54.23 ± 0.54    |        0.475 |
| sorted64 | BN-adapt | DoC (matched)    | 53.01 ± 0.47 | 53.01 ± 0.47    |        0.475 |
| sorted64 | BN-adapt | ATC-MC (src)     | 33.50 ± 2.19 | 33.50 ± 2.19    |        0.483 |
| sorted64 | BN-adapt | ATC-MC (matched) | 33.77 ± 1.42 | 33.77 ± 1.42    |        0.483 |
| sorted64 | BN-adapt | ATC-NE (src)     | 29.38 ± 2.19 | 29.38 ± 2.19    |        0.464 |
| sorted64 | BN-adapt | ATC-NE (matched) | 30.33 ± 1.60 | 30.33 ± 1.60    |        0.464 |
| sorted64 | BN-adapt | GDE              | 27.07 ± 0.52 | 27.07 ± 0.52    |        0.884 |
| sorted64 | TENT     | AC               | 63.84 ± 0.30 | 63.84 ± 0.30    |        0.171 |
| sorted64 | TENT     | AC-TS (src)      | 55.27 ± 0.33 | 55.27 ± 0.33    |        0.121 |
| sorted64 | TENT     | AC-TS (matched)  | 52.24 ± 0.43 | 52.24 ± 0.43    |        0.121 |
| sorted64 | TENT     | DoC (src)        | 60.97 ± 0.25 | 60.97 ± 0.25    |        0.171 |
| sorted64 | TENT     | DoC (matched)    | 59.25 ± 0.16 | 59.25 ± 0.16    |        0.171 |
| sorted64 | TENT     | ATC-MC (src)     | 45.82 ± 1.69 | 45.82 ± 1.69    |        0.191 |
| sorted64 | TENT     | ATC-MC (matched) | 40.39 ± 1.18 | 40.39 ± 1.18    |        0.265 |
| sorted64 | TENT     | ATC-NE (src)     | 43.93 ± 1.61 | 43.93 ± 1.61    |        0.068 |
| sorted64 | TENT     | ATC-NE (matched) | 38.85 ± 1.16 | 38.85 ± 1.16    |        0.174 |
| sorted64 | TENT     | GDE              | 25.99 ± 0.47 | 25.99 ± 0.47    |        0.841 |
