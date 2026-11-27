# Etiketli Veri Olmadan Adapte Edilmiş Modele Güvenebilir miyiz?

**Test Zamanı Adaptasyonu Sonrasında Etiketsiz Doğruluk Tahmin Edicilerinin Geçerliliği ve Uyarlama Kapısı**

*Can We Trust the Adapted Model Without Labels? Validity of Label-Free Accuracy Estimators after Test-Time Adaptation and an Adapt-or-Not Gate*

Ozal Omarli — Danışman: Dr. Öğr. Üyesi Ramin Abbaszadi — OSTİM Teknik Üniversitesi, Derin Öğrenme Final Projesi

---

## Özet

Bu depo, test zamanı adaptasyonundan (TTA) sonra modelin doğruluğunun etiket kullanmadan ne ölçüde tahmin edilebileceğini inceleyen çalışmanın kodunu, ön kaydını, sonuçlarını ve belgelerini içerir. Beş bağımsız CIFAR ResNet-18 modeli; temiz CIFAR-10, 15 CIFAR-10-C bozulması (şiddet 5) ve CIFAR-10.1 üzerinde, i.i.d. ve etiket-sıralı akışlarda Source, BN-adapt ve TENT ile çalıştırılmıştır. Ardından AC, AC-TS, DoC, ATC-MC, ATC-NE ve GDE tahmin edicilerinin hataları ön kayıtlı hipotez testleriyle ölçülmüş ve etiketsiz bir "uyarla / uyarlama" kapısı değerlendirilmiştir.

**Makale:** [Türkçe](docs/makale/TMLR_Makale_TR.pdf) · [English](docs/makale/TMLR_Article_EN.pdf)

## Depo yapısı

```
Final/                  ana proje (ttamon paketi)
  ttamon/               veri, modeller, yöntemler, tahmin ediciler, metrikler, istatistik, termal koruyucu
  scripts/              eğitim, tahmin toplama, analiz, şekiller, kaynakça doğrulama, LaTeX üretimi
  tests/                pytest birim testleri
  results/              all_runs.csv, summary.json, tables/, logs/   (ham logitler: Release)
  figures/              analiz şekilleri
  paper/                kaynakça kayıtları ve doğrulama raporları
  latex/                makale, öneri, rapor ve literatür derlemesi şablonları + TMLR stil dosyaları
  presentation/         HTML sunum kaynakları ve üreticileri
  PREREGISTRATION.md    ön kayıtlı hipotezler ve sapmalar
src/, scripts/, tests/  denetlenen orijinal kod tabanı (Final/scripts/verify_audit.py bunu kullanır)
results/                orijinal kod tabanının kayıtları (tohum 42–44 eğitim eğrileri dahil)
AUDIT.md                orijinal kod tabanının denetim bulguları
data/                   veri kümeleri (depoda yok; bkz. data/README.md)
docs/                   teslim belgeleri (PDF/HTML)
```

## Büyük dosyalar (GitHub Releases)

Model ağırlıkları ve ham logitler depoya değil, **Releases** sayfasına (`v1.0`) eklenmiştir:

| Dosya | Yerleştirileceği yer |
|---|---|
| `resnet18_cifar10_seed42.pth` … `seed46.pth` | `src/models/checkpoints/` |
| `raw_logits.zip` | `Final/results/` içine açın (→ `Final/results/raw/`) |
| `SHA256SUMS.txt` | indirilen dosyaların doğrulanması için |

Checkpoint SHA256 değerleri `Final/results/logs/checkpoint_hashes_final.json` ile aynıdır.

## Kurulum

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r Final\requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
cd Final
..\.venv\Scripts\python.exe -m pytest -q
```

## Yeniden üretim

* **Yalnızca tablo ve şekiller** (GPU ve yeniden eğitim gerekmez): veri kümelerini `data/README.md`'ye göre indirin (etiketler buradan okunur), `raw_logits.zip` dosyasını `Final/results/` içine açın, ardından `Final/` içinde `python scripts/analyze.py` ve `python scripts/make_figures.py` çalıştırın.
* **Tüm hat:** veri kümelerini `data/README.md`'ye göre indirin, checkpoint'leri yerleştirin ve `Final/run_all.ps1` (Windows) ya da `Final/run_all.sh` çalıştırın.

Makale ve sunumdaki tüm sayılar `Final/results/summary.json` ve `Final/results/tables/` dosyalarından otomatik okunur. Tohumlar, ortam ve donanım ayrıntıları `Final/README.md` dosyasındadır.

## Lisans notu

`Final/latex/_style/tmlr-style-file-main/` altındaki TMLR stil dosyaları kendi lisansıyla (Apache 2.0) dağıtılır. CIFAR-10, CIFAR-10-C ve CIFAR-10.1 veri kümeleri bu depoda yer almaz; kendi kaynaklarının koşullarına tabidir.
