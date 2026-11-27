# AUDIT.md — Depo Denetim Raporu

**Denetim tarihi:** 2026-11-08
**Kapsam:** `src/`, `scripts/`, `tests/`, `data/`, `results/` (loglar + `OVERNIGHT_REPORT.md`), `Vize/` (proposal.html, sunum.html, kaynakca_dogrulama.md).
**Yöntem:** Kodun tamamı okundu; testler çalıştırıldı; iddialar `Final/scripts/verify_audit.py` ile gerçek çalıştırmalarla yeniden hesaplandı (çıktı: `Final/results/logs/audit_verification.json`). Kaynakça API ile yeniden doğrulandı (`Final/scripts/verify_refs.py`).

Durum etiketleri: **DÜZELTİLDİ** · **DOĞRULANDI** (iddia doğru çıktı) · **DOĞRULANAMADI** (kanıt mevcut değil) · **BELGELENDİ** (düzeltilemez; sınırlılık olarak raporlandı).

## Özet tablo

| # | Önem | Bulgu | Durum |
|---|---|---|---|
| A1 | Yüksek | `scripts/verify_datasets.py` var olmayan `ALL_CORRUPTIONS`'ı import ediyor → betik hiç çalışmıyor | DÜZELTİLDİ |
| A2 | Yüksek | `src/evaluate_phase1.verify_checkpoint_integrity` eski tek-anahtarlı hash formatını (`meta["sha256"]`) bekliyor; hash dosyası tohum bazlı formata geçtiği için `KeyError` → `scripts/run_phase1.py` çalışmıyor | DÜZELTİLDİ |
| A3 | Orta | `src/train_source.py` tohumu sabit kodluyor (42) ve log'u tohum eksiz dosyaya yazıyor; seed 43/44 modelleri depoda olmayan sürümlenmemiş bir değişiklikle üretilmiş | DÜZELTİLDİ (CLI + tohum ekli log; Final'de yeni betik) |
| A4 | Yüksek | `scripts/benchmark_e2e.py`'de TENT optimizer'ı BN dönüşümünden ÖNCE `model.parameters()` ile kuruluyor → yeni BN afin parametreleri optimizer'da yok, **TENT hiç güncellenmiyor**; öneri belgesindeki TENT süreleri bu hatalı yapılandırmadan geliyor | DÜZELTİLDİ (gösterim: V6) |
| A5 | Yüksek | `src/methods/tent.py` tahmini güncellemeden SONRA ikinci bir ileri geçişle üretiyor; resmi Tent'te tahmin, kaybın hesaplandığı ileri geçişten gelir. Faz-1 TENT sayıları bu sapmalı protokolle üretilmiş | DÜZELTİLDİ (`predict` parametresi; Final'de resmi protokol; etki ablasyonla ölçüldü) |
| A6 | Yüksek | Bozulma başına "%95 GA"lar n=3 tohumun bootstrap'ı → 15/15 bozulmada aralık tam olarak [min, max]; %95 GA değildir | DÜZELTİLDİ (Final'de t-GA / hiyerarşik bootstrap / Wilcoxon) |
| A7 | Yüksek | Raporda "Gürültü ailesi GA [%7.09, %9.21]" — hiç hesaplanmamış; iki farklı bozulmanın aralık uçlarının birleştirilmesi (ve gerçek alt uç 6.32) | DÜZELTİLDİ (doğru t-GA, n=3: [5.79, 9.84]) |
| A8 | Orta | "GATE B: tohum 42 önceki Faz-1 ile 0.00 farkla örtüştü" — önceki sonuç dosyasının üzerine yazılmış; karşılaştırma kanıtı yok | DOĞRULANAMADI → yerine bağımsız yeniden üretim yapıldı (V3) |
| A9 | Orta | "Tohum" tanımı model + akış sırasını birleştiriyor; model varyansı ile akış varyansı ayrıştırılamıyor | DÜZELTİLDİ (Final'de ayrı faktörler + akış ablasyonu) |
| A10 | Düşük | Seed 42 eğitim süresi log'da tam 47.5 dk (diğerleri tam hassasiyetli float) — elle girilmiş olabilir | DOĞRULANAMADI |
| A11 | Yüksek | Kaynakça: Recht ve ark. "Do CIFAR-10 Classifiers Generalize to CIFAR-10?" ICML 2019 olarak verilmiş; bu çalışma arXiv:1806.00451 (2018) ön baskısıdır (ICML 2019 makalesi ImageNet sürümüdür) | DÜZELTİLDİ |
| A12 | Orta | `Vize/kaynakca_dogrulama.md` "MATCH (Düzeltildi)" diyor ama `scripts/verify_refs.py` hâlâ yanlış yazar/arXiv kimliği içeriyor → rapor betikten üretilemez (elle düzenlenmiş); yıl kontrolü ±1 tolerans (2019↔2018 hatası yakalanmıyor); yayın yeri hiç doğrulanmıyor | DÜZELTİLDİ (yeni betik; tam eşleşme; yayın yeri API'den) |
| A13 | Düşük | Öneri belgesi kaynakçayı "Tümü [VERIFY]" diye etiketlerken doğrulama raporu "tümü MATCH" diyor — tutarsız durum | DÜZELTİLDİ (yeni öneri) |
| A14 | Orta | Bütçe modeli: Protokol A (60 koşum) "≈8 dk/tohum" tahmini; ölçülen Faz-1 135 koşum = 37.17 dk (≈12.4 dk/tohum, 45 koşum) → ~2× düşük tahmin; Protokol B'de CIFAR-10.1 koşumları ve doğrulayıcı hücrelerin 5–10 akışı bütçelenmemiş; EATA 15 sn varsayım | BELGELENDİ (Final'de bütçe ölçülen sürelerden) |
| A15 | Orta | EATA "ana deney yöntemi" olarak listelenmiş ama uygulanmamış; Fisher betiği hiç çalıştırılmamış; H1 EATA'ya bağımlı | BELGELENDİ (pivot sonrası kapsam dışı) |
| A16 | Düşük | `data/cifar-10-python.tar.gz` 2.2 MB, bozuk (gzip akışı yarıda kesik); kullanılmıyor | BELGELENDİ (parquet kullanılıyor; dosya silinmedi) |
| A17 | Orta | CIFAR-10-C resmi olmayan bir HF aynasından (YashP54/cifar10-c), sabit 128 baytlık npy başlığı varsayımıyla HTTP Range ile indirilmiş; sağlama toplamı yok | KISMEN DOĞRULANDI (V4: etiketler %100 hizalı, görüntü korelasyonu 0.81–0.98, Gauss σ kalıntısı 0.0995≈0.10 → şiddet 5 ile tutarlı); resmi Zenodo sağlama toplamı: DOĞRULANAMADI |
| A18 | Düşük | `requirements.txt`, README, çalıştırma betiği yok; `Final/` ve `experiments/` boş; Python `random` ve DataLoader işçi tohumları sabitlenmemiş | DÜZELTİLDİ (Final/) |
| A19 | Düşük | PACS modülü var ama veri yok (ölü kod); Dirichlet örnekleyici tekrar ile örnekliyor (α=0.1'de aynı görüntü akışta defalarca) | BELGELENDİ (Final'de kullanılmıyor) |
| A20 | Düşük | BN-adapt saf test-partisi istatistikleri kullanıyor (Schneider ve ark.'nın önsel ağırlıklı karışımı değil; Tent makalesindeki "BN" taban çizgisine karşılık gelir) — adlandırma nüansı | BELGELENDİ |
| A21 | Düşük | Faz-1 meta verisinde temiz doğruluklar sabit kodlanmış (hash dosyasından okunmuyor) | BELGELENDİ (değerler doğru çıktı, V2) |

## Doğrulanan iddialar (gerçek çalıştırma)

| İddia (kaynak) | Yeniden hesaplanan | Sonuç |
|---|---|---|
| Checkpoint SHA256 (42/43/44) | 3/3 eşleşti | DOĞRULANDI |
| Temiz test doğruluğu %93.86 / %93.44 / %93.88 | %93.86 / %93.44 / %93.88 | DOĞRULANDI |
| En iyi val %94.84 (ep 46) / %93.54 (ep 50) / %93.88 (ep 47) | checkpoint içi epok 46/50/47, val yeniden hesap aynı | DOĞRULANDI |
| ResNet-18 ≈11.2 M parametre | 11,173,962 | DOĞRULANDI |
| Faz-1 ortalamaları (Source 54.03, BN 75.82, TENT 79.54; Δ=+3.72) JSON ile tutarlı | ham koşumlardan yeniden ortalama: tutarsızlık yok | DOĞRULANDI (JSON içi tutarlılık) |
| Faz-1 süre 37.17 dk | JSON meta: 37.17 | DOĞRULANDI |
| Faz-1 koşumlarının yeniden üretilebilirliği | Source/BN 12/12 birebir; TENT maks. 0.13 pp | DOĞRULANDI (TENT'te küçük sapma) |
| Bozulma başına %95 GA | 15/15'te GA = [min, max] (n=3) | YANLIŞ (A6) |
| Gürültü ailesi GA [7.09, 9.21] | gerçek uçlar [6.32, 9.21]; doğru t-GA [5.79, 9.84] | YANLIŞ (A7) |
| Benchmark TENT süreleri | optimizer'da 0 BN parametresi, ağırlık değişmiyor | GEÇERSİZ ölçüm (A4) |

### V3 — Faz-1 kısmi yeniden üretim (orijinal kod yolu, seed 42/43 × 3 bozulma × 3 yöntem)

| Tohum | Bozulma | Yöntem | Raporlanan doğr. | Yeniden üretilen | |fark| (pp) |
|---|---|---|---|---|---|
| 42 | gaussian_noise | Source | 23.95 | 23.95 | 0.00 |
| 42 | gaussian_noise | BN-adapt | 67.44 | 67.44 | 0.00 |
| 42 | gaussian_noise | TENT | 74.53 | 74.47 | 0.06 |
| 42 | contrast | Source | 30.49 | 30.49 | 0.00 |
| 42 | contrast | BN-adapt | 84.37 | 84.37 | 0.00 |
| 42 | contrast | TENT | 83.83 | 83.83 | 0.00 |
| 42 | jpeg_compression | Source | 72.67 | 72.67 | 0.00 |
| 42 | jpeg_compression | BN-adapt | 72.02 | 72.02 | 0.00 |
| 42 | jpeg_compression | TENT | 75.79 | 75.79 | 0.00 |
| 43 | gaussian_noise | Source | 29.57 | 29.57 | 0.00 |
| 43 | gaussian_noise | BN-adapt | 63.93 | 63.93 | 0.00 |
| 43 | gaussian_noise | TENT | 71.73 | 71.86 | 0.13 |
| 43 | contrast | Source | 27.53 | 27.53 | 0.00 |
| 43 | contrast | BN-adapt | 82.60 | 82.60 | 0.00 |
| 43 | contrast | TENT | 82.51 | 82.49 | 0.02 |
| 43 | jpeg_compression | Source | 74.56 | 74.56 | 0.00 |
| 43 | jpeg_compression | BN-adapt | 70.00 | 70.00 | 0.00 |
| 43 | jpeg_compression | TENT | 75.64 | 75.64 | 0.00 |

**Sonuç:** Source ve BN-adapt koşumlarının 12/12'si birebir (0.00 pp) yeniden üretildi. TENT koşumlarında en büyük fark 0.13 pp (ortalama 0.035 pp) — GPU üzerindeki geri yayılımın bit düzeyinde deterministik olmamasıyla tutarlı. Dolayısıyla Faz-1 sayıları **pratikte yeniden üretilebilir**, ancak raporun "tüm 45 koşumda birebir 0.00 fark" iddiası TENT için **doğrulanmadı** (A8).

## Yapılan düzeltmeler (dosya bazında)

* `scripts/verify_datasets.py` — import düzeltildi (A1).
* `src/evaluate_phase1.py` — hash doğrulaması tohum bazlı formatı destekliyor (A2); TENT protokolü açıkça `after_update` (Faz-1 yeniden üretimi için) (A5).
* `src/methods/tent.py` — `predict={"official","after_update"}` parametresi; varsayılan resmi protokol (A5).
* `src/train_source.py` — `--seed/--epochs` CLI, tohum ekli log (A3).
* `scripts/benchmark_e2e.py` — TENT optimizer'ı dönüşümden sonra kuruluyor (A4).
* `scripts/run_phase1.py` — kullanımdan kaldırıldı (tek checkpoint ile sahte tohumlar + aynı dosyanın üzerine yazma) (A2/A8).
* Yeni temiz proje `Final/`: termal koruyucu, deterministik tohumlama, ayrık model/akış faktörleri, doğru istatistik, ön kayıt, kaynakçanın API ile üretimi, README, requirements, çalıştırma betiği.
* Eski öneri ve sunum (`Vize/proposal.html`, `Vize/sunum.html`, `Vize/kaynakca_dogrulama.md`) sınıf arkadaşıyla çakışan konu nedeniyle `Arsiv/eski_konu_vize/` altına taşındı (silinmedi); `results/OVERNIGHT_REPORT.md` başına denetim notu eklendi.

## Çalıştırma sırasında karşılaşılan ve belgelenen durumlar

* Orijinal CPU tabanlı eğitim veri hattı, bu dizüstü bilgisayarda CPU'yu (ACPI termal bölge) sürekli 85 °C'ye taşıdı; termal koruyucu iş yükünü defalarca duraklattı. Yarıda kesilen seed-45 koşusu (epok 23) ve eski BN uygulamasıyla alınan 90 ön koşum silinmeden `Final/_aborted/` altına taşındı; nihai sonuçlarda kullanılmadı.
* Yeni modeller (45, 46) aynı protokolle, ancak augmentasyon GPU'da (aynı dağılım, farklı RNG akışı) eğitildi.
