# Ön Kayıt (Pre-registration) — Ana deneyler koşturulmadan önce yazılmıştır

**Tarih:** 2026-11-08 (ana deneyler ve ek modeller [seed 45, 46] tamamlanmadan önce)
**Proje:** Etiketli Veri Olmadan Adapte Edilmiş Modele Güvenebilir miyiz? TTA Sonrası Etiketsiz Doğruluk Tahmin Edicilerinin Geçerliliği

## Şeffaflık notu
Faz-1 pilotunda (orijinal konu) CIFAR-10-C şiddet-5 üzerinde Source/BN-adapt/TENT doğrulukları ve ECE
değerleri görülmüştür (TENT ECE ≈ BN-adapt ECE). Bu nedenle H1 tamamen "kör" değildir. H2–H5 ile ilgili
hiçbir ölçüm (tahmin edici hataları, uyuşma, kapı pişmanlığı, i.i.d. olmayan akış, N=16) bu yazı
yazılırken mevcut değildir.

## Birimler ve tasarım
* 5 bağımsız kaynak model (tohum 42–46), 17 alan (temiz, 15× CIFAR-10-C şiddet 5, CIFAR-10.1),
  3 yöntem (Source, BN-adapt, TENT [Adam, lr=1e-3, 1 adım, resmi tahmin protokolü]),
  3 koşul: `iid64` (ana), `sorted64` (etiket-sıralı, i.i.d. olmayan akış), `iid16`.
* Tüm modeller aynı akışı görür (akış tohumu 2026); akış-sırası varyansı ayrı ablasyonda ölçülür.
* İstatistiksel birim: **alan** (n = 16 kaymalı alan; temiz alan ayrıca raporlanır). Tohumlar alan içinde
  ortalanır (sahte tekrar/pseudo-replication önlenir); tohumlar arası std ayrıca raporlanır.
* Testler: iki yönlü Wilcoxon işaretli sıra testi, Holm düzeltmesi (her hipotez ailesi içinde), α = 0.05.
  Etki büyüklüğü: eşleştirilmiş sıra-biserial korelasyon. GA: alan→tohum hiyerarşik bootstrap (10.000).

## Hipotezler
* **H1 (Kalibrasyon):** `iid64`'te kaymalı alanlarda TENT'in aşırı güveni (ortalama güven − doğruluk),
  BN-adapt'ınkinden büyüktür. Test: Wilcoxon (TENT vs BN-adapt), n=16.
* **H2 (Tahmin edici geçerliliği):** Kaynak-referanslı güven tabanlı tahmin edicilerin (AC, AC-TS, DoC,
  ATC-MC, ATC-NE) mutlak tahmin hatası, TENT ile adapte edilmiş modelde BN-adapt'a göre daha büyüktür.
  Test: her tahmin edici için Wilcoxon (|hata|_TENT vs |hata|_BN), n=16, Holm (5 test).
* **H3 (Uyuşma sağlamlığı):** Bağımsız modeller arası uyuşmaya dayalı GDE tahmin edicisinin MAE'si, adapte
  edilmiş modellerde (BN-adapt ve TENT) en iyi güven tabanlı tahmin edicinin MAE'sinden düşüktür.
  "En iyi" tahmin edici, *temiz doğrulama kümesindeki* değil, test alanlarındaki ortalama MAE ile seçildiğinden
  bu karşılaştırma GDE aleyhine muhafazakârdır. Test: Wilcoxon, n=16, Holm (2 test).
* **H4 (i.i.d. olmayan akış):** `sorted64` koşulunda BN-adapt ve TENT, Source'un altına düşer ve
  güven tabanlı tahmin edicilerin MAE'si `iid64`'e göre artar. Test: Wilcoxon, n=16.
* **H5 (Etiketsiz uyarlama kapısı):** Her (koşul, alan, model) için {Source, BN-adapt, TENT} içinden
  tahmini doğruluğu en yüksek olanı seçen kapı, en iyi *sabit* politikadan (koşullar üzerinden havuzlanmış)
  daha düşük ortalama pişmanlığa (oracle − seçilen doğruluk) sahiptir. Test: Wilcoxon, n = 3 koşul × 16 alan = 48 (sapma sonrası: 2 koşul × 16 = 32; bkz. Sapmalar).

**Ek netleştirme (analiz kodu yazılırken, hiçbir kapı/tahmin edici sonucu görülmeden eklendi):**
H5'teki *birincil* kapı GDE (uyuşma) tahminlerini kullanır; diğer tahmin edicilerle kurulan kapılar
keşifseldir. H3'teki "en iyi güven tabanlı tahmin edici", 5 tahmin edici × 2 referans türü (kaynak / eşleşik)
= 10 aday arasından test MAE'si en düşük olandır. H4 için birincil ölçüm: (i) BN-adapt ve TENT doğruluğu
vs Source, (ii) 5 güven tabanlı tahmin edicinin (kaynak referanslı) ortalama |hata|'sı `sorted64` vs `iid64`;
bu 4 test birlikte Holm ile düzeltilir.

## Karar kuralı
Holm-düzeltilmiş p < 0.05 **ve** etki yönü hipotezle uyumlu → "desteklendi"; p ≥ 0.05 → "desteklenmedi
(kanıt yetersiz)"; anlamlı ve ters yön → "çürütüldü". Tüm sonuçlar (olumsuz olanlar dahil) raporlanır.

## Keşifsel (exploratory) analizler
TENT öğrenme oranı (1e-4, 1e-2), orijinal koddaki "güncelleme-sonrası tahmin" protokolü, referans türü
(kaynak model vs yöntemle eşleşik doğrulama), akış sırası varyansı, sınıf bazlı hata analizi.

## Sapmalar (Deviations) — 2026-11-08, hiçbir hipotez sonucu analiz edilmeden önce
1. **`iid16` koşulu kapsamdan çıkarıldı.** Neden: Dizüstü bilgisayarda CPU ve GPU ortak soğutmayı paylaştığı için
   termal koruyucu (CPU 85 °C / GPU 80 °C) iş yükünü sık sık duraklattı; tam plan ~10 saat sürecekti. N=16 TENT
   koşumları en pahalı kalemdi. Sonuç: H1–H4 zaten `iid64`/`sorted64` üzerinde tanımlıdır; **H5 havuzu 3 yerine
   2 koşul × 16 alan = 32 birim** üzerinden test edilir.
2. **TENT lr=1e-4 ablasyonu çıkarıldı** (keşifsel; lr=1e-2 ve güncelleme-sonrası tahmin ablasyonları korunur).
3. **Akış-sırası ablasyonundan Source çıkarıldı** (Source tahmini akış sırasından matematiksel olarak bağımsızdır).
Bu sapmalar yazıldığında yalnızca seed 42–44 için iid64 koşulunun ilk ~80 koşumu mevcuttu ve hiçbiri analiz edilmemişti.
