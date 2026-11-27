# Termal Koruma — Desteklenen Özellikler Raporu

**Makine:** Acer Nitro AN515-58 · Intel Core i5-12500H (12 çekirdek / 16 iş parçacığı) · NVIDIA GeForce RTX 4050 Laptop GPU (6 GB) · Windows 11 Pro · yönetici yetkisi yok
**Betik:** `Final/ttamon/thermal_guard.py` (süreç içi koruyucu + bağımsız arka plan izleyici)
**Algılama komutu:** `python -m ttamon.thermal_guard --probe`

| Özellik | Durum | Açıklama |
|---|---|---|
| GPU sıcaklığı | **Destekleniyor** | `nvidia-smi --query-gpu=temperature.gpu` (NVIDIA resmi aracı) |
| GPU kullanım oranı | **Destekleniyor** | `nvidia-smi --query-gpu=utilization.gpu` (loglanır) |
| CPU sıcaklığı | **Kısmen (vekil)** | `psutil.sensors_temperatures()` Windows'ta yok. Yönetici gerektirmeyen `Win32_PerfFormattedData_Counters_ThermalZoneInformation` (ACPI `\_TZ.TZ00`, onda bir Kelvin) okunur. Bu, çekirdek/paket sıcaklığı değil ACPI termal bölge sıcaklığıdır; yaklaşık 5 sn çözünürlükle güncellenir. |
| `MSAcpi_ThermalZoneTemperature` (WMI) | Desteklenmiyor | "Access denied" — yönetici yetkisi gerektirir; denenir, başarısızsa atlanır |
| GPU fan hızı okuma | Desteklenmiyor | Dizüstü GPU'da `nvidia-smi fan.speed` = `[N/A]` |
| Fan kontrolü | **Kullanılmadı** | İşletim sisteminin resmi bir fan kontrol API'si yok (üretici aracı NitroSense resmi programatik arayüz sunmuyor). Kural gereği yalnızca iş yükü kısıldı. |
| BIOS / firmware / voltaj / saat hızı / overclock | **Asla değiştirilmez** | Kod bu tür hiçbir çağrı içermez |

## Politika
* Yumuşak sınırlar: **CPU 85 °C, GPU 80 °C**.
* `warn` bölgesi (sınır − 5 °C): her parti öncesi 0,05 sn uyku (duty-cycle).
* `hot` (sınıra ulaşıldı): iş yükü **tamamen duraklatılır**; CPU ≤ 77 °C ve GPU ≤ 72 °C olunca (8 °C histerezis) devam eder.
* Batch size / worker azaltma: `recommend_batch_size()` / `recommend_workers()` mevcuttur, ancak **TTA deneylerinde kullanılmaz** — TTA sonucu batch size'a bağlı olduğundan bu, deney sonuçlarını değiştirirdi. Eğitimde de batch size sabit tutuldu (optimizasyon protokolü). Böylece termal olaylar yalnızca süreyi uzatır, sonuçları etkilemez.

## Gözlenen davranış (bu projede)
Ayrıntılı CSV logları: `Final/results/logs/thermal_*.csv`; özetler `thermal_collect_summary.json` ve `train_curves_seed*.json` (`thermal` alanı).

| İş yükü | Duraklatma sayısı | Toplam duraklatma | Not |
|---|---|---|---|
| Kaynak model eğitimi, seed 45 (GPU veri hattı) | 25 | 11,5 dk | deney toplama ile eşzamanlı |
| Kaynak model eğitimi, seed 46 | 9 | 3,7 dk | |
| TTA tahmin toplama (tüm tamamlanan koşular) | 30 | 12,9 dk | `thermal_collect_summary.json` |
| İptal edilen CPU veri hatlı eğitim denemesi | — | — | CPU sürekli 85 °C'ye çıktı; GPU'ya taşındı (AUDIT) |

* **GPU** hiçbir zaman 80 °C sınırına ulaşmadı (gözlenen tepe **74,4 °C**); kısmaların tamamı CPU (ACPI bölgesi) kaynaklıdır.
* **CPU** gözlenen tepe **90,1 °C** (ACPI bölgesi). Neden: deneylerde CPU sıcaklığı her 3 GPU okumasında bir (≈15 sn) okunuyordu; ACPI bölgesi bu sürede 85 °C eşiğini aşabildi. Koruyucu her aşımda iş yükünü durdurdu ve CPU ≤ 77 °C olunca devam etti. Sonradan varsayılan `cpu_poll_every=1` (≈5 sn) yapıldı.
* Termal kısma sonuçları etkilemez: batch size, sıra ve tohumlar sabittir; yalnızca duvar saati süresi uzar.

