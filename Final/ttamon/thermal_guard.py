"""
Termal Koruma (Thermal Guard) Modülü
====================================

Amaç: Uzun GPU/CPU iş yüklerinde (eğitim, TTA deneyleri) CPU ve GPU sıcaklıklarını
arka planda izleyip CSV'ye kaydetmek ve yumuşak sınırlara (CPU 85 °C, GPU 80 °C)
ulaşıldığında iş yükünü OTOMATİK olarak azaltmak; soğuduktan sonra devam etmek.

GÜVENLİK İLKELERİ (değiştirilemez):
  * BIOS, firmware, voltaj, saat hızı / overclock ayarlarına KESİNLİKLE dokunulmaz.
  * Fan kontrolü yapılmaz: Bu makinede (Acer Nitro AN515-58, Windows 11) işletim
    sisteminin resmi olarak desteklediği bir fan kontrol API'si yoktur; nvidia-smi
    dizüstü GPU'da fan hızını "[N/A]" döndürür. Bu nedenle yalnızca iş yükü kısılır.
  * Kısma yöntemleri: (1) adımlar arası kısa uyku (duty-cycle), (2) soğuyana kadar
    tam duraklatma, (3) protokolü etkilemeyen iş yükleri için batch size / worker
    önerisini yarıya indirme. TTA deneylerinde batch size bilimsel bir değişken
    olduğundan (TTA sonucu batch size'a bağlıdır) deneylerde YALNIZCA uyku/duraklatma
    kullanılır; böylece termal kısma sonuçları değiştirmez, yalnızca süreyi uzatır.

Sensör kaynakları (otomatik algılanır, desteklenenler THERMAL_SUPPORT.md'de raporlanır):
  * GPU: `nvidia-smi --query-gpu=temperature.gpu` (NVIDIA resmi aracı).
  * CPU (Linux): psutil.sensors_temperatures().
  * CPU (Windows): psutil bu platformda sıcaklık sağlamaz. Yönetici yetkisi
    gerektirmeyen "Win32_PerfFormattedData_Counters_ThermalZoneInformation"
    performans sayacı (ACPI termal bölge, Kelvin) okunur. Bu, çekirdek/paket
    sıcaklığı değil ACPI bölge sıcaklığıdır (vekil ölçüm). MSAcpi_ThermalZoneTemperature
    yönetici yetkisi ister; denenir, erişim yoksa atlanır.

Kullanım (süreç içi):
    from ttamon.thermal_guard import ThermalGuard
    guard = ThermalGuard(log_csv="results/thermal_log.csv")
    guard.start()
    for batch in loader:
        guard.throttle()           # sıcaksa bekler, ılıksa kısa uyur
        ...
    guard.stop()

Kullanım (bağımsız arka plan izleyici):
    python -m ttamon.thermal_guard --monitor --log results/thermal_monitor.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# ---------------------------------------------------------------------------
# Sensör okuyucular
# ---------------------------------------------------------------------------
def read_gpu_temp() -> Optional[float]:
    """NVIDIA GPU sıcaklığını (°C) nvidia-smi ile okur; yoksa None."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=CREATE_NO_WINDOW,
        ).stdout.strip().splitlines()
        vals = [float(v) for v in out if v.strip() and v.strip() != "[N/A]"]
        return max(vals) if vals else None
    except Exception:
        return None


def read_gpu_util() -> Optional[float]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=CREATE_NO_WINDOW,
        ).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except Exception:
        return None


def _read_cpu_temp_linux() -> Optional[float]:
    if psutil is None or not hasattr(psutil, "sensors_temperatures"):
        return None
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        return None
    vals = []
    for name, entries in (temps or {}).items():
        for e in entries:
            if e.current and ("core" in (e.label or "").lower() or name in ("coretemp", "k10temp", "cpu_thermal", "acpitz")):
                vals.append(float(e.current))
    return max(vals) if vals else None


def _read_cpu_temp_windows() -> Tuple[Optional[float], str]:
    """Windows ACPI termal bölge sıcaklığı (°C). (değer, kaynak) döndürür."""
    ps = shutil.which("powershell") or shutil.which("powershell.exe")
    if ps is None:
        return None, "none"
    queries = [
        ("perfcounter_thermalzone",
         "Get-CimInstance -Namespace root/cimv2 -ClassName "
         "Win32_PerfFormattedData_Counters_ThermalZoneInformation | "
         "ForEach-Object { $_.HighPrecisionTemperature }"),
        ("msacpi_thermalzone",
         "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature | "
         "ForEach-Object { $_.CurrentTemperature }"),
    ]
    for src, q in queries:
        try:
            out = subprocess.run(
                [ps, "-NoProfile", "-NonInteractive", "-Command", q],
                capture_output=True, text=True, timeout=15, creationflags=CREATE_NO_WINDOW,
            ).stdout.strip().split()
            vals = []
            for v in out:
                try:
                    vals.append(float(v) / 10.0 - 273.15)  # onda bir Kelvin -> °C
                except ValueError:
                    pass
            vals = [v for v in vals if 0.0 < v < 125.0]
            if vals:
                return max(vals), src
        except Exception:
            continue
    return None, "none"


def read_cpu_temp() -> Tuple[Optional[float], str]:
    if sys.platform.startswith("linux"):
        v = _read_cpu_temp_linux()
        return v, ("psutil" if v is not None else "none")
    if os.name == "nt":
        return _read_cpu_temp_windows()
    return None, "none"


def probe_support() -> dict:
    """Bu makinede hangi termal özelliklerin desteklendiğini raporlar."""
    gpu = read_gpu_temp()
    cpu, cpu_src = read_cpu_temp()
    fan = None
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            fan = subprocess.run(
                [exe, "--query-gpu=fan.speed", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, creationflags=CREATE_NO_WINDOW,
            ).stdout.strip()
        except Exception:
            fan = None
    return {
        "platform": sys.platform,
        "gpu_temp_supported": gpu is not None,
        "gpu_temp_now_c": gpu,
        "gpu_fan_query": fan,
        "cpu_temp_supported": cpu is not None,
        "cpu_temp_source": cpu_src,
        "cpu_temp_now_c": None if cpu is None else round(cpu, 1),
        "psutil_sensors_temperatures": bool(psutil is not None and hasattr(psutil, "sensors_temperatures")),
        "fan_control": "DESTEKLENMİYOR / KULLANILMADI (resmi OS API yok; yalnızca iş yükü kısılır)",
        "bios_firmware_voltage_oc_changes": "YAPILMAZ",
    }


# ---------------------------------------------------------------------------
# Koruyucu
# ---------------------------------------------------------------------------
@dataclass
class ThermalGuard:
    cpu_limit: float = 85.0
    gpu_limit: float = 80.0
    hysteresis: float = 8.0          # devam etmek için limitin bu kadar altına inmeli
    warn_margin: float = 5.0         # limit - warn_margin üstü: kısa uykularla kısma
    poll_seconds: float = 5.0
    cpu_poll_every: int = 1          # her turda CPU oku (deneylerde 3 kullanıldı; 15 sn gecikme 85 °C aşımına yol açtı)
    warn_sleep: float = 0.05         # ılık bölgede her throttle() çağrısında uyku (sn)
    max_pause_seconds: float = 1800  # tek bir duraklatmanın üst sınırı (güvenlik amaçlı log)
    log_csv: Optional[str] = None
    state_file: Optional[str] = None
    verbose: bool = True

    cpu_temp: Optional[float] = field(default=None, init=False)
    gpu_temp: Optional[float] = field(default=None, init=False)
    cpu_source: str = field(default="none", init=False)
    level: str = field(default="ok", init=False)   # ok | warn | hot
    n_pauses: int = field(default=0, init=False)
    paused_seconds: float = field(default=0.0, init=False)
    n_warn_sleeps: int = field(default=0, init=False)

    def __post_init__(self):
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._tick = 0
        if self.log_csv:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_csv)), exist_ok=True)
            new = not os.path.exists(self.log_csv)
            self._fh = open(self.log_csv, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._fh)
            if new:
                self._writer.writerow(["unix_time", "iso_time", "cpu_temp_c", "cpu_source",
                                       "gpu_temp_c", "gpu_util_pct", "level", "event"])
        else:
            self._fh = None
            self._writer = None

    # -- izleme döngüsü ----------------------------------------------------
    def _classify(self) -> str:
        hot = (self.cpu_temp is not None and self.cpu_temp >= self.cpu_limit) or \
              (self.gpu_temp is not None and self.gpu_temp >= self.gpu_limit)
        if hot:
            return "hot"
        warm = (self.cpu_temp is not None and self.cpu_temp >= self.cpu_limit - self.warn_margin) or \
               (self.gpu_temp is not None and self.gpu_temp >= self.gpu_limit - self.warn_margin)
        return "warn" if warm else "ok"

    def _cool_enough(self) -> bool:
        ok_cpu = self.cpu_temp is None or self.cpu_temp <= self.cpu_limit - self.hysteresis
        ok_gpu = self.gpu_temp is None or self.gpu_temp <= self.gpu_limit - self.hysteresis
        return ok_cpu and ok_gpu

    def _log(self, event: str = "", util: Optional[float] = None):
        if self._writer is None:
            return
        now = time.time()
        with self._lock:
            self._writer.writerow([
                f"{now:.1f}", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                "" if self.cpu_temp is None else f"{self.cpu_temp:.1f}", self.cpu_source,
                "" if self.gpu_temp is None else f"{self.gpu_temp:.1f}",
                "" if util is None else f"{util:.0f}", self.level, event,
            ])
            self._fh.flush()
        if self.state_file:
            try:
                with open(self.state_file, "w", encoding="utf-8") as f:
                    json.dump({"time": now, "cpu_temp_c": self.cpu_temp, "gpu_temp_c": self.gpu_temp,
                               "level": self.level}, f)
            except Exception:
                pass

    def poll_once(self):
        self.gpu_temp = read_gpu_temp()
        if self._tick % self.cpu_poll_every == 0:
            self.cpu_temp, self.cpu_source = read_cpu_temp()
        self._tick += 1
        prev = self.level
        self.level = self._classify()
        util = read_gpu_util()
        event = f"{prev}->{self.level}" if prev != self.level else ""
        self._log(event, util)

    def _run(self):
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.poll_seconds)

    def start(self) -> "ThermalGuard":
        self.poll_once()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ThermalGuard")
        self._thread.start()
        if self.verbose:
            print(f"[ThermalGuard] başladı | CPU {self.cpu_temp} °C ({self.cpu_source}) | "
                  f"GPU {self.gpu_temp} °C | limitler CPU {self.cpu_limit} / GPU {self.gpu_limit}")
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_seconds + 20)
        self._log("stop")
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None

    # -- iş yükü tarafı ------------------------------------------------------
    def throttle(self):
        """İş yükünün adımları arasında çağrılır. Sıcaksa soğuyana kadar bekler."""
        if self.level == "hot":
            t0 = time.time()
            self.n_pauses += 1
            self._log("pause_start")
            if self.verbose:
                print(f"\n[ThermalGuard] SICAKLIK SINIRI: CPU={self.cpu_temp} GPU={self.gpu_temp} °C -> "
                      f"iş yükü duraklatıldı (devam eşiği: CPU<={self.cpu_limit - self.hysteresis}, "
                      f"GPU<={self.gpu_limit - self.hysteresis})", flush=True)
            while True:
                time.sleep(self.poll_seconds)
                if self._thread is None or not self._thread.is_alive():
                    self.poll_once()
                if self._cool_enough():
                    break
                if time.time() - t0 > self.max_pause_seconds:
                    self._log("pause_exceeded_max_continue_waiting")
                    t0 = time.time()
            dt = time.time() - t0
            self.paused_seconds += dt
            self.level = self._classify()
            self._log(f"pause_end_{dt:.0f}s")
            if self.verbose:
                print(f"[ThermalGuard] soğudu, devam ediliyor ({dt:.0f} sn bekleme).", flush=True)
        elif self.level == "warn":
            self.n_warn_sleeps += 1
            time.sleep(self.warn_sleep)

    def recommend_batch_size(self, base: int, minimum: int = 8) -> int:
        """Yalnızca batch size'ın bilimsel değişken OLMADIĞI iş yükleri için."""
        if self.level == "hot":
            return max(minimum, base // 4)
        if self.level == "warn":
            return max(minimum, base // 2)
        return base

    def recommend_workers(self, base: int) -> int:
        if self.level in ("warn", "hot"):
            return max(0, base // 2)
        return base

    def summary(self) -> dict:
        return {"n_pauses": self.n_pauses, "paused_seconds": round(self.paused_seconds, 1),
                "n_warn_sleeps": self.n_warn_sleeps, "cpu_limit": self.cpu_limit,
                "gpu_limit": self.gpu_limit, "cpu_source": self.cpu_source}


def main():
    ap = argparse.ArgumentParser(description="Arka plan termal izleyici (CPU/GPU)")
    ap.add_argument("--monitor", action="store_true", help="Sürekli izle ve CSV'ye yaz")
    ap.add_argument("--probe", action="store_true", help="Desteklenen sensörleri raporla")
    ap.add_argument("--log", default="results/thermal_monitor.csv")
    ap.add_argument("--state-file", default=None)
    ap.add_argument("--cpu-limit", type=float, default=85.0)
    ap.add_argument("--gpu-limit", type=float, default=80.0)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=0.0, help="saniye; 0 = sınırsız")
    args = ap.parse_args()
    if args.probe or not args.monitor:
        print(json.dumps(probe_support(), indent=2, ensure_ascii=False))
        if not args.monitor:
            return
    g = ThermalGuard(cpu_limit=args.cpu_limit, gpu_limit=args.gpu_limit, poll_seconds=args.poll,
                     log_csv=args.log, state_file=args.state_file)
    g.start()
    t0 = time.time()
    try:
        while args.duration <= 0 or time.time() - t0 < args.duration:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        g.stop()


if __name__ == "__main__":
    main()
