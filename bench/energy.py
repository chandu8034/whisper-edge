import threading, time, pynvml

RAPL = "/sys/class/powercap/intel-rapl:0"

def _read(path):
    with open(path) as f:
        return int(f.read())

class EnergyMeter:
    def __init__(self, interval=0.05):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.dt = interval
        try:
           _read(f"{RAPL}/energy_uj")
           self.rapl_max = _read(f"{RAPL}/max_energy_range_uj")
           self.rapl = True
        except (OSError, ValueError):
           self.rapl = False
    def start(self):
        self.samples, self.on = [], True
        self.cpu0 = _read(f"{RAPL}/energy_uj") if self.rapl else 0
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
    def _loop(self):
        while self.on:
            watts = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
            self.samples.append((time.perf_counter(), watts))
            time.sleep(self.dt)
    def stop(self):
        self.on = False
        self.t.join()
        gpu_j = sum((t2 - t1) * p1 for (t1, p1), (t2, _) in zip(self.samples, self.samples[1:]))
        cpu_j = None
        if self.rapl:
            d = _read(f"{RAPL}/energy_uj") - self.cpu0
            if d < 0:
                d += self.rapl_max # counter wrapped
            cpu_j = d / 1e6
        return gpu_j, cpu_j
