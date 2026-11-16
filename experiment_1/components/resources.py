"""Variant-level CUDA allocator peaks, including the resident source cache."""
import time

class ResourceRecorder:
    def __init__(self, runner):
        self.runner = runner
        self.values = {}
        self.shared_peak_allocated_bytes = None
        self.started = {}

    def begin(self):
        if self.runner.device.type == "cuda":
            self.runner.torch.cuda.synchronize(self.runner.device)
            self.runner.torch.cuda.reset_peak_memory_stats(self.runner.device)

    def __call__(self, variant, event):
        if self.runner.device.type != "cuda":
            if event == "start":
                self.started[variant] = time.perf_counter()
            else:
                self.values[variant] = {"cuda_baseline_allocated_bytes": None,
                    "cuda_peak_allocated_bytes": None, "cuda_peak_extra_allocated_bytes": None,
                    "variant_wall_seconds": time.perf_counter() - self.started[variant]}
            return
        cuda, device = self.runner.torch.cuda, self.runner.device
        cuda.synchronize(device)
        if event == "start":
            if self.shared_peak_allocated_bytes is None:
                self.shared_peak_allocated_bytes = cuda.max_memory_allocated(device)
            cuda.reset_peak_memory_stats(device)
            self.values[variant] = {"cuda_baseline_allocated_bytes": cuda.memory_allocated(device)}
            self.started[variant] = time.perf_counter()
        else:
            peak = cuda.max_memory_allocated(device)
            values = self.values[variant]
            values.update(cuda_peak_allocated_bytes=peak,
                          cuda_peak_extra_allocated_bytes=max(0, peak - values["cuda_baseline_allocated_bytes"]),
                          variant_wall_seconds=time.perf_counter() - self.started[variant])
