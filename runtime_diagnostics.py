"""Small, credential-free environment and fitted-device diagnostics."""
from __future__ import annotations

import importlib.metadata
import platform


def runtime_metadata() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    info["packages"] = {}
    for name in ["torch", "tabicl", "tabpfn", "numpy", "scikit-learn", "skrub", "huggingface-hub"]:
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    import torch
    info.update(cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
                mps_available=bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
    info["cuda_devices"] = [
        {"index": i, "name": torch.cuda.get_device_name(i),
         "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())]
    return info


def fitted_devices(estimator, model_name: str) -> list[str]:
    if model_name in {"ridge", "catboost"}:
        return ["cpu"]  # These constructors do not enable GPU training.
    for name in ("devices_", "device_"):
        value = getattr(estimator, name, None)
        if value is not None:
            return [str(v) for v in value] if isinstance(value, (tuple, list)) else [str(value)]
    return ["unknown"]
