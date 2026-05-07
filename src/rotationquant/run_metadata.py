from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _run_text(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    """Collect versions for the packages that affect experiment reproducibility."""
    versions: dict[str, str] = {}
    for package in ["torch", "transformers", "safetensors", "datasets", "numpy", "pandas", "tqdm"]:
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[package] = "not-installed"
    return versions


def torch_runtime_status() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch_importable": False}
    status: dict[str, object] = {
        "torch_importable": True,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    if hasattr(torch, "mps"):
        status["mps_device_count"] = int(torch.mps.device_count())
    return status


def make_run_id(experiment: str, timestamp: str | None = None) -> tuple[str, str]:
    """Return a filesystem-friendly run id and its ISO timestamp."""
    if timestamp is None:
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    compact_time = timestamp.replace("-", "").replace(":", "").replace("+08:00", "").replace("T", "_")
    return f"{compact_time}_{experiment}", timestamp


def create_run_output_dir(base_output_dir: str | Path, experiment: str) -> tuple[Path, str, str]:
    """Create a timestamped output directory for one experiment invocation."""
    run_id, timestamp = make_run_id(experiment)
    output_dir = Path(base_output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, run_id, timestamp


def build_run_metadata(
    *,
    experiment: str,
    args: object,
    output_dir: Path,
    run_id: str,
    timestamp: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a compact metadata record for an experiment output directory."""
    metadata: dict[str, object] = {
        "experiment": experiment,
        "run_id": run_id,
        "timestamp": timestamp,
        "timezone": "Asia/Shanghai",
        "cwd": str(Path.cwd()),
        "output_dir": str(output_dir),
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": _run_text(["git", "status", "--short"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "torch_runtime": torch_runtime_status(),
        "args": vars(args),
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_run_metadata(metadata: dict[str, object], output_dir: Path, filename: str = "run_metadata.json") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
