#!/usr/bin/env python3
"""
One-click bootstrap for training environment.

What it does:
1) Creates a virtual environment (.venv) if missing
2) Installs Python dependencies
3) Installs PyTorch (CUDA or CPU build)
4) Downloads + extracts dataset archives (optional)
5) Verifies manifests and PyTorch/CUDA visibility

Can run in background:
    python setup_all.py --background
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER_URL = "https://23rb2p37-8000.euw.devtunnels.ms"
DEFAULT_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
DEFAULT_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


class Colors:
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"


def print_info(msg: str) -> None:
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")


def print_warn(msg: str) -> None:
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def print_header(msg: str) -> None:
    print()
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BLUE}{msg}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print()


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    pretty = " ".join(cmd)
    print_info(f"Run: {pretty}")
    subprocess.run(cmd, check=True, cwd=str(cwd or ROOT))


def get_venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv(venv_dir: Path) -> Path:
    venv_python = get_venv_python(venv_dir)
    if venv_python.exists():
        print_info(f"Virtualenv already exists: {venv_dir}")
        return venv_python

    print_info(f"Creating virtualenv: {venv_dir}")
    run_cmd([sys.executable, "-m", "venv", str(venv_dir)])
    return venv_python


def install_dependencies(
    venv_python: Path,
    requirements_path: Path,
    torch_index_url: str,
    skip_torch: bool,
) -> None:
    run_cmd([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run_cmd([str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)])

    if skip_torch:
        print_warn("Skipping explicit torch install (flag --skip-torch)")
        return

    run_cmd(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--index-url",
            torch_index_url,
            "torch",
            "torchvision",
            "torchaudio",
        ]
    )


def get_remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as response:
        size_hdr = response.headers.get("Content-Length")
        if not size_hdr:
            return None
        return int(size_hdr)


def download_with_curl_resume(url: str, output_path: Path, retries: int) -> None:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is not installed")

    resume_disabled = False
    for attempt in range(1, retries + 1):
        print_info(f"curl download attempt {attempt}/{retries}: {output_path.name}")
        cmd = [curl, "--fail", "--location", "--retry", "3", "--retry-delay", "2"]
        if not resume_disabled:
            # -C - enables resume support if partial file exists.
            cmd.extend(["--continue-at", "-"])
        cmd.extend(["--output", str(output_path), url])

        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode == 0:
            return

        # curl 33: server doesn't support byte ranges. Fallback to full download.
        if proc.returncode == 33 and not resume_disabled:
            print_warn("Server does not support resume (HTTP ranges). Restarting full download.")
            if output_path.exists():
                output_path.unlink()
            resume_disabled = True
            continue

        if attempt < retries:
            wait_s = min(20, attempt * 3)
            print_warn(f"curl attempt failed, retry in {wait_s}s")
            time.sleep(wait_s)
    raise RuntimeError(f"curl failed after {retries} attempts for {url}")


def download_with_urllib(url: str, output_path: Path, retries: int) -> None:
    for attempt in range(1, retries + 1):
        print_info(f"urllib download attempt {attempt}/{retries}: {output_path.name}")
        try:
            def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
                downloaded = block_num * block_size
                if total_size <= 0:
                    return
                percent = min(100.0, downloaded * 100.0 / total_size)
                print(f"\r  {percent:6.2f}% ({downloaded // (1024 * 1024)} MB)", end="")

            urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
            print()
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            print()
            if attempt == retries:
                raise RuntimeError(f"urllib failed after {retries} attempts for {url}") from exc
            wait_s = min(20, attempt * 3)
            print_warn(f"urllib attempt failed ({exc}), retry in {wait_s}s")
            time.sleep(wait_s)


def download_file(url: str, output_path: Path, desc: str, retries: int) -> None:
    print_info(f"Downloading {desc} from {url}")
    existing_size = output_path.stat().st_size if output_path.exists() else 0
    if existing_size > 0:
        print_warn(f"Found partial file: {output_path.name} ({existing_size // (1024 * 1024)} MB)")

    remote_size = None
    try:
        remote_size = get_remote_size(url)
    except Exception:
        # Some servers/proxies may not allow HEAD; proceed anyway.
        remote_size = None

    if remote_size is not None and output_path.exists() and output_path.stat().st_size >= remote_size:
        print_warn(f"{output_path.name} already fully downloaded, skip")
        return

    if shutil.which("curl"):
        download_with_curl_resume(url, output_path, retries=retries)
    else:
        if output_path.exists():
            print_warn("curl not found; deleting partial file (urllib has no reliable resume)")
            output_path.unlink()
        download_with_urllib(url, output_path, retries=retries)

    if remote_size is not None:
        final_size = output_path.stat().st_size
        if final_size != remote_size:
            raise RuntimeError(
                f"Downloaded size mismatch for {output_path.name}: got {final_size}, expected {remote_size}"
            )
    print_info(f"Downloaded: {output_path}")


def extract_zip(zip_path: Path, extract_to: Path, desc: str) -> None:
    print_info(f"Extracting {desc} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print_info(f"Extracted: {desc}")


def find_existing_archive(name: str, candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def resolve_archive_path(name: str, explicit_path: str | None, candidates: list[Path], default_path: Path) -> Path:
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.exists() and p.is_file():
            return p
        raise RuntimeError(f"{name} not found at explicit path: {p}")
    return find_existing_archive(name, candidates) or default_path


def extract_raw_sessions_zip(zip_path: Path, root_dir: Path, dataset_dir: Path, raw_sessions_dir: Path) -> None:
    print_info(f"Extracting raw_sessions.zip -> auto-detect destination")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if any(n.startswith("dataset/raw_sessions/") for n in names):
            target = root_dir
            reason = "archive contains dataset/raw_sessions/*"
        elif any(n.startswith("raw_sessions/") for n in names):
            target = dataset_dir
            reason = "archive contains raw_sessions/*"
        else:
            target = raw_sessions_dir
            reason = "archive has flat structure"
        print_info(f"raw_sessions.zip target: {target} ({reason})")
        zf.extractall(target)
    print_info("Extracted: raw_sessions.zip")


def normalize_manifests_location(root_dir: Path, dataset_dir: Path) -> None:
    manifests_dir = dataset_dir / "manifests"
    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"
    if train_manifest.exists() and val_manifest.exists():
        return

    candidate_dirs = [
        root_dir / "manifests",
        dataset_dir / "dataset" / "manifests",
    ]
    # Also check shallow nested paths under project root (e.g. extracted with extra top-level folder).
    candidate_dirs.extend([p for p in root_dir.glob("*/manifests") if p.is_dir()])

    for candidate in candidate_dirs:
        c_train = candidate / "train_manifest.jsonl"
        c_val = candidate / "val_manifest.jsonl"
        if c_train.exists() and c_val.exists():
            manifests_dir.mkdir(parents=True, exist_ok=True)
            print_warn(f"Normalizing manifests location: {candidate} -> {manifests_dir}")
            shutil.copy2(c_train, train_manifest)
            shutil.copy2(c_val, val_manifest)
            return


def extract_dataset_zip(zip_path: Path, root_dir: Path, dataset_dir: Path) -> None:
    print_info("Extracting dataset.zip -> auto-detect destination")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if any(n.startswith("dataset/manifests/") for n in names):
            target = root_dir
            reason = "archive contains dataset/manifests/*"
        elif any(n.startswith("manifests/") for n in names):
            target = dataset_dir
            reason = "archive contains manifests/*"
        else:
            target = root_dir
            reason = "fallback extraction to project root"
        print_info(f"dataset.zip target: {target} ({reason})")
        zf.extractall(target)
    print_info("Extracted: dataset.zip")
    normalize_manifests_location(root_dir, dataset_dir)


def ensure_dataset(
    server_url: str,
    cleanup_zips: bool,
    download_retries: int,
    no_download_archives: bool,
    dataset_zip_path: str | None,
    raw_sessions_zip_path: str | None,
) -> None:
    dataset_dir = ROOT / "dataset"
    raw_sessions_dir = dataset_dir / "raw_sessions"
    manifests_dir = dataset_dir / "manifests"
    archives_dir = dataset_dir / "archives"

    dataset_dir.mkdir(exist_ok=True)
    raw_sessions_dir.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)

    archived_dataset_zip = archives_dir / "dataset.zip"
    archived_raw_sessions_zip = archives_dir / "raw_sessions.zip"

    # Strong preference for dataset/archives to avoid accidental usage of stale root copies.
    if dataset_zip_path:
        dataset_zip = resolve_archive_path("dataset.zip", dataset_zip_path, [], archived_dataset_zip)
    elif archived_dataset_zip.exists():
        dataset_zip = archived_dataset_zip
    else:
        dataset_zip = resolve_archive_path(
            "dataset.zip",
            None,
            [
                dataset_dir / "dataset.zip",
                ROOT / "dataset.zip",
                ROOT.parent / "dataset.zip",
            ],
            archived_dataset_zip,
        )

    if raw_sessions_zip_path:
        raw_sessions_zip = resolve_archive_path("raw_sessions.zip", raw_sessions_zip_path, [], archived_raw_sessions_zip)
    elif archived_raw_sessions_zip.exists():
        raw_sessions_zip = archived_raw_sessions_zip
    else:
        raw_sessions_zip = resolve_archive_path(
            "raw_sessions.zip",
            None,
            [
                dataset_dir / "raw_sessions.zip",
                ROOT / "raw_sessions.zip",
                ROOT.parent / "raw_sessions.zip",
            ],
            archived_raw_sessions_zip,
        )

    print_info(f"Resolved dataset archive: {dataset_zip}")
    print_info(f"Resolved raw sessions archive: {raw_sessions_zip}")

    if not manifests_dir.exists():
        if dataset_zip.exists():
            print_warn(f"Using existing archive: {dataset_zip}")
        else:
            if no_download_archives:
                raise RuntimeError(f"dataset.zip not found locally. Expected one of: {archives_dir}, {dataset_dir}, {ROOT}")
            download_file(f"{server_url}/dataset.zip", dataset_zip, "dataset.zip", retries=download_retries)
        extract_dataset_zip(dataset_zip, ROOT, dataset_dir)
    else:
        print_warn("dataset/manifests already exists, skip dataset extraction")

    if not (raw_sessions_dir / "session_0001").exists():
        if raw_sessions_zip.exists():
            print_warn(f"Using existing archive: {raw_sessions_zip}")
        else:
            if no_download_archives:
                raise RuntimeError(
                    f"raw_sessions.zip not found locally. Expected one of: {archives_dir}, {dataset_dir}, {ROOT}"
                )
            download_file(
                f"{server_url}/raw_sessions.zip",
                raw_sessions_zip,
                "raw_sessions.zip",
                retries=download_retries,
            )
        extract_raw_sessions_zip(raw_sessions_zip, ROOT, dataset_dir, raw_sessions_dir)
    else:
        print_warn("dataset/raw_sessions/session_0001 already exists, skip raw sessions extraction")

    normalize_manifests_location(ROOT, dataset_dir)
    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"
    if not train_manifest.exists() or not val_manifest.exists():
        raise RuntimeError("Dataset setup incomplete: train/val manifests are missing.")

    print_info("Dataset verified: manifests are present")

    if cleanup_zips:
        for p in (dataset_zip, raw_sessions_zip):
            if p.exists():
                p.unlink()
                print_info(f"Removed archive: {p.name}")


def check_torch(venv_python: Path) -> None:
    code = textwrap.dedent(
        """
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA devices: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        """
    )
    run_cmd([str(venv_python), "-c", code])


def spawn_background() -> None:
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"setup_{time.strftime('%Y%m%d_%H%M%S')}.log"

    child_argv = [arg for arg in sys.argv[1:] if arg != "--background"]
    child_argv.append("--run-installer")
    cmd = [sys.executable, str(Path(__file__).name), *child_argv]

    with open(log_path, "w", encoding="utf-8") as log_file:
        kwargs = {
            "cwd": str(ROOT),
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            proc = subprocess.Popen(cmd, creationflags=flags, **kwargs)
        else:
            proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)

    print_header("Bootstrap started in background")
    print_info(f"PID: {proc.pid}")
    print_info(f"Log: {log_path}")
    print_info("Tail log command:")
    if os.name == "nt":
        print(f"  Get-Content -Wait {log_path}")
    else:
        print(f"  tail -f {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click setup for CS2 training")
    parser.add_argument("--background", action="store_true", help="Start installer in background and exit")
    parser.add_argument("--run-installer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--venv-dir", default=".venv", help="Virtualenv directory (default: .venv)")
    parser.add_argument(
        "--profile",
        choices=["training", "full"],
        default="training",
        help="Install profile: training (server-friendly) or full (capture + training)",
    )
    parser.add_argument(
        "--requirements",
        default=None,
        help="Optional custom requirements file path (overrides --profile)",
    )
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset download/extraction")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Dataset server base URL")
    parser.add_argument("--download-retries", type=int, default=6, help="Download retries for dataset archives")
    parser.add_argument(
        "--no-download-archives",
        action="store_true",
        help="Never download dataset archives; use only local dataset.zip/raw_sessions.zip",
    )
    parser.add_argument(
        "--dataset-zip-path",
        default=None,
        help="Explicit path to local dataset.zip (absolute or relative to project root)",
    )
    parser.add_argument(
        "--raw-sessions-zip-path",
        default=None,
        help="Explicit path to local raw_sessions.zip (absolute or relative to project root)",
    )
    parser.add_argument("--cleanup-zips", action="store_true", help="Delete downloaded zip files after extraction")
    parser.add_argument("--skip-torch", action="store_true", help="Skip explicit torch/torchvision/torchaudio install")
    parser.add_argument("--cpu-only", action="store_true", help="Install CPU-only PyTorch wheels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.background and not args.run_installer:
        spawn_background()
        return

    print_header("CS2 Training Bootstrap")

    venv_dir = ROOT / args.venv_dir
    if args.requirements:
        requirements_path = (ROOT / args.requirements).resolve()
    else:
        requirements_path = ROOT / ("requirements_train.txt" if args.profile == "training" else "requirements.txt")
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
    print_info(f"Using profile: {args.profile}")
    print_info(f"Requirements file: {requirements_path}")

    torch_index = DEFAULT_TORCH_CPU_INDEX if args.cpu_only else DEFAULT_TORCH_CUDA_INDEX
    if args.cpu_only:
        print_warn("CPU-only mode selected for PyTorch")
    else:
        print_info(f"Using CUDA PyTorch index: {torch_index}")

    venv_python = ensure_venv(venv_dir)
    install_dependencies(
        venv_python=venv_python,
        requirements_path=requirements_path,
        torch_index_url=torch_index,
        skip_torch=args.skip_torch,
    )

    if args.skip_dataset:
        print_warn("Skipping dataset setup (flag --skip-dataset)")
    else:
        ensure_dataset(
            server_url=args.server_url,
            cleanup_zips=args.cleanup_zips,
            download_retries=args.download_retries,
            no_download_archives=args.no_download_archives,
            dataset_zip_path=args.dataset_zip_path,
            raw_sessions_zip_path=args.raw_sessions_zip_path,
        )

    if not args.skip_torch:
        check_torch(venv_python)

    print_header("Setup Complete")
    if os.name == "nt":
        train_cmd = (
            ".\\.venv\\Scripts\\python -m training.train "
            "--train-manifest dataset/manifests/train_manifest.jsonl "
            "--val-manifest dataset/manifests/val_manifest.jsonl "
            "--dataset-root dataset --world-size 2"
        )
    else:
        train_cmd = (
            "./.venv/bin/python -m training.train "
            "--train-manifest dataset/manifests/train_manifest.jsonl "
            "--val-manifest dataset/manifests/val_manifest.jsonl "
            "--dataset-root dataset --world-size 2"
        )
    print("Training command:")
    print(f"  {train_cmd}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print_error(f"Command failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)
    except Exception as exc:
        print_error(str(exc))
        sys.exit(1)
