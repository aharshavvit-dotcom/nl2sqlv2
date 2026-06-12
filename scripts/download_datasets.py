from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import requests
from requests import Response
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.bird_adapter import _external_hf_datasets
from scripts.dataset_paths import (
    BIRD_FULL_DIR,
    BIRD_MINI_DEV_DIR,
    BIRD_MINI_DEV_HF_DIR,
    RAW_DATA_DIR,
    SPIDER_DIR,
    WIKISQL_DIR,
    ensure_dataset_dirs,
    expected_files_for_dataset,
    parse_dataset_list,
    resolve_bird_mini_dir,
)


WIKISQL_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
SPIDER_DEFAULT_URL = "https://drive.google.com/uc?id=1iRDVHLr4mX2aG9mNSFzN3ZC_xrTmM6A3"
FULL_BIRD_URLS: dict[str, str] = {
    "train.zip": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
    "dev.zip": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
}
DEFAULT_CONNECT_TIMEOUT = 30
DEFAULT_READ_TIMEOUT = 300
DEFAULT_RETRIES = 5
CHUNK_SIZE = 1024 * 1024


class DownloadIncompleteError(RuntimeError):
    pass


def download_file(
    url: str,
    path: Path,
    force: bool = False,
    retries: int = DEFAULT_RETRIES,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: int = DEFAULT_READ_TIMEOUT,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if force and path.exists():
        path.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        existing_size = path.stat().st_size if path.exists() else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(connect_timeout, read_timeout),
            ) as response:
                if response.status_code == 416:
                    return
                response.raise_for_status()
                mode, initial, total = _download_mode_and_total(response, existing_size)
                with path.open(mode) as fh, tqdm(
                    total=total,
                    initial=initial if total else 0,
                    unit="B",
                    unit_scale=True,
                    desc=path.name,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
                            progress.update(len(chunk))
                if total and path.stat().st_size < total:
                    raise DownloadIncompleteError(
                        f"{path.name} is incomplete: {path.stat().st_size} of {total} bytes"
                    )
                return
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(f"{path.name}: download interrupted ({exc}); retrying {attempt + 1}/{retries}...")
                continue
            raise DownloadIncompleteError(
                f"{path.name}: download interrupted after {retries} attempts. "
                "Rerun the same command to resume."
            ) from last_error


def _download_mode_and_total(response: Response, existing_size: int) -> tuple[str, int, int | None]:
    content_range = response.headers.get("content-range")
    if response.status_code == 206 and content_range:
        total_text = content_range.rsplit("/", 1)[-1]
        total = int(total_text) if total_text.isdigit() else None
        return "ab", existing_size, total
    if existing_size and response.status_code == 200:
        print("Server did not honor resume request; restarting this file.")
    content_length = response.headers.get("content-length")
    total = int(content_length) if content_length and content_length.isdigit() else None
    return "wb", 0, total


def extract_zip(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)


def extract_tar_bz2(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="r:*") as archive:
        archive.extractall(destination)


def normalize_nested_folder(source_dir: Path) -> None:
    for nested_name in ["data", "spider_data", "spider"]:
        nested = source_dir / nested_name
        if not nested.is_dir():
            continue
        for child in nested.iterdir():
            if child.name in {".DS_Store", "__MACOSX"}:
                continue
            target = source_dir / child.name
            if target.exists():
                continue
            shutil.move(str(child), str(target))


def dataset_has_expected_files(dataset_name: str, dataset_dir: Path) -> bool:
    expected = expected_files_for_dataset(dataset_name)
    if dataset_name == "spider":
        has_core = all((dataset_dir / item).exists() for item in expected)
        return has_core and ((dataset_dir / "database").exists() or (dataset_dir / "databases").exists())
    if dataset_name in {"bird-mini", "bird"}:
        return (
            (dataset_dir / "mini_dev_sqlite.json").exists()
            or (dataset_dir / "dataset_dict.json").exists()
            or (dataset_dir / "state.json").exists()
        )
    if not expected:
        return dataset_dir.exists() and any(path.name != ".gitkeep" for path in dataset_dir.iterdir())
    return all((dataset_dir / item).exists() for item in expected)


def download_wikisql(force: bool = False) -> dict[str, Any]:
    WIKISQL_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = WIKISQL_DIR / "data.tar.bz2"
    if dataset_has_expected_files("wikisql", WIKISQL_DIR) and not force:
        return {"dataset": "wikisql", "status": "ready", "notes": "existing files found"}
    if force or not dataset_has_expected_files("wikisql", WIKISQL_DIR):
        download_file(WIKISQL_URL, archive_path, force=force)
    extract_tar_bz2(archive_path, WIKISQL_DIR)
    normalize_nested_folder(WIKISQL_DIR)
    status = "ready" if dataset_has_expected_files("wikisql", WIKISQL_DIR) else "incomplete"
    return {"dataset": "wikisql", "status": status, "notes": str(WIKISQL_DIR)}


def download_spider(force: bool = False, spider_url: str | None = None) -> dict[str, Any]:
    SPIDER_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = SPIDER_DIR / "spider_data.zip"
    normalize_nested_folder(SPIDER_DIR)
    if dataset_has_expected_files("spider", SPIDER_DIR) and not force:
        return {"dataset": "spider", "status": "ready", "notes": "existing files found"}

    url = spider_url or SPIDER_DEFAULT_URL
    try:
        import gdown

        gdown.download(url, str(archive_path), quiet=False, fuzzy=True)
        if archive_path.exists():
            extract_zip(archive_path, SPIDER_DIR)
            normalize_nested_folder(SPIDER_DIR)
    except Exception as exc:
        message = (
            "Automatic Spider download failed. Download Spider from the official Yale "
            "Spider page and extract it into data/raw/spider/."
        )
        print(message)
        print(f"Reason: {exc}")
        return {"dataset": "spider", "status": "manual_required", "notes": message}

    status = "ready" if dataset_has_expected_files("spider", SPIDER_DIR) else "manual_required"
    notes = str(SPIDER_DIR) if status == "ready" else (
        "Download Spider from the official Yale Spider page and extract it into data/raw/spider/."
    )
    return {"dataset": "spider", "status": status, "notes": notes}


def download_bird_mini(force: bool = False) -> dict[str, Any]:
    BIRD_MINI_DEV_DIR.mkdir(parents=True, exist_ok=True)
    existing_dir = resolve_bird_mini_dir()
    if dataset_has_expected_files("bird-mini", existing_dir) and not force:
        return {"dataset": "bird-mini", "status": "ready", "notes": f"existing files found in {existing_dir}"}
    BIRD_MINI_DEV_HF_DIR.mkdir(parents=True, exist_ok=True)
    with _external_hf_datasets() as hf_datasets:
        dataset = hf_datasets.load_dataset("birdsql/bird_mini_dev")
        dataset.save_to_disk(str(BIRD_MINI_DEV_HF_DIR))
    status = "ready" if dataset_has_expected_files("bird-mini", BIRD_MINI_DEV_HF_DIR) else "incomplete"
    return {"dataset": "bird-mini", "status": status, "notes": str(BIRD_MINI_DEV_HF_DIR)}


def download_bird_full(
    force: bool = False,
    include_full_bird: bool = False,
    retries: int = DEFAULT_RETRIES,
    read_timeout: int = DEFAULT_READ_TIMEOUT,
) -> dict[str, Any]:
    if not include_full_bird:
        return {
            "dataset": "bird-full",
            "status": "skipped",
            "notes": "Full BIRD requires --include-full-bird.",
        }
    print("Full BIRD is large. Continue only if you have enough disk space.")
    BIRD_FULL_DIR.mkdir(parents=True, exist_ok=True)
    if not FULL_BIRD_URLS:
        return {
            "dataset": "bird-full",
            "status": "manual_required",
            "notes": "Configure FULL_BIRD_URLS in scripts/download_datasets.py before downloading full BIRD.",
        }
    for name, url in FULL_BIRD_URLS.items():
        output = BIRD_FULL_DIR / name
        try:
            download_file(url, output, force=force, retries=retries, read_timeout=read_timeout)
            if output.suffix == ".zip":
                extract_zip(output, BIRD_FULL_DIR)
            elif output.suffixes[-2:] == [".tar", ".bz2"] or output.suffix in {".tgz", ".tar"}:
                extract_tar_bz2(output, BIRD_FULL_DIR)
        except (DownloadIncompleteError, zipfile.BadZipFile, tarfile.TarError) as exc:
            return {
                "dataset": "bird-full",
                "status": "incomplete",
                "notes": f"{exc} Partial file kept at {output}. Rerun the same command to resume.",
            }
    return {"dataset": "bird-full", "status": "downloaded", "notes": str(BIRD_FULL_DIR)}


def download_selected(
    datasets: list[str],
    force: bool = False,
    include_full_bird: bool = False,
    spider_url: str | None = None,
    retries: int = DEFAULT_RETRIES,
    read_timeout: int = DEFAULT_READ_TIMEOUT,
) -> list[dict[str, Any]]:
    ensure_dataset_dirs()
    results: list[dict[str, Any]] = []
    for dataset_name in datasets:
        if dataset_name == "wikisql":
            results.append(download_wikisql(force=force))
        elif dataset_name == "spider":
            results.append(download_spider(force=force, spider_url=spider_url))
        elif dataset_name in {"bird", "bird-mini", "bird-mini-dev"}:
            results.append(download_bird_mini(force=force))
        elif dataset_name == "bird-full":
            results.append(
                download_bird_full(
                    force=force,
                    include_full_bird=include_full_bird,
                    retries=retries,
                    read_timeout=read_timeout,
                )
            )
        else:
            results.append({"dataset": dataset_name, "status": "unknown", "notes": "unknown dataset"})
    return results


def print_status_summary(results: list[dict[str, Any]]) -> None:
    print("\nDataset status summary")
    for result in results:
        print(f"- {result['dataset']}: {result['status']} ({result['notes']})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, help="Comma-separated: wikisql,spider,bird-mini,bird-full")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-full-bird", action="store_true")
    parser.add_argument("--spider-url", default=None)
    parser.add_argument("--output-root", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--read-timeout", type=int, default=DEFAULT_READ_TIMEOUT)
    args = parser.parse_args()
    if args.output_root != RAW_DATA_DIR:
        print("Custom --output-root is accepted for compatibility; default project paths are still used by adapters.")
    results = download_selected(
        parse_dataset_list(args.datasets),
        force=args.force,
        include_full_bird=args.include_full_bird,
        spider_url=args.spider_url,
        retries=args.retries,
        read_timeout=args.read_timeout,
    )
    print_status_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
