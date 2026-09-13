#!/usr/bin/env python3
"""Fail when a wheel filename advertises tags its WHEEL metadata does not."""

from __future__ import annotations

import glob
import sys
import zipfile
from pathlib import Path


def _expand(arguments: list[str]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        matches = [Path(match) for match in glob.glob(argument, recursive=True)]
        if matches:
            paths.extend(matches)
        else:
            candidate = Path(argument)
            if candidate.is_file():
                paths.append(candidate)
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ValueError("no wheel files matched")
    return unique


def _filename_tags(path: Path) -> set[str]:
    if path.suffix != ".whl":
        raise ValueError(f"not a wheel: {path}")
    try:
        _prefix, python_tag, abi_tag, platform_tag = path.name[:-4].rsplit("-", 3)
    except ValueError as exc:
        raise ValueError(f"invalid wheel filename: {path.name}") from exc
    return {
        f"{python_part}-{abi_part}-{platform_part}"
        for python_part in python_tag.split(".")
        for abi_part in abi_tag.split(".")
        for platform_part in platform_tag.split(".")
    }


def _metadata_tags(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        wheel_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(wheel_files) != 1:
            raise ValueError(
                f"expected exactly one .dist-info/WHEEL in {path.name}, "
                f"found {len(wheel_files)}"
            )
        metadata = archive.read(wheel_files[0]).decode("utf-8")
    tags = {
        line.partition(":")[2].strip()
        for line in metadata.splitlines()
        if line.startswith("Tag:") and line.partition(":")[2].strip()
    }
    if not tags:
        raise ValueError(f"no Tag entries in {path.name} WHEEL metadata")
    return tags


def validate(path: Path) -> None:
    filename_tags = _filename_tags(path)
    metadata_tags = _metadata_tags(path)
    if filename_tags != metadata_tags:
        raise ValueError(
            f"wheel tag mismatch for {path.name}: filename={sorted(filename_tags)} "
            f"metadata={sorted(metadata_tags)}"
        )
    print(f"OK {path.name}: {', '.join(sorted(metadata_tags))}")


def main(arguments: list[str]) -> int:
    try:
        wheels = _expand(arguments)
        for wheel in wheels:
            validate(wheel)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
