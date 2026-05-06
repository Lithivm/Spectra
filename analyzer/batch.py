"""Batch analysis — CSV export helpers."""

import csv
from pathlib import Path
from typing import Any

BATCH_COLUMNS = [
    "filename", "filepath", "format", "duration", "sample_rate",
    "channels", "bitrate",
    # tags
    "title", "artist", "album", "year", "genre", "track",
    # quality
    "peak_db", "rms",
    "integrated_lufs", "short_term_lufs", "lra_lu", "true_peak_db",
    "clipping_count", "clipping_longest_ms",
    "cutoff_hz", "upsampling_ok",
    "dynamic_range_db",
    "bit_depth", "freq_range_low", "freq_range_high",
]


def export_batch_csv(results: list[dict], dest: Path) -> None:
    with open(dest, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BATCH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def flatten_analysis(md: dict, qa: dict | None, filepath: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}
    row["filename"] = filepath.name
    row["filepath"] = str(filepath)
    row["format"] = md.get("format", "")
    row["duration"] = md.get("duration", "")
    row["sample_rate"] = md.get("sample_rate", "")
    row["channels"] = md.get("channels", "")
    row["bitrate"] = md.get("bitrate", "")
    row["title"] = md.get("标题", "")
    row["artist"] = md.get("艺术家", "")
    row["album"] = md.get("专辑", "")
    row["year"] = md.get("年份", "")
    row["genre"] = md.get("流派", "")
    row["track"] = md.get("音轨", "")

    if qa:
        row["peak_db"] = qa.get("peak_db", "")
        row["rms"] = qa.get("rms", "")
        row["bit_depth"] = qa.get("bit_depth", "")
        fr = qa.get("freq_range", (0, 0))
        row["freq_range_low"] = fr[0]
        row["freq_range_high"] = fr[1]

        clip = qa.get("clipping", {})
        row["clipping_count"] = clip.get("count", 0)
        row["clipping_longest_ms"] = clip.get("longest_ms", 0)

        ups = qa.get("upsampling", {})
        row["upsampling_ok"] = ups.get("ok", True)
        row["cutoff_hz"] = ups.get("cutoff_hz", "")

        dr = qa.get("dynamic_range", {})
        row["dynamic_range_db"] = dr.get("dr", 0)

        loud = qa.get("loudness", {})
        if loud:
            row["integrated_lufs"] = loud.get("integrated_lufs", "")
            row["short_term_lufs"] = loud.get("short_term_lufs", "")
            row["lra_lu"] = loud.get("lra_lu", "")
            row["true_peak_db"] = loud.get("true_peak_db", "")

    return row
