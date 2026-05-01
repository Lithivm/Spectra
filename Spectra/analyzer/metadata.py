"""从音频文件提取元数据，映射为中文键名。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mutagen
import mutagen.mp3
import mutagen.mp4
import mutagen.flac
import mutagen.oggvorbis
import mutagen.asf
import mutagen.wave
import mutagen.monkeysaudio
import mutagen.apev2
from mutagen.id3 import ID3
from mutagen.monkeysaudio import MonkeysAudio as _Monkeysaudio
from mutagen.mp4 import MP4

# 各格式常见的帧/键名 → 中文
_MAPS: dict[str, dict[str, str]] = {
    "MP3": {
        "TIT2": "标题",
        "TPE1": "艺术家",
        "TALB": "专辑",
        "TYER": "年份",
        "TCON": "流派",
        "TRCK": "音轨",
        "TPUB": "发行商",
        "TCOM": "作曲",
        "TKEY": "调性",
        "TSSE": "编码设备",
    },
    "MPEG-4": {
        "©art": "艺术家",
        "©alb": "专辑",
        "©nam": "标题",
        "©day": "年份",
        "©cmt": "备注",
        "©gen": "流派",
        "trkn": "音轨",
        "disk": "碟片",
        "©wrt": "作曲",
        "©grp": "组名",
    },
    "OGGVORBIS": {
        "TITLE": "标题",
        "ARTIST": "艺术家",
        "ALBUM": "专辑",
        "DATE": "年份",
        "GENRE": "流派",
        "TRACKNUMBER": "音轨",
        "DISCNUMBER": "碟片",
        "COMPOSER": "作曲",
        "PERFORMER": "表演者",
        "DESCRIPTION": "描述",
        "COMMENT": "备注",
    },
    "FLAC": {
        "TITLE": "标题",
        "ARTIST": "艺术家",
        "ALBUM": "专辑",
        "DATE": "年份",
        "GENRE": "流派",
        "TRACKNUMBER": "音轨",
        "DISCNUMBER": "碟片",
        "COMPOSER": "作曲",
        "PERFORMER": "表演者",
        "COPYRIGHT": "版权",
    },
    "ASF": {
        "Title": "标题",
        "Author": "艺术家",
        "AlbumTitle": "专辑",
        "Description": "描述",
        "Date": "年份",
        "Genre": "流派",
        "TrackNumber": "音轨",
        "Composer": "作曲",
        "WM/AlbumSubTitle": "专辑副标题",
    },
    "APE": {
        "Title": "标题",
        "Artist": "艺术家",
        "Album": "专辑",
        "Date": "年份",
        "Genre": "流派",
        "Tracknumber": "音轨",
        "Composer": "作曲",
        "Publisher": "发行商",
    },
    "WAVE": {
        "TIT2": "标题",
        "TPE1": "艺术家",
        "TALB": "专辑",
        "TYER": "年份",
        "TCON": "流派",
        "TRCK": "音轨",
        "TCOM": "作曲",
    },
}

# 各格式的 mime types
_MIME_TYPES: dict[str, str] = {
    "MP3": "audio/mpeg",
    "MPEG-4": "audio/mp4",
    "OGGVORBIS": "audio/ogg",
    "FLAC": "audio/flac",
    "ASF": "application/x-ms-wma",
    "WAVE": "audio/wav",
    "APE": "audio/x-ape",
}


def get_metadata(filepath: str | Path) -> dict[str, Any]:
    """打开音频文件并返回格式化的元数据。

    返回的字典包含:
        - format: 音频格式
        - bitrate: 比特率 (bps)
        - duration: 时长 (秒)
        - channels: 声道数
        - sample_rate: 采样率
        - title, artist, album, year, genre, track 等
    """
    filepath = Path(filepath)
    audio = mutagen.File(filepath)

    if audio is None:
        return {}

    # 判断格式
    raw = audio.__class__.__name__
    fmt = raw.rpartition("Decoder")[0] if "Decoder" in raw else raw
    if audio.__class__.__name__.startswith("MP3"):
        fmt = "MP3"
    elif audio.__class__.__name__.startswith("MP4"):
        fmt = "MPEG-4"
    elif audio.__class__.__name__ == "FLAC":
        fmt = "FLAC"
    elif audio.__class__.__name__ == "OGGVORBIS":
        fmt = "OGGVORBIS"
    elif audio.__class__.__name__ == "ASF":
        fmt = "ASF"
    elif audio.__class__.__name__ == "WAVE":
        fmt = "WAVE"
    elif audio.__class__.__name__.startswith("Monkeysaudio"):
        fmt = "APE"

    result: dict[str, Any] = {
        "filename": filepath.name,
        "filepath": filepath,
        "format": fmt,
        "mime_type": _MIME_TYPES.get(fmt),
    }

    # 通用信息
    try:
        result["bitrate"] = audio.info.bitrate
    except Exception:
        pass
    try:
        result["duration"] = audio.info.length
    except Exception:
        pass

    # 解码相关
    try:
        info = audio.info
        if hasattr(info, "channels"):
            result["channels"] = info.channels
        if hasattr(info, "sample_rate"):
            result["sample_rate"] = info.sample_rate
    except Exception:
        pass

    # 元数据映射
    mapping = _MAPS.get(fmt, {})
    if mapping and hasattr(audio, "tags"):
        for key, label in mapping.items():
            try:
                val = audio.tags.get(key)
                if val is not None:
                    # ASF/MP4 可能返回 list
                    if isinstance(val, list):
                        result[label] = ", ".join(str(v) for v in val)
                    else:
                        result[label] = str(val)
            except Exception:
                continue

    return result
