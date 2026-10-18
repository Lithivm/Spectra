"""输出通道枚举 —— 只列「系统（默认设备）」+ WASAPI 共享/独占。

纯数据层：不 import sounddevice、不碰 Qt、不开任何流，所以可以脱机单测。
调用方（UI）把 ``sd.query_devices()`` / ``sd.query_hostapis()`` 的结果喂进来。

为什么不列 MME / DirectSound / WDM-KS：
``WasapiSettings`` 对非 WASAPI host API 的设备必失败（本机实测 PaErrorCode -9984），
列出来只会造成「点了就报错」。WDM-KS 是另一条免混音器路径，本版不做。

索引是机器相关的、会变（拔插设备就重排）→ 一律按 **host API 名 + 设备名** 解析，
不写死 index。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

WASAPI_HOST_API = "Windows WASAPI"

MODES = ("system", "shared", "exclusive")
MODE_TR = {"system": "系统", "shared": "共享", "exclusive": "独占"}


@dataclass(frozen=True)
class OutputChannel:
    """一条可选输出通道。``device_index`` 是本次运行期的索引，不持久化。"""

    host_api: str
    device_name: str
    mode: str                     # system | shared | exclusive
    device_index: int
    mix_sr: int                   # 设备 mix format 率：共享必用它 → 必然 SRC
    max_channels: int
    is_default: bool = False

    @property
    def key(self) -> str:
        """持久化键：host API 名 + 设备名 + 模式（不含 index）。"""
        return f"{self.host_api}|{self.device_name}|{self.mode}"

    def is_passthrough(self, source_sr: int) -> bool:
        """直通 = 按文件采样率请求且不必重采样。

        共享走 mix format 率 → 文件率不同就必然 SRC；系统条走默认设备，同理。
        独占按文件率直接请求，成功即直通。
        """
        if self.mode == "exclusive":
            return True
        return source_sr == self.mix_sr

    # ---- 文案（UI 层传 lang.t 进来的翻译函数） ----

    def status_text(self, *, full: bool = True, tr: Callable[[str], str] | None = None,
                    name: str | None = None) -> str:
        """状态栏：协议 + 设备名；宽度不够时缩到只剩协议。

        独占的「独占」标记保留在缩略形态里 —— 独占会静音其他应用，这是列表里
        最要紧的信息，纯协议表达不了。共享/系统缩略后只剩协议。
        """
        tr = tr or (lambda s: s)
        proto = tr(MODE_TR["system"]) if self.mode == "system" else tr("WASAPI")
        parts = [proto]
        if self.mode == "exclusive":
            parts.append(tr(MODE_TR["exclusive"]))
        if full:
            parts.append(name or self.device_name)
        return " · ".join(parts)

    def menu_label(self, tr: Callable[[str], str] | None = None, note: str | None = None) -> str:
        """展开列表里的一行。``note`` 是 UI 追加的附注（mix 率 / 未探测 / 回退）。"""
        tr = tr or (lambda s: s)
        base = f"{tr('系统') if self.mode == 'system' else tr('WASAPI')}"
        base += f"（{tr(MODE_TR[self.mode])}） — {self.device_name}"
        if note:
            base += f" · {note}"
        return base


def enumerate_channels(devices: Iterable[Mapping],
                       hostapis: Iterable[Mapping],
                       default_output: int,
                       *,
                       wasapi_host_api: str = WASAPI_HOST_API) -> list[OutputChannel]:
    """按「系统（默认设备）+ WASAPI（共享/独占）」列出输出通道。

    - 只列有输出能力的设备（``max_output_channels > 0``）
    - 非 WASAPI host API 的设备不列（系统那条除外，它走 sounddevice 默认设备）
    - 同一台设备在 WASAPI 下给出共享 + 独占两行
    """
    devices = list(devices)
    hostapis = list(hostapis)

    def api_name(idx: int) -> str:
        idx = int(idx)
        return hostapis[idx]["name"] if 0 <= idx < len(hostapis) else ""

    entries: list[OutputChannel] = []

    if 0 <= int(default_output) < len(devices):
        d = devices[int(default_output)]
        if d["max_output_channels"] > 0:
            entries.append(OutputChannel(
                host_api=api_name(d["hostapi"]),
                device_name=d["name"],
                mode="system",
                device_index=int(d["index"]),
                mix_sr=int(d["default_samplerate"]),
                max_channels=int(d["max_output_channels"]),
                is_default=True,
            ))

    for d in devices:
        if api_name(d["hostapi"]) != wasapi_host_api or d["max_output_channels"] <= 0:
            continue
        for mode in ("shared", "exclusive"):
            entries.append(OutputChannel(
                host_api=wasapi_host_api,
                device_name=d["name"],
                mode=mode,
                device_index=int(d["index"]),
                mix_sr=int(d["default_samplerate"]),
                max_channels=int(d["max_output_channels"]),
            ))

    return entries


def resolve_channel(entries: Iterable[OutputChannel],
                    host_api: str,
                    device_name: str,
                    mode: str,
                    *,
                    fallback_mode: str = "system") -> OutputChannel | None:
    """按名字找回上次选的通道（设备拔了/改名 → 回落到系统那条）。"""
    entries = list(entries)
    for e in entries:
        if e.host_api == host_api and e.device_name == device_name and e.mode == mode:
            return e
    for e in entries:
        if e.mode == fallback_mode:
            return e
    return None


def elide(text: str, budget: int, ellipsis: str = "…") -> str:
    """按字符预算截断（UI 用 QFontMetrics 换算预算后传进来）。"""
    if budget <= 0 or len(text) <= budget:
        return text
    keep = max(1, budget - len(ellipsis))
    return text[:keep] + ellipsis


@dataclass(frozen=True)
class ChannelCapability:
    """一次探测的结果：这台设备在这个模式下到底能不能开。

    独占能力只有真开一次流才知道（PortAudio 不报能力），所以这里存的是「实际可用」的
    通道 —— 独占失败回退到共享时，``channel`` 是共享那条，``fallback_from`` 记着原请求。
    """

    channel: OutputChannel
    ok: bool
    dtype: str                          # 'float32' | 'int32' | 'int16'
    error: str | None = None
    fallback_from: OutputChannel | None = None
    probed: bool = True                 # False = 还没探测过（UI 显示「未探测」）


@dataclass(frozen=True)
class StreamInfo:
    """本次真正跑起来的流：UI 状态栏/调试用。"""

    channel: OutputChannel
    dtype: str
    source_sr: int          # 文件采样率
    requested_sr: int       # 向设备请求的率（独占按文件率请求；共享必用 mix format 率）
    actual_sr: int
    error: str | None = None
    fallback_from: OutputChannel | None = None

    @property
    def passthrough(self) -> bool:
        """直通 = 设备实际跑的采样率就是文件采样率（没有被 SRC）。"""
        return self.actual_sr > 0 and self.actual_sr == self.source_sr


def parse_key(key: str) -> tuple[str, str, str]:
    """持久化键（host API 名 | 设备名 | 模式）→ 三元组。空键 → 空三元组。"""
    parts = str(key or "").split("|")
    if len(parts) != 3:
        return ("", "", "")
    return (parts[0], parts[1], parts[2])
