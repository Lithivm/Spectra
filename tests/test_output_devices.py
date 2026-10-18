"""输出通道枚举层测试 —— 喂假设备表，不 import sounddevice、不开设备。

假表照本机形状摆：MME 默认输出（索引会变）、DirectSound 同设备、WASAPI 段、
以及一个只有输入能力的设备。
"""

from spectra.backend.output_devices import enumerate_channels, resolve_channel, elide


DEVICES = [
    {"index": 0, "name": "Microsoft 映射器 - Output", "hostapi": 0,
     "max_output_channels": 2, "max_input_channels": 0, "default_samplerate": 44100},
    {"index": 1, "name": "Speakers (Realtek)", "hostapi": 0,
     "max_output_channels": 2, "max_input_channels": 2, "default_samplerate": 44100},
    {"index": 2, "name": "Mic (Realtek)", "hostapi": 0,
     "max_output_channels": 0, "max_input_channels": 1, "default_samplerate": 44100},
    {"index": 3, "name": "DAC", "hostapi": 1,
     "max_output_channels": 2, "max_input_channels": 0, "default_samplerate": 44100},
    {"index": 4, "name": "DAC", "hostapi": 2,
     "max_output_channels": 2, "max_input_channels": 0, "default_samplerate": 88200},
    {"index": 5, "name": "Realtek Digital Output", "hostapi": 2,
     "max_output_channels": 2, "max_input_channels": 0, "default_samplerate": 48000},
]
HOSTAPIS = [
    {"name": "MME", "default_output_device": 1},
    {"name": "Windows DirectSound", "default_output_device": 3},
    {"name": "Windows WASAPI", "default_output_device": 4},
]
DEFAULT_OUTPUT = 1


def _entries():
    return enumerate_channels(DEVICES, HOSTAPIS, DEFAULT_OUTPUT)


class TestEnumeration:
    def test_system_row_first(self):
        entries = _entries()
        assert entries[0].mode == "system"
        assert entries[0].device_name == "Speakers (Realtek)"
        assert entries[0].is_default is True

    def test_only_output_capable_devices(self):
        names = {e.device_name for e in _entries()}
        assert "Mic (Realtek)" not in names

    def test_non_wasapi_rows_not_listed(self):
        # MME / DirectSound 条目不列：WasapiSettings 对它们必失败（-9984）
        apis = {e.host_api for e in _entries() if e.mode != "system"}
        assert apis == {"Windows WASAPI"}
        modes = {e.mode for e in _entries() if e.mode != "system"}
        assert modes == {"shared", "exclusive"}

    def test_shared_and_exclusive_pair_per_device(self):
        wasapi = [e for e in _entries() if e.mode != "system"]
        assert [(e.device_name, e.mode) for e in wasapi] == [
            ("DAC", "shared"), ("DAC", "exclusive"),
            ("Realtek Digital Output", "shared"), ("Realtek Digital Output", "exclusive"),
        ]

    def test_mix_format_rate_kept_for_shared(self):
        shared = [e for e in _entries() if e.mode == "shared"]
        assert shared[0].mix_sr == 88200          # 本机 DAC 的 WASAPI mix format 率
        assert shared[1].mix_sr == 48000

    def test_indexes_come_from_the_table_not_hardcoded(self):
        """同一设备换个 index 也要能解析出来（拔插会重排）。"""
        moved = [dict(d) for d in DEVICES]
        moved[4]["index"] = 17
        entries = enumerate_channels(moved, HOSTAPIS, DEFAULT_OUTPUT)
        dac = [e for e in entries if e.device_name == "DAC"]
        assert all(e.device_index == 17 for e in dac)


class TestResolve:
    def test_roundtrip_by_name(self):
        entries = _entries()
        target = [e for e in entries if e.mode == "exclusive"][0]
        got = resolve_channel(entries, "Windows WASAPI", "DAC", "exclusive")
        assert got == target
        assert got.key == "Windows WASAPI|DAC|exclusive"

    def test_falls_back_to_system_when_device_gone(self):
        got = resolve_channel(_entries(), "Windows WASAPI", "DAC 已改名", "exclusive")
        assert got is not None and got.mode == "system"

    def test_returns_none_when_no_system_row(self):
        assert resolve_channel([e for e in _entries() if e.mode != "system"],
                               "Windows WASAPI", "没了", "shared") is None


class TestPassthrough:
    def test_exclusive_requests_source_rate(self):
        e = resolve_channel(_entries(), "Windows WASAPI", "DAC", "exclusive")
        assert e.is_passthrough(44100) is True

    def test_shared_must_resample_unless_rates_match(self):
        e = resolve_channel(_entries(), "Windows WASAPI", "DAC", "shared")
        assert e.is_passthrough(44100) is False
        assert e.is_passthrough(88200) is True


class TestText:
    def test_status_full_has_device_name(self):
        e = resolve_channel(_entries(), "Windows WASAPI", "DAC", "exclusive")
        assert e.status_text() == "WASAPI · 独占 · DAC"

    def test_status_narrow_keeps_exclusive_marker(self):
        e = resolve_channel(_entries(), "Windows WASAPI", "DAC", "exclusive")
        assert e.status_text(full=False) == "WASAPI · 独占"

    def test_status_narrow_shared_is_protocol_only(self):
        e = resolve_channel(_entries(), "Windows WASAPI", "DAC", "shared")
        assert e.status_text(full=False) == "WASAPI"

    def test_status_system_row(self):
        e = _entries()[0]
        assert e.status_text() == "系统 · Speakers (Realtek)"
        assert e.status_text(full=False) == "系统"

    def test_status_uses_translated_names(self):
        tr = {"系统": "System", "独占": "Exclusive", "WASAPI": "WASAPI"}.get
        e = resolve_channel(_entries(), "Windows WASAPI", "DAC", "exclusive")
        assert e.status_text(full=False, tr=tr) == "WASAPI · Exclusive"
        assert _entries()[0].status_text(full=False, tr=tr) == "System"

    def test_elide_device_name(self):
        e = resolve_channel(_entries(), "Windows WASAPI", "DAC", "shared")
        assert e.status_text(name=elide("Realtek Digital Output", 12)) == "WASAPI · Realtek Dig…"

    def test_menu_label_shape(self):
        e = resolve_channel(_entries(), "Windows WASAPI", "DAC", "exclusive")
        # 协议在分组表头上，行里不重复 —— 行只写模式 + 设备名
        assert e.menu_label() == "独占 — DAC"
        assert e.menu_label(note="不可用") == "独占 — DAC · 不可用"
        assert e.menu_label(name=elide("Realtek Digital Output", 12)) == "独占 — Realtek Dig…"
