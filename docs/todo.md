# TODO — 播放改立体声 + WASAPI 输出通道选择

明天开工。下面是已经确认的事实与规划；文末是开工前还需要你点头的几条。

---

## 1. 现状（实测过，不是猜测）

- **播放实际是单声道混缩**：`src/spectra/frontend/main_window.py:1154` 传的是 `analyzer.waveform`（= `src/spectra/backend/core.py:98` 的 `np.mean(data, axis=0)`）。所以立体声文件的声像与左右差异全丢，不相关信号求和 ≈ −3 dB，反相内容相消。测试覆盖不到（`tests/test_main_window.py` 里用的是 `analyzer.data`）。
- **独占现在不可能开**：`_probe_wasapi()` 取的是默认输出设备（本机 index 3 = **MME** host API），`WasapiSettings` 对它一律报 `PaErrorCode -9984`。
- 同一块 DAC 在 **Windows WASAPI** host API 下是 index 18（WASAPI mix format = 88200）。实测（本机 YULONG Audio DSD DAC）：

  | 路径 | 结果 |
  |---|---|
  | WASAPI dev18 `exclusive=True` int32 @44100 / 88200 / 96000 | OK，`actual_sr` == 请求率 → 设备被切到文件原生率，**无 SRC（真直通）** |
  | WASAPI dev18 `exclusive=True` float32 @44100 | OK（该 DAC 独占支持 float32 → 连量化都没有） |
  | WASAPI dev18 `exclusive=True` int16 @44100 | OK |
  | WASAPI dev18 **共享** int32 @44100 | FAIL `Invalid sample rate` → 共享必须用 mix format 率（本机 88200）→ 共享必然 SRC |
  | WDM-KS dev28（同设备，不传 extra_settings） | OK @44100 int32/float32（另一条免混音器路径，先不做） |

- **int32 溢出实测**：`1.0 × 2³¹` 绕回成 `-2147483648`（`1.2` 同样绕回）→ int32 路径必须先饱和限幅，且用 float64 乘 `INT32_MAX` 再 cast。
- **24-bit 源 → int32 是 exact**：`v/2²³ × 2³¹ = v·256`（float64 下精确），目标格式为 int32 时不需要 dither；只有硬件把独占格式定在 24/16-bit 时才有截断问题。
- 索引号是机器相关的、会变 → 代码必须按 **host API 名 + 设备名** 解析，不能写死 index。

---

## 2. 已定的设计

### 2.1 立体声
- `main_window.py` 播放改传 `analyzer.data`（`(channels, samples)`），波形显示仍用 mixdown 不变。
- 单声道文件复制成 2 声道（不依赖 Windows 引擎 upmix）。

### 2.2 UI：播放按钮右边的点击展开列表
列出读出来的输出通道，每行一条：

```
系统（默认设备）
WASAPI（共享） — YULONG Audio DSD DAC
WASAPI（独占） — YULONG Audio DSD DAC
WASAPI（共享） — Realtek Digital Output
WASAPI（独占） — Realtek Digital Output
…
```

- 只列 **有输出能力** 的设备；`系统` 那条走 sounddevice 默认设备。
- 是否把 MME / DirectSound 条目也列出来：**建议不列**——`WasapiSettings` 对非 WASAPI host API 必失败，列出来只会造成「点了就报错」。
- 当前生效的通道要在列表里标记出来（勾选/高亮），并在状态栏显示实际结果：设备名 · 实际采样率 · 格式(int32/float32) · 是否直通。
- 播放中切换 → 先关流再开流；切换后保留播放位置（复用 `track_position`）。
- 用 `render_icon` + `styles` token，文案走 `lang.t()`。
- **不做音量提示 / 软音量 / headroom**（已定：会用这个开关的人有独立 DAC）。独占绕过混音器、其他应用被静音这件事只在列表里写一句 `(独占)` 就够了。

### 2.3 直通判定
- 独占：按文件采样率直接请求；成功 = 直通（不重采样）。失败才回退。
- 共享：必须用设备的 mix format 率 → 必然重采样（这条要在状态栏如实显示「SRC: 48k→88.2k」）。

### 2.4 测试要加
- 立体声不被 mean（`channels == 2`）
- mono → 2ch 复制
- int32 限幅（1.0 / 1.2 / −1.2 不得绕回 INT_MIN）
- float32 ← int24 往返 exact（24-bit 源经 int32 目标不丢位）
- 独占失败的回退路径（mock）
- 枚举层只出「系统 + WASAPI」

---

## 3. 开工前还需要你确认

1. **独占失败怎么办**：自动回退到共享 + 状态栏写明「已回退，非直通」，还是直接报错停止？我倾向自动回退。
2. **独占 dtype**：优先 float32（零量化，本机可用）还是强制 int32（硬件位精确）？我倾向 float32 优先、失败降级 int32。
3. **MME / DirectSound 条目**列不列（建议不列）。
4. **通道选择要不要持久化**（QSettings / 配置文件，下次启动记住）——默认每次启动选「系统」。
5. **版本号**：未推送的 6 个 commit 是 mel + 审查修正，按 tag 进一位 = **v1.4**（`pyproject.toml` 的 `0.2.0` 同步成 `1.4.0`）。立体声 + WASAPI 是并进 v1.4 还是单独发 v1.5？我倾向单独发 v1.5，先把 v1.4 发掉。

---

## 4. 建议的提交拆分

1. `feat: 播放改真立体声（传 analyzer.data）+ mono 复制 2ch` + 测试
2. `feat: WASAPI 输出通道选择（共享/独占）—— 按 host API 名枚举 + 直通判定 + int32 限幅 + 回退` + 测试
3. `feat: 播放按钮右侧输出通道展开列表 + 状态栏实际通道显示`（UI）
4. `docs: CLAUDE.md §2.7 播放段按新实现改写`

CI 跑 `python -m pytest tests/`（`pythonpath=["src"]`）+ ruff（`--select=E,F,W --ignore=E501,E402,F405,F403`）。
