# Spectra — 音频分析工具架构文档

> iZotope RX-style audio analysis desktop application built on PyQt6, PyAV, and librosa.

---

## 1. 整体架构概览

### 1.1 包布局（2026-10-17 起）

```
src/spectra/{__main__.py, lang.py, backend/, frontend/, assets/}
```

- `backend/` = 原 `analyzer/`（解码 + 分析），`frontend/` = 原 `ui/`（Qt + OpenGL），import 前缀统一 `spectra.`
- 测试靠 `pyproject.toml` 的 `[tool.pytest.ini_options] pythonpath = ["src"]`；根 `main.py` 是入口薄壳（未 pip install 时把 `src` 挂上 sys.path），`pip install .` 后走 `spectra` 命令（`spectra.__main__:main`）

```
┌─────────────────────────────────────────────────────────────────┐
│                    main_window.py (entry)                       │
│  PyQt6 QMainWindow + drag-drop + central area                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Toolbar: open, play/pause, palette, mode, FFT, scale,    │  │
│  │           lang toggle, save PNG                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  WaveformWidget (aligned with spectrogram)          │  │  │
│  │  ├─────────────────────────────────────────────────────┤  │  │
│  │  │  [YAxis] [SpectrogramGLWidget] [ColorBar]           │  │  │
│  │  │          ↑ cursor overlay + wheel zoom              │  │  │
│  │  ├─────────────────────────────────────────────────────┤  │  │
│  │  │  PlaybackSlider (seek bar, aligned with spectrogram)│  │  │
│  │  ├─────────────────────────────────────────────────────┤  │  │
│  │  │  XAxis (time, round-minute labels)                  │  │  │
│  │  ├─────────────────────────────────────────────────────┤  │  │
│  │  │  MetadataPanel (right sidebar)                      │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                              │  │
│  PlaybackEngine (sounddevice OutputStream)                   │  │
│  — audio playback with slider sync                           │  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块

### 2.1 模块拆分

`src/spectra/backend/core.py` 是门面，下面八个模块（其中 `scales.py` 只是刻度换算，`convert.py` / `output_devices.py` 是播放侧的数据层）：

| 模块                       | 职责                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `src/spectra/backend/_state.py`    | FFTW wisdom 管理、STFT 缓存（OrderedDict 手动 LRU，`_MAX_STFT_CACHE = 8`，key 含 hop_length）、`_max_reduce_with_carry` |
| `src/spectra/backend/spectrum.py`  | `_SpectrumMixin` — STFT、多分辨率、相位重分配、mel、MFCC、流式渲染                            |
| `src/spectra/backend/quality.py`   | `_QualityMixin` — 削波、过采样检测、TT DR、LUFS、true peak                                |
| `src/spectra/backend/loudness.py`  | EBU R128 原生采样率实现（K 权重 + 分块流式 + polyphase FIR 真峰值），被 quality 调用            |
| `src/spectra/backend/scales.py`    | Slaney mel 刻度（Hz↔mel 分段），CPU 侧唯一实现；shader 的 mel 分支是它的镜像                    |
| `src/spectra/backend/convert.py`   | 播放前声道/格式准备：`prepare_channels`（mono 复制到设备声道数，不靠系统 upmix）、`quantize`（dtype 转换，int32 先限幅） |
| `src/spectra/backend/output_devices.py` | 输出通道数据层：纯数据、不 import sounddevice、不开设备 → 可脱机单测。`OutputChannel` / `ChannelCapability` / `StreamInfo` / `enumerate_channels` / `resolve_channel` / `parse_key` / `elide` |
| `src/spectra/backend/core.py`       | `AudioAnalyzer` 门面类，继承两个 mixin，保留 load/waveform/info                            |

外部一律走门面：`from spectra.backend.core import AudioAnalyzer`（UI 和测试都只认这个）。

### 2.2 音频加载 — `src/spectra/backend/load.py`

- 主解码器：**PyAV** (libav) — 支持 FLAC, OPUS, WAV, MP3, M4A, AAC, WMA, APE, OGG, TTA, AIFF
- PyAV 解码失败时有 ffmpeg 子进程回退（容错兜底，不保证所有格式）
- 所有格式统一输出 `(numpy.ndarray, sample_rate, source_format)` 形式，shape 为 `(channels, samples)`
- `source_format` 记录 PyAV 首帧的 format name（如 `"s16p"`、`"fltp"`），用于位深感知的削波分类；ffmpeg 回退路径返回 `None`

### 2.3 元数据解析 — `src/spectra/backend/metadata.py`

基于 **mutagen** 的多格式元数据提取，内部统一 `str → Any` 字典返回。

### 2.4 音频分析

#### AudioAnalyzer 对象模型

```python
class AudioAnalyzer(_SpectrumMixin, _QualityMixin):
    filepath: Path
    data: np.ndarray | None  # (channels, samples) float32
    sample_rate: int
    duration: float
    channels: int
    metadata: dict
    _source_format: str | None  # PyAV format name, e.g. "s16p", "fltp"
```

#### 关键算法

- **多分辨率 STFT** — 三频段重叠拼接：低频 0–320Hz (n_fft=8192)、中频 280–3200Hz (n_fft=2048)、高频 2800Hz–Nyquist (n_fft=512)，固定 hop=512；三频段**统一到最细的公共行网格**（4097 行，间距 sr/8192），各频段按最近 bin 映射、重叠区取逐行 max → `freqs` 严格单调、无映射空洞；各频段先算幅度、最后统一转 dB（`ref="peak"` 不再逐段各自归一）
- **相位重分配频谱图** — iZotope RX 风格（Auger-Flandrin, IEEE TASSP 1995）。三路 STFT：原始 `S`、窗导数 `D`（→ 频率修正 `f̂ = f_k − Im(D·conj(X))/|X|² × sr/2π`）、窗×局部时间斜坡 `C`（→ 群延迟 `t̂ = j·hop/sr + Re(C·conj(X))/|X|² / sr`）；符号和缩放已用啁啓真值校过（`Re` 是群延迟，不是 `Im`）
- **削波检测** — 多声道独立 flat-top 检测，**位深感知门限**：整数量化格式用 `1.0 − 2·LSB`（s16 → 0.99994），float/未知格式用 0.9999（旧写死的 0.999 在 16 bit 下距顶不到 2 LSB，实测会误报 160 次）；硬/软分类用 `np.ptp` 平坦度 + `2·LSB` 平坦门限，双桶统计（单样本 / 多样本），返回 `channels_affected` 和 `single_sample_count`
- **高频截止检测** — `scipy.signal.welch` PSD（nperseg=8192, noverlap=4096），噪底 P5 + 信号参考 P90（2–12kHz），三因子置信度：对比度 0.4 + 频谱斜率 0.35（dB/oct，上 1/3 频段线性回归）+ Gibbs ringing 0.25（截止点 ±10% 趋势线偏差 >3dB）
- **动态范围（TT DR）** — K 加权 300 ms 窗（无重叠）逐块响度的 P95 − P10，**未门控**，`dr_basis = "TT_DR_300ms_ungated"`；LRA 是另一个读数（3 s 块 + 门控），两者不再混为一谈
- **LUFS (EBU R128)** — `src/spectra/backend/loudness.py`：K 权重在**原生采样率**上设计（不再降到 12 kHz —— 12 kHz 的 Nyquist 只有 6 kHz，高架中心 16.8 kHz 整段被削）；分块 + cumsum，滤波器状态跳块连续、块网格按 hop 对齐，与整段直算一致（maxdiff 1e-13）；门控用绝对 −70 / 相对 −10 LU，channel gain 按 `(channels, samples)` 广播
- **True Peak** — polyphase FIR（4× 过采样）全频带，分块与整段同 taps 结果完全一致；sr < 16 kHz 返回 None
- **mel 口径：已统一为 Slaney（2026-10-17 定）** — CPU 侧唯一实现是 `src/spectra/backend/scales.py`（分段：1 kHz 以下线性、每 mel 66.67 Hz；以上对数，`logstep = ln(6.4)/27`，1 kHz 处 mel = 15），Y 轴刻线与 cursor 读数都走它（不再 import librosa）；shader 的 mel 分支是它的 GLSL 镜像；mel 滤波器组用 librosa 默认（= Slaney）→ 三者同一口径。选 Slaney 的理由：它的形状（低频等宽段 + 高频对数段）与临界带宽同构，比 HTK 单式更贴基准（相对 1 kHz 的每单位带宽倍数：2 kHz 2.00 vs Bark 1.98、10 kHz 10.0 vs 13.7；HTK 是 1.59 / 6.29），高频空间更大、利于看截止与 ringing。代价：1 kHz 处有个折点（两侧斜率差 3%）；与 torchaudio（默认 HTK）对拍 MFCC 要显式 `mel_scale='slaney'`，还要连滤波器组归一化（librosa 按带宽归一、torchaudio 默认不归一）一起固定。常量漂移由 `tests/test_scales.py` 钉死

#### 性能优化策略

- **相位重分配**：`S_sq` 和 `mag` 从 `S` 的实部/虚部一次性推导，省去重复的 `np.abs` 调用
- **多分辨率 STFT**：逐行幅度累加、最后统一转 dB；不再逐频段各自归一化
- **高频截止检测**：`scipy.signal.welch` 一次调用覆盖全文件，确定性结果（无随机片段）
- **True Peak 去重**：`analyze_quality()` 计算一次 true peak 后传入 `_measure_loudness()` 复用
- **STFT 缓存键**：`(filepath, fingerprint, mode, n_fft, hop, window, ref)` 由 `_cache_key()` 单点生成，UI 快路径用 `cached_result()` —— 调用方自己拼键跟不上（hop 是文件长度决定的）

### 2.5 配色方案 — `src/spectra/backend/palette.py`

#### 架构

- `palette.py` 是唯一的配色数据源（纯 numpy，零 Qt、零 matplotlib —— 色标是硬编码的，采自 matplotlib 原版关键点，所以打包时排掉 matplotlib 是安全的）
- 所有曲线参数集中在两个常量：
  
  ```python
  SPECTRA_CURVE = {"power": 0.5, "lo": 0.15, "span": 0.70}
  LINEAR_CURVE  = {"power": 1.0, "lo": 0.0,  "span": 1.0}
  ```
- `get_curve_params(name)` 返回对应 dict，`set_palette()` 零硬编码

#### 渲染逻辑（`spectrogram.frag`）

shader 对所有配色统一执行：

```glsl
t = pow(t_raw, u_curve_power);
t = clamp((t - u_curve_lo) / u_curve_span, 0.0, 1.0);
vec4 cA = texture(u_colormap, vec2(t, 0.5));
vec4 cB = texture(u_colormap2, vec2(t, 0.5));
fragColor = mix(cA, cB, u_lut_mix);   // LUT 交叉淡化
```

- spectra 配色：power=0.5, lo=0.15, span=0.70（底噪截断 + 提亮）
- 标准配色：power=1.0, lo=0.0, span=1.0（恒等变换，完全线性）
- **LUT 交叉淡化**：`u_colormap2` + `u_lut_mix`（0=A，1=B）。换配色时新 LUT 上传纹理 B，动画驱动 `u_lut_mix` 0→1（200ms），完成后写回 A、mix 归零；中途再换则只更新 B 内容并从当前 mix 继续（A 不动 = 淡入起点）

#### 配色列表

- **spectra**：自定义色板 + 专属曲线，默认配色
- **inferno / viridis / plasma / magma / hot / coolwarm / seismic / turbo / jet**：matplotlib 标准 LUT，线性映射，显示效果由 matplotlib 原版决定

#### LUT 构建

- `build_lut_np(palette_name)` — 返回 shape=(256, 4) uint8 RGBA LUT
- `is_spectra(palette_name)` — 判断是否使用自定义亮度曲线
- `get_curve_params(palette_name)` — 返回 `{"power", "lo", "span"}` dict
- 所有配色通过 `_build_lut_from_stops` 从硬编码色标（`_STOPS_TABLE`）线性插值，零外部依赖
- dB 范围：-120 到 0 dB
- **spectra 平底**：`_STOPS_TABLE["spectra"]` 首点 = `BG_CANVAS`（#000000 纯黑），与 styles token 同步；floor 由 `_rebuild_lut`/`set_palette` 取 LUT 首行 RGB，经 `u_floor` uniform 进 shader

#### 调参方式

只改 `palette.py` 中的 `SPECTRA_CURVE`，不涉及任何其他文件。

#### 标准配色保真度

- 标准配色 LUT 从 `_STOPS_TABLE` 中的硬编码色标线性插值生成（色标采样自 matplotlib 原版关键点）
- LUT alpha 统一为 255（shader 无 blending，alpha 通道不参与渲染，仅占位）
- 零外部依赖，打包时无需包含 matplotlib

### 2.6 渲染器 — `src/spectra/frontend/spectrogram_widget.py`

#### SpectrogramGLWidget (OpenGL)

- GPU-accelerated via `QOpenGLWidget`
- dB 矩阵上传为 `GL_R32F` 2D 纹理
- GLSL fragment shader 做 y 轴映射 + colormap LUT 查询
- Shader 文件外置：`src/spectra/frontend/shaders/spectrogram.vert` / `spectrogram.frag`
- 流式加载：texture 初始化为 `-120.0` dB（噪声底），`GL_NEAREST` 过滤，软边界过渡
- **视图状态**：`_view_t0/_view_t1`（时间窗口）、`_view_f0/_view_f1`（频率窗口），通过 shader uniform 实现 GPU 端缩放
- **光标信息**：`setMouseTracking(True)`，`mouseMoveEvent` 发射 `cursor_info(time, freq, dB, px)` 信号
- **滚轮缩放**：`wheelEvent` 以光标位置为中心缩放时间轴，Shift+滚轮缩放频率轴，双击重置；缩放/重置均经 `_animate_view_to(start)` 缓动（120ms OutCubic），每帧 `view_changed.emit()` 驱动坐标轴跟随
- **LUT 交叉淡化**：`set_palette()` 换配色时新 LUT 上传纹理 B，动画驱动 `u_lut_mix` 0→1（200ms）；完成写回 A。GL 未初始化时走立即路径（首次 upload）
- **加载浮层淡入/淡出**：`show_progress()`/`hide_progress()` 经 `_fade_overlay(target)` 驱动 QGraphicsOpacityEffect（150ms），完成后移除 effect 免常开开销
- **HiDPI 支持**：`resizeGL` 使用 `devicePixelRatio` 设置物理像素 viewport
- **LUT 缓存**：`build_lut` / `build_lut_np` 按配色名缓存结果，避免重复计算
- **亮度曲线**：`set_palette()` 调用 `get_curve_params(name)` 获取参数，通过 uniform 传入 shader（详见 2.5）
- **画布平底**：`u_floor` uniform（= LUT 首行 RGB = styles.BG_CANVAS）。流式加载的硬门限与软边界过渡都 fade 到 floor 而非纯黑，未填充区与画布背景一致

#### 频率轴模式

- `u_scale_mode` uniform：0=linear, 1=log, 2=mel, 3=bark
- mel 映射：Slaney 分段（`f < 1000 → f / (200/3)`；否则 `15 + ln(f/1000) / (ln(6.4)/27)`）——必须与 `src/spectra/backend/scales.py` 同常量（刻线、cursor 都走 `scales.mel_frac`，见 2.4 的 mel 条目）
- bark 映射：Zwicker & Fastl 心理声学模型，Newton-Raphson 4 次迭代

#### 坐标轴组件

- `_YAxisWidget` — 频率轴（左），支持 `view_f0/view_f1` 参数
- `_XAxisWidget` — 时间轴（下），支持 `view_t0/view_t1` 参数
- `_ColorBarWidget` — dB 色条（右），渐变条宽度 7px
- 三个组件 `pad_top=0, pad_bot=0`，与声谱图完全对齐

### 2.7 音频播放 — `src/spectra/frontend/playback_engine.py`

- 基于 `sounddevice.OutputStream`，输出通道可选：**系统（默认设备）/ WASAPI 共享 / WASAPI 独占**；只列有输出能力的设备，不列 MME / DirectSound / WDM-KS（「系统」那条本身就是默认设备，这台机器上属 **MME**（index 3）——所以 `WasapiSettings` 对它必报 `-9984`；各 host API 各自的默认设备不同：WASAPI 18、WDM-KS 28）
- 数据源是 `analyzer.data`（真立体声，不再是 mixdown）；单声道复制成设备声道数（`convert.prepare_channels`）
- 回调帧计数器追踪位置（无 DAC time 抖动）；播放/暂停/停止/Seek + 拖拽跟踪 (`track_position`)
- **构造时不开流**：独占能力只能真开一次流才知道（PortAudio 不报能力），所以 `__init__` 不探测；UI 启动时经 `_ProbeWorker`（后台线程）探测一次，结果按通道缓存（`capability()` / `cached_capability()`）——独占会静音其他应用，不能卡在 UI 线程
- dtype 优先级：独占 `float32 → int32 → int16`（float32 零量化），共享只 `float32`
- 采样率：独占按文件原生率直接请求（成功 = 直通）；共享必须用设备 mix format 率（必然 SRC）
- **回退不静默**：独占失败自动回退共享，`StreamInfo.fallback_from` 记下原始请求，状态栏显示「回退」
- `passthrough` 定义 = `actual_sr == source_sr`（不是 == 请求率），共享路径不会被误报成直通
- 换通道保住播放位置（位置存秒，按新采样率折帧）；`_resample()` 失败抛 RuntimeError，不再静默不重采样
- int32 先限幅再转：float64 × 2³¹ + `np.rint` + `np.clip`（不限幅则 `1.0 × 2³¹` 绕回 `-2147483648`）；24-bit 源 → int32 精确（`v·256`）
- 持久化存 `host_api|device_name|mode`（不是 index，索引机器相关、拔插会重排），找不到则回退系统默认
- `output_ready` 信号把**实际生效**的通道回报给 UI（可能 ≠ 用户点的那条，因为回退）；`available_channels()`（不叫 `channels`，那个名字被 `self.channels` 实例属性占了）
- 重采样：soxr 优先，scipy 回退

### 2.8 播放进度条 — `_PlaybackSlider`（`src/spectra/frontend/main_window.py`）

- 自定义 QWidget，位于声谱图与 X 轴之间（grid row 2, col 0–2 跨三列）
- 轨道 + 进度填充 + 可拖拽圆形滑块；绘制/鼠标均用 `pad = SIDE` 与声谱图精确对齐
- hover/拖拽时滑块半径 4→6px 缓动（`_hover_grow`，120ms，`FloatAnim`）
- 拖拽时实时跳转播放位置
- 播放时滑块自动跟随，停止时归零

### 2.9 动画系统 — `src/spectra/frontend/anim.py`

- `FloatAnim(QVariantAnimation)`：浮点缓动助手（默认 OutCubic），`start_from(v0, v1)` 重启、`updateCurrentValue(value)` → `on_tick`、`finished` → 末帧精确落地后 `on_finish`
- **复用同一动画对象时必须 `rebind(on_tick=..., on_finish=...)`**：闭包若捕获了局部变量（如新 effect、新的 start 元组），不重绑就会用旧闭包跑
- `updateCurrentValue` 内 try/except RuntimeError → `stop()`：widget 在动画中途被销毁时安全停表（PyQt6 对已删 C++ 对象的方法调用抛 RuntimeError）
- 动画对象必须挂在 widget 属性上（GC 安全），时长：缩放 120ms / LUT 淡化 200ms / 浮层 150ms / 滑块 hover 120ms / 元数据错峰 140ms

### 2.10 无边框窗口与图标 — `src/spectra/frontend/main_window.py` / `src/spectra/frontend/icons.py`

- `FramelessWindowHint` + `_RootWidget`（8px 边缘热区 → `windowHandle().startSystemResize(edges)`）+ `_TitleBarCard`（空白处拖拽移动、双击最大化/还原）
- 窗口控制按钮（—/□/×）在 toolbar 尾部；Win11 圆角经 ctypes `DwmSetWindowAttribute`（showEvent，失败静默回退）
- 快捷键：Ctrl+O 打开、Space 播放/暂停（焦点在按钮/下拉框时跳过）、Ctrl+S 保存 PNG；tooltip 带快捷键提示
- `src/spectra/frontend/icons.py`：内联 SVG 模板 + `render_icon(name, size, color, dpr=1.0)`（QSvgRenderer→QImage 物理像素栅格化→QPixmap 盖 dpr 戳→QIcon），HiDPI/缩放屏下图标不糊；dpr 取 `_icon_dpr()`（屏幕级，widget 未 show 前自身 dpr 不可靠）
- **Logo**：`_load_brand_logo()` 用 `assets/logo_mark.png`（透明 S-mark，44px×dpr），缺失时回退 `assets/logo.png` → 纯文本 "Spectra"；frozen/开发双路径查找
- **cursor label**：透明底（`background: transparent`）+ 保留边框，浮在声谱图上方不遮挡画布

---

## 3. UI 组件树

```
MainWindow (QMainWindow)
├── toolbar (_TitleBarCard, 拖拽/双击最大化)
│   ├── logo (24px SVG) + brand_label "Spectra"
│   ├── open_btn
│   ├── play_label + play_btn (SVG icon toggle)
│   ├── output_btn (QToolButton + QMenu, 输出通道展开列表, 文本型, 固定 220×32)
│   ├── palette_label + palette_combo
│   ├── mode_label + mode_combo
│   ├── fft_label + fft_combo
│   ├── yscale_label + yscale_combo
│   ├── save_btn
│   ├── lang_btn
│   └── min_btn / max_btn / close_btn (无边框窗口控制)
├── central_widget
│   ├── left
│   │   ├── wave_card (内衬 well: 4px BG_CANVAS, 无边框)
│   │   │   └── WaveformWidget
│   │   └── spec_card (内衬 well: 4px BG_CANVAS, 无边框)
│   │       └── QGridLayout
│   │           ├── filename_widget (row 0, col 0-2)
│   │           ├── YAxisWidget (row 1, col 0, width=36)
│   │           ├── SpectrogramGLWidget (row 1, col 1, stretch)
│   │           ├── _empty_hint (row 1, col 1 overlay, 空态提示, 加载后隐藏)
│   │           ├── ColorBarWidget (row 1, col 2, width=36)
│   │           ├── PlaybackSlider (row 2, col 0-2 span, height=20)
│   │           └── XAxisWidget (row 3, col 0-2, height=36)
│   └── right
│       └── MetadataPanel (width=310)
└── status_bar (zoom hint left-aligned + `_output_label` 专用栏位：协议 · 模式 · 设备名)
```

### 关键 UI 设计模式

**i18n 系统**

- `lang.t("中文", "English")` 统一翻译入口
- `on_lang_change(callback)` 注册回调，绑定方法用 `weakref.WeakMethod` 自动管理生命周期
- `toggle_lang()` 自动清理失效弱引用；返回 `unsubscribe()` 函数防泄漏

**样式系统 (`ui.styles`)**

- 全局 CSS token：`BG_BASE`, `BG_SURFACE`, `TEXT_PRI`, `ACCENT` 等
- 深色主题一致性

**safe_slot 装饰器**

- 所有 Qt signal-slot 主线程回调使用 `@safe_slot` 装饰器

**MetadataPanel 语言切换**

- 存储 `_section_labels`、`_info_rows`、`_tag_rows`、`_analysis_rows` widget 引用列表
- `_retranslate_with_data` 直接遍历引用列表，无需遍历布局树

**输出通道选择（play_btn 右侧）**

- `QToolButton(InstantPopup)` + `QMenu`；按钮**固定尺寸** `_OUTPUT_BTN_W × BTN_H`（220×32）——文案变短也不能缩，宽度跳会顶开后面的控件
- **菜单按协议分组** `addSection()`（系统声音（默认输出设备） / WASAPI（共享 / 独占）），行里只写**模式 — 设备名**（协议在表头上，不重复），系统行附注它实际走的 host API；action 按 `channel.key` 存在 `_output_actions`
- `QMenu` 不在全局样式表里 → 悬停高亮靠 `_output_menu_qss()` 自带（`QMenu::item:selected` 用 `BG_RAISED`）；按钮 hover/pressed 有底色反馈（`_output_btn_qss()`），只改描边太弱
- 当前行用文本前缀 `✓ ` 表示（stylesheet 下 `QMenu::indicator` 不画勾）；探过且 `ok=False` 的条目置灰 + 标「不可用」；当前生效通道以 `playback.info` 为准（≠ 用户点的那条时也要反映，因为回退）
- 状态栏用 `addPermanentWidget(_output_label)`，和文件信息（`_status_label`）分开；文案只写**协议 + 设备名**，装不下则缩到只剩协议（用 `QFontMetrics` 算宽 + `output_devices.elide`），`resizeEvent` 里重算
- 缩窄时「独占」标记必须留住 —— 它意味着静音其他应用，信息不能丢
- 术语翻译：后端 `status_text` / `menu_label` 只认中文键，UI 用 `_tr(zh)` 接 `lang.t()`（`_OUTPUT_TR_EN`）
- `_ProbeWorker(QThread)`：启动自动探测一次（测试里可换成同步替身）；测试不 `show()` 窗口，避开 offscreen 平台的 GL paint 抖动

---

## 4. 批量分析

### CSV 导出 (`src/spectra/backend/batch.py`)

`flatten_analysis()` 合并 metadata + quality analysis 为单行。所有 `BATCH_COLUMNS` 始终填充（含默认值），避免导出时 KeyError。

---

## 5. 文件依赖图

```
spectra.frontend.main_window
  ├── spectra.backend.core (AudioAnalyzer)
  ├── spectra.backend._state (_stft_cache, _MAX_STFT_CACHE=8, _stft_lock)
  ├── spectra.backend.load (load_audio, is_audio_file)
  ├── spectra.backend.batch (flatten_analysis, export_batch_csv)
  ├── spectra.backend.palette (PALETTE, build_lut_np, is_spectra, get_curve_params)
  ├── spectra.backend.scales (Slaney mel：hz_to_mel / mel_to_hz / mel_frac)
  ├── spectra.backend.output_devices (OutputChannel, ChannelCapability, MODE_TR, elide, parse_key, resolve_channel)
  ├── spectra.lang (t, toggle_lang, on_lang_change)
  ├── spectra.frontend.spectrogram_widget (SpectrogramGLWidget, 轴/色条)
  ├── spectra.frontend.waveform_widget (WaveformWidget)
  ├── spectra.frontend.metadata_panel (MetadataPanel)
  ├── spectra.frontend.playback_engine (PlaybackEngine)
  │     ├── spectra.backend.output_devices (enumerate_channels, resolve_channel, parse_key)
  │     └── spectra.backend.convert (prepare_channels, quantize)
  ├── spectra.frontend.progress_bar (加载浮层)
  ├── spectra.frontend.anim (FloatAnim — 动画助手)
  ├── spectra.frontend.icons (render_icon — SVG 图标)
  └── spectra.frontend.styles (color tokens)

spectra.backend.core
  ├── spectra.backend._state
  ├── spectra.backend.spectrum (_SpectrumMixin)
  ├── spectra.backend.quality (_QualityMixin)
  │     └── spectra.backend.loudness (R128 K 权重 / polyphase 真峰值)
  ├── spectra.backend.load
  └── spectra.backend.metadata
```

---

## 6. PyInstaller 打包注意事项

- **`numba` 不能加入 `excludes`**：librosa 懒加载依赖
- **`hiddenimports`**：pyloudnorm（只被 `loudness.py` 的 `k_weight_coeffs()` 在函数内引用，PyInstaller 静态分析能看到）、pyfftw.interfaces、scipy.signal、sklearn.utils._cython_blas、soxr
- **入口与 datas**：根 `main.py` 是薄壳 → `spectra.__main__:main`；spec 里 `pathex=['src']`，datas 打进 `src/spectra/assets/*.png`（→ `assets`）和 `src/spectra/frontend/shaders/*`（→ `spectra/frontend/shaders`，运行时兑底读 `_MEIPASS/spectra/frontend/shaders`）
- **`console=False`**（生产）+ 文件日志兜底
- **`upx_exclude`**：`.pyd` 和 numpy/scipy DLL 不压缩
- Shader 文件需打入 datas：`src/spectra/frontend/shaders/spectrogram.vert`、`spectrogram.frag`
- `src/spectra/frontend/icons.py` 用 `PyQt6.QtSvg`（QSvgRenderer）—— PyInstaller 自动分析该 import；图标为内联 SVG 字符串，无资源文件

---

## 7. 延迟加载策略

重量级库（librosa、pyfftw、pyloudnorm）不在模块顶层导入，而是延迟到首次使用时加载；scipy 只在 `loudness.py` 顶层（`firwin`/`lfilter` 是热路径），在 `quality.py` 里延迟：

- `src/spectra/backend/_state.py`：`import pyfftw` 移入 `_ensure_wisdom()` / `_flush_wisdom()`
- `src/spectra/backend/spectrum.py`：`import librosa` 移入各方法内部
- `src/spectra/backend/quality.py`：`import scipy.signal`（welch/medfilt）移入 `_detect_high_freq_cutoff()`；这个模块已经不再出现 librosa / pyloudnorm
- `src/spectra/backend/loudness.py`：`from pyloudnorm.iirfilter import IIRfilter` 移入 `k_weight_coeffs()`（只要系数，计算路径是本地实现）
- `src/spectra/backend/core.py`：`_ensure_librosa()` 在 `load()` 中调用，含 FutureWarning 抑制
- `src/spectra/frontend/main_window.py`：`AudioAnalyzer` 导入推迟到 `_LoadWorker.run()` / `_BatchWorker.run()`

---

## 8. 扩展点

1. **新格式支持** — 扩展 `SUPPORTED_EXTENSIONS` + PyAV

2. **新配色方案** — 在 `PALETTE` 加条目，标准配色用 `matplotlib.cm` 导出；自定义配色需在 `SPECTRA_CURVE` 或新建 curve 常量中定义曲线参数

3. **新分析指标** — 在 `_QualityMixin.analyze_quality()` 添加

4. **多语言扩展** — `lang.t()` 或 gettext

5. **TODO: metadata 中文键名改造** — `src/spectra/backend/metadata.py` 的 `_MAPS` 用中文字符串作字典键（`"标题"`、`"艺术家"` 等），`core.py` 的 `_TAG_TR` 做翻译中转。改为英文常量键名（`"title"`、`"artist"`）+ UI 层翻译，可消除 `batch.py` 中的硬编码中文键、补全 `_TAG_TR` 缺失条目（`发行商`、`调性`、`编码设备` 等）。影响范围：`metadata.py`、`core.py`、`batch.py`、`metadata_panel.py`、`test_analyzer.py`

6. **TODO: 声谱渲染亮度与配色优化** — 当前问题：
   
   - **感知均匀配色（viridis/magma/inferno 等）整体亮度低**：这些配色有 40-60% 暗区，线性映射下大部分能量集中在 -60~-30 dB 范围会显示为暗色
   - **spectra 配色已有亮度曲线**（`SPECTRA_CURVE`），标准配色使用线性映射
   - **可能的解决方案**：收窄 dB 范围（-100 或 -90）减少暗区覆盖、研究 Spek 的具体配色实现
   - 影响范围：`src/spectra/backend/palette.py`、`src/spectra/frontend/spectrogram_widget.py`、`src/spectra/frontend/shaders/spectrogram.frag`

> mel 口径已统一为 Slaney（见 2.4），原先的 TODO 7 已完成；今后改 mel 刻度只改 `src/spectra/backend/scales.py` + shader 的 `hz_to_mel`/`mel_to_hz`，两处常量由 `tests/test_scales.py` 钉住。

---

> 最后更新: 2026-10-18 (播放侧两件事：① 真立体声（传 `analyzer.data`，mono 复制到设备声道数）② WASAPI 输出通道选择 —— `src/spectra/backend/output_devices.py` 数据层 + `convert.quantize`（int32 先限幅）+ 引擎按通道流/直通/回退 + UI 展开列表和状态栏专用栏位 + QSettings 按名字持久化 + 启动后台探测一次（不在 `__init__` 开流）；版本 1.5.0)
> 上一更新: 2026-10-17 (mel 口径统一为 Slaney：新增 `src/spectra/backend/scales.py` 作 CPU 侧唯一实现，刻线/cursor 不再走 librosa，shader mel 分支改成分段镜像，`tests/test_scales.py` 钉常量)
> 上一更新: 2026-10-17 (CLAUDE.md 事实校正：模块表补 `loudness.py`、外部导入前缀改 `spectra.backend.core`、STFT 缓存写成 `_MAX_STFT_CACHE=8` 手动 LRU、§5 依赖图按新布局重写（补 lang/anim/icons/progress_bar/loudness）、§6 入口与 datas、§7 延迟加载对齐（quality 不再出现 librosa/pyloudnorm）、mel 口径 Slaney/HTK 不统一已量化并列入 TODO 7)
> 上一更新: 2026-10-17 (目录重构：`src/spectra/{backend(原 analyzer), frontend(原 ui), lang.py, __main__.py, assets/}` + import 前缀 `spectra.` + PyInstaller datas / CI PYTHONPATH / pyproject packages 跟着改；根目录清理：run_check.py 删、乱命名 PNG 残留删、`_remote_ref/` 忽略、`!spectra.spec` 反忽略、`assets/*.png` 不再被 `*.png` 误排（之前干净 clone 没有品牌 logo）)
> 上一更新: 2026-10-17 (代码审查修正落地：R128 原生采样率 loudness.py、TT DR 与 LRA 分开、位深感知削波 2LSB、真 dBFS 基准、Auger-Flandrin 重分配、multi 统一行网格、流式 carry 取组内 max、缓存键统一 `cached_result`、UI P0 崩溃路径 + 信号遮蔽 + 测试挂死（模态对话框）)
> 上一更新: 2026-10-07 (视觉升级 Round 5：画布/平底纯黑 #000000（BG_CANVAS + spectra LUT 首点同步）、主按钮金色渐变 hover/pressed(PRIMARY_HOVER_A/B)、图标 HiDPI dpr 渲染(render_icon + _icon_dpr)、窗口控制按钮描边式 SVG、品牌 logo 改用 assets/logo_mark.png(S-mark, DPR-aware)、toolbar 按钮 BG_BASE 填充 BTN_H=32)
> 上一更新: 2026-10-05 (视觉升级 Round 4：spectra 平底黑→BG_CANVAS(#1a1d1f) + `u_floor` uniform（硬/软门限 fade 到 floor 而非纯黑）、wave_card/spec_card 内衬 well(4px BG_CANVAS,无边框)、toolbar logo(声谱条 SVG)、cursor label 透明底、open_btn 显式样式不依赖全局)
> 上一更新: 2026-07-14 (UI 现代化重构 Phase 0–4：design token 统一、无边框窗口+SVG图标+快捷键、动画系统 FloatAnim（缩放缓动/LUT交叉淡化/浮层淡入淡出/滑块hover/元数据错峰）)
> 上一更新: 2026-06-16 (配色方案系统重构 + 削波/高频检测算法重写：多声道削波、位深感知、Welch PSD、多因子置信度)
> 基于文件: main.py(薄壳), src/spectra/__main__.py, src/spectra/lang.py, src/spectra/backend/{core,_state,spectrum,quality,loudness,load,metadata,palette,batch}.py, src/spectra/frontend/{main_window,spectrogram_widget,waveform_widget,metadata_panel,playback_engine,batch_dialog,progress_bar,anim,icons,styles}.py, src/spectra/frontend/shaders/spectrogram.vert, src/spectra/frontend/shaders/spectrogram.frag, spectra.spec

---

## Agent skills

### Issue tracker

Issues 与 spec 存于 GitHub Issues，用 `gh` CLI 操作。见 `docs/agents/issue-tracker.md`。

### Triage labels

五个标准角色使用默认标签：`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`。见 `docs/agents/triage-labels.md`。

### Domain docs

Single-context：根目录 `CONTEXT.md` + `docs/adr/`。见 `docs/agents/domain.md`。
