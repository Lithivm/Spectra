# Spectra — 音频分析工具架构文档

> iZotope RX-style audio analysis desktop application built on PyQt6, PyAV, and librosa.

---

## 1. 整体架构概览

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

`analyzer/core.py` 已拆分为四个模块：

| 模块                     | 职责                                                                                |
| ---------------------- | --------------------------------------------------------------------------------- |
| `analyzer/_state.py`   | FFTW wisdom 管理、STFT 缓存 (LRU maxsize=8, key 含 hop_length)、`_max_reduce_with_carry` |
| `analyzer/spectrum.py` | `_SpectrumMixin` — STFT、多分辨率、相位重分配、mel、MFCC、流式渲染                                  |
| `analyzer/quality.py`  | `_QualityMixin` — 削波、过采样检测、DR、LUFS、true peak                                      |
| `analyzer/core.py`     | `AudioAnalyzer` 门面类，继承两个 mixin，保留 load/waveform/info                              |

所有外部导入 `from analyzer.core import AudioAnalyzer` 无需改动。

### 2.2 音频加载 — `analyzer/load.py`

- 主解码器：**PyAV** (libav) — 支持 FLAC, OPUS, WAV, MP3, M4A, AAC, WMA, APE, OGG, TTA, AIFF
- PyAV 解码失败时有 ffmpeg 子进程回退（容错兜底，不保证所有格式）
- 所有格式统一输出 `(numpy.ndarray, sample_rate, source_format)` 形式，shape 为 `(channels, samples)`
- `source_format` 记录 PyAV 首帧的 format name（如 `"s16p"`、`"fltp"`），用于位深感知的削波分类；ffmpeg 回退路径返回 `None`

### 2.3 元数据解析 — `analyzer/metadata.py`

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

- **多分辨率 STFT** — 三频段重叠拼接：低频 0–320Hz (n_fft=8192)、中频 280–3200Hz (n_fft=2048)、高频 2800Hz–Nyquist (n_fft=512)，固定 hop=512
- **相位重分配频谱图** — iZotope RX 风格（Auger-Flandrin, IEEE TASSP 1995）。三路 STFT：原始 `S`、时间导数 `S_t`、频率导数 `S_f`，通过 `ω_corr` 和 `τ_corr` 计算瞬时频率和群延迟修正量
- **削波检测** — 多声道独立 flat-top 检测（阈值 0.999），双桶统计（单样本 / 多样本），硬/软分类用 `np.ptp` 平坦度 + 位深感知阈值（`source_format` → 量化步长），返回 `channels_affected` 和 `single_sample_count`
- **高频截止检测** — `scipy.signal.welch` PSD（nperseg=8192, noverlap=4096），噪底 P5 + 信号参考 P90（2–12kHz），三因子置信度：对比度 0.4 + 频谱斜率 0.35（dB/oct，上 1/3 频段线性回归）+ Gibbs ringing 0.25（截止点 ±10% 趋势线偏差 >3dB）
- **动态范围** — P95-P10 帧 RMS 差值（TT DR Meter 标准）
- **LUFS (EBU R128)** — `pyloudnorm`，降采样保护，LRA 精确插值

#### 性能优化策略

- **相位重分配**：`S_sq` 和 `mag` 从 `S` 的实部/虚部一次性推导，省去重复的 `np.abs` 调用
- **多分辨率 STFT 频率去重**：用 `np.diff` + boolean mask 向量化替代 Python 逐元素循环
- **高频截止检测**：`scipy.signal.welch` 一次调用覆盖全文件，确定性结果（无随机片段）
- **True Peak 去重**：`analyze_quality()` 计算一次 true peak 后传入 `_measure_loudness()` 复用

### 2.5 配色方案 — `analyzer/palette.py`

#### 架构

- `palette.py` 是唯一的配色数据源（零 Qt 依赖，numpy + matplotlib）
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

#### 调参方式

只改 `palette.py` 中的 `SPECTRA_CURVE`，不涉及任何其他文件。

#### 标准配色保真度

- 标准配色 LUT 从 `_STOPS_TABLE` 中的硬编码色标线性插值生成（色标采样自 matplotlib 原版关键点）
- LUT alpha 统一为 255（shader 无 blending，alpha 通道不参与渲染，仅占位）
- 零外部依赖，打包时无需包含 matplotlib

### 2.6 渲染器 — `ui/spectrogram_widget.py`

#### SpectrogramGLWidget (OpenGL)

- GPU-accelerated via `QOpenGLWidget`
- dB 矩阵上传为 `GL_R32F` 2D 纹理
- GLSL fragment shader 做 y 轴映射 + colormap LUT 查询
- Shader 文件外置：`ui/shaders/spectrogram.vert` / `spectrogram.frag`
- 流式加载：texture 初始化为 `-120.0` dB（噪声底），`GL_NEAREST` 过滤，软边界过渡
- **视图状态**：`_view_t0/_view_t1`（时间窗口）、`_view_f0/_view_f1`（频率窗口），通过 shader uniform 实现 GPU 端缩放
- **光标信息**：`setMouseTracking(True)`，`mouseMoveEvent` 发射 `cursor_info(time, freq, dB, px)` 信号
- **滚轮缩放**：`wheelEvent` 以光标位置为中心缩放时间轴，Shift+滚轮缩放频率轴，双击重置；缩放/重置均经 `_animate_view_to(start)` 缓动（120ms OutCubic），每帧 `view_changed.emit()` 驱动坐标轴跟随
- **LUT 交叉淡化**：`set_palette()` 换配色时新 LUT 上传纹理 B，动画驱动 `u_lut_mix` 0→1（200ms）；完成写回 A。GL 未初始化时走立即路径（首次 upload）
- **加载浮层淡入/淡出**：`show_progress()`/`hide_progress()` 经 `_fade_overlay(target)` 驱动 QGraphicsOpacityEffect（150ms），完成后移除 effect 免常开开销
- **HiDPI 支持**：`resizeGL` 使用 `devicePixelRatio` 设置物理像素 viewport
- **LUT 缓存**：`build_lut` / `build_lut_np` 按配色名缓存结果，避免重复计算
- **亮度曲线**：`set_palette()` 调用 `get_curve_params(name)` 获取参数，通过 uniform 传入 shader（详见 2.5）

#### 频率轴模式

- `u_scale_mode` uniform：0=linear, 1=log, 2=mel, 3=bark
- mel 映射：`mel = 2595 * log10(1 + f/700)`
- bark 映射：Zwicker & Fastl 心理声学模型，Newton-Raphson 4 次迭代

#### 坐标轴组件

- `_YAxisWidget` — 频率轴（左），支持 `view_f0/view_f1` 参数
- `_XAxisWidget` — 时间轴（下），支持 `view_t0/view_t1` 参数
- `_ColorBarWidget` — dB 色条（右），渐变条宽度 7px
- 三个组件 `pad_top=0, pad_bot=0`，与声谱图完全对齐

### 2.7 音频播放 — `ui/playback_engine.py`

- 基于 `sounddevice.OutputStream`，WASAPI 共享模式（低延迟 ~10ms）
- 回调帧计数器追踪位置（无 DAC time 抖动）
- 播放/暂停/停止/Seek + 拖拽跟踪 (`track_position`)
- 启动时探测 WASAPI 设备默认采样率，`load()` 时自动重采样（soxr 优先，scipy 回退）

### 2.8 播放进度条 — `_PlaybackSlider`（`ui/main_window.py`）

- 自定义 QWidget，位于声谱图与 X 轴之间（grid row 2, col 0–2 跨三列）
- 轨道 + 进度填充 + 可拖拽圆形滑块；绘制/鼠标均用 `pad = SIDE` 与声谱图精确对齐
- hover/拖拽时滑块半径 4→6px 缓动（`_hover_grow`，120ms，`FloatAnim`）
- 拖拽时实时跳转播放位置
- 播放时滑块自动跟随，停止时归零

### 2.9 动画系统 — `ui/anim.py`

- `FloatAnim(QVariantAnimation)`：浮点缓动助手（默认 OutCubic），`start_from(v0, v1)` 重启、`updateCurrentValue(value)` → `on_tick`、`finished` → 末帧精确落地后 `on_finish`
- **复用同一动画对象时必须 `rebind(on_tick=..., on_finish=...)`**：闭包若捕获了局部变量（如新 effect、新的 start 元组），不重绑就会用旧闭包跑
- `updateCurrentValue` 内 try/except RuntimeError → `stop()`：widget 在动画中途被销毁时安全停表（PyQt6 对已删 C++ 对象的方法调用抛 RuntimeError）
- 动画对象必须挂在 widget 属性上（GC 安全），时长：缩放 120ms / LUT 淡化 200ms / 浮层 150ms / 滑块 120ms

### 2.10 无边框窗口与图标 — `ui/main_window.py` / `ui/icons.py`

- `FramelessWindowHint` + `_RootWidget`（8px 边缘热区 → `windowHandle().startSystemResize(edges)`）+ `_TitleBarCard`（空白处拖拽移动、双击最大化/还原）
- 窗口控制按钮（—/□/×）在 toolbar 尾部；Win11 圆角经 ctypes `DwmSetWindowAttribute`（showEvent，失败静默回退）
- 快捷键：Ctrl+O 打开、Space 播放/暂停（焦点在按钮/下拉框时跳过）、Ctrl+S 保存 PNG；tooltip 带快捷键提示
- `ui/icons.py`：内联 SVG 模板 + `render_icon(name, size, color)`（QSvgRenderer→QPixmap→QIcon），零资源文件依赖

---

## 3. UI 组件树

```
MainWindow (QMainWindow)
├── toolbar (_TitleBarCard, 拖拽/双击最大化)
│   ├── brand_label "Spectra"
│   ├── open_btn
│   ├── play_label + play_btn (SVG icon toggle)
│   ├── palette_label + palette_combo
│   ├── mode_label + mode_combo
│   ├── fft_label + fft_combo
│   ├── yscale_label + yscale_combo
│   ├── save_btn
│   ├── lang_btn
│   └── min_btn / max_btn / close_btn (无边框窗口控制)
├── central_widget
│   ├── left
│   │   ├── wave_card (margins 36/0/36/0 — aligned with spectrogram)
│   │   │   └── WaveformWidget
│   │   └── spec_card
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
└── status_bar (zoom hint left-aligned)
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

---

## 4. 批量分析

### CSV 导出 (`analyzer/batch.py`)

`flatten_analysis()` 合并 metadata + quality analysis 为单行。所有 `BATCH_COLUMNS` 始终填充（含默认值），避免导出时 KeyError。

---

## 5. 文件依赖图

```
main_window.py
  ├── analyzer/core.py (AudioAnalyzer)
  ├── analyzer/_state.py (_stft_cache, _stft_lock)
  ├── analyzer/load.py (load_audio, is_audio_file)
  ├── analyzer/batch.py (flatten_analysis, export_batch_csv)
  ├── ui/spectrogram_widget.py (SpectrogramGLWidget, axes, colorbar)
  ├── ui/waveform_widget.py (WaveformWidget)
  ├── ui/metadata_panel.py (MetadataPanel)
  ├── ui/playback_engine.py (PlaybackEngine)
  ├── ui/anim.py (FloatAnim — 动画助手)
  └── ui/icons.py (render_icon — SVG 图标)
  ├── analyzer/palette.py (PALETTE, build_lut_np, is_spectra, get_curve_params)
  ├── lang.py (t, toggle_lang, on_lang_change)
  └── ui/styles.py (color tokens)

analyzer/core.py
  ├── analyzer/_state.py
  ├── analyzer/spectrum.py (_SpectrumMixin)
  ├── analyzer/quality.py (_QualityMixin)
  ├── analyzer/load.py
  └── analyzer/metadata.py
```

---

## 6. PyInstaller 打包注意事项

- **`numba` 不能加入 `excludes`**：librosa 懒加载依赖
- **`hiddenimports`**：pyloudnorm、pyfftw.interfaces、scipy.signal、sklearn.utils._cython_blas、soxr
- **`console=False`**（生产）+ 文件日志兜底
- **`upx_exclude`**：`.pyd` 和 numpy/scipy DLL 不压缩
- Shader 文件需打入 datas：`ui/shaders/spectrogram.vert`、`spectrogram.frag`
- `ui/icons.py` 用 `PyQt6.QtSvg`（QSvgRenderer）—— PyInstaller 自动分析该 import；图标为内联 SVG 字符串，无资源文件

---

## 7. 延迟加载策略

重量级库（librosa、pyfftw、scipy、pyloudnorm）不在模块顶层导入，而是延迟到首次使用时加载：

- `analyzer/_state.py`：`import pyfftw` 移入 `_ensure_wisdom()` / `_flush_wisdom()`
- `analyzer/spectrum.py`：`import librosa` 移入各方法内部
- `analyzer/quality.py`：`import librosa` 移入 `_measure_dynamic_range()`，`import pyloudnorm` 移入 `_measure_loudness()`，`import scipy.signal` 移入 `_detect_high_freq_cutoff()`
- `analyzer/core.py`：`_ensure_librosa()` 在 `load()` 中调用，含 FutureWarning 抑制
- `ui/main_window.py`：`AudioAnalyzer` 导入推迟到 `_LoadWorker.run()` / `_BatchWorker.run()`

---

## 8. 扩展点

1. **新格式支持** — 扩展 `SUPPORTED_EXTENSIONS` + PyAV

2. **新配色方案** — 在 `PALETTE` 加条目，标准配色用 `matplotlib.cm` 导出；自定义配色需在 `SPECTRA_CURVE` 或新建 curve 常量中定义曲线参数

3. **新分析指标** — 在 `_QualityMixin.analyze_quality()` 添加

4. **多语言扩展** — `lang.t()` 或 gettext

5. **TODO: metadata 中文键名改造** — `analyzer/metadata.py` 的 `_MAPS` 用中文字符串作字典键（`"标题"`、`"艺术家"` 等），`core.py` 的 `_TAG_TR` 做翻译中转。改为英文常量键名（`"title"`、`"artist"`）+ UI 层翻译，可消除 `batch.py` 中的硬编码中文键、补全 `_TAG_TR` 缺失条目（`发行商`、`调性`、`编码设备` 等）。影响范围：`metadata.py`、`core.py`、`batch.py`、`metadata_panel.py`、`test_analyzer.py`

6. **TODO: 声谱渲染亮度与配色优化** — 当前问题：
   
   - **感知均匀配色（viridis/magma/inferno 等）整体亮度低**：这些配色有 40-60% 暗区，线性映射下大部分能量集中在 -60~-30 dB 范围会显示为暗色
   - **spectra 配色已有亮度曲线**（`SPECTRA_CURVE`），标准配色使用线性映射
   - **可能的解决方案**：收窄 dB 范围（-100 或 -90）减少暗区覆盖、研究 Spek 的具体配色实现
   - 影响范围：`analyzer/palette.py`、`ui/spectrogram_widget.py`、`ui/shaders/spectrogram.frag`

---

> 最后更新: 2026-07-14 (UI 现代化重构 Phase 0–4：design token 统一、无边框窗口+SVG图标+快捷键、动画系统 FloatAnim（缩放缓动/LUT交叉淡化/浮层淡入淡出/滑块hover/元数据错峰）)
> 上一更新: 2026-06-16 (配色方案系统重构 + 削波/高频检测算法重写：多声道削波、位深感知、Welch PSD、多因子置信度)
> 基于文件: main.py, ui/main_window.py, analyzer/core.py, analyzer/_state.py, analyzer/spectrum.py, analyzer/quality.py, analyzer/load.py, analyzer/metadata.py, analyzer/batch.py, analyzer/palette.py, ui/spectrogram_widget.py, ui/metadata_panel.py, ui/waveform_widget.py, ui/playback_engine.py, ui/batch_dialog.py, ui/styles.py, ui/shaders/spectrogram.vert, ui/shaders/spectrogram.frag, lang.py, spectra.spec

---

## Agent skills

### Issue tracker

Issues 与 spec 存于 GitHub Issues，用 `gh` CLI 操作。见 `docs/agents/issue-tracker.md`。

### Triage labels

五个标准角色使用默认标签：`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`。见 `docs/agents/triage-labels.md`。

### Domain docs

Single-context：根目录 `CONTEXT.md` + `docs/adr/`。见 `docs/agents/domain.md`。
