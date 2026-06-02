# Spectra — 音频分析工具架构文档

> iZotope RX-style audio analysis desktop application built on PyQt6, PyAV, and librosa.

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    main_window.py (entry)                        │
│  PyQt6 QMainWindow + drag-drop + central area                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Toolbar: open, play/pause, palette, mode, FFT, scale,   │  │
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
│                                                             │  │
│  PlaybackEngine (sounddevice OutputStream)                     │  │
│  — audio playback with slider sync                             │  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块

### 2.1 模块拆分

`analyzer/core.py` 已拆分为四个模块：

| 模块 | 职责 |
|------|------|
| `analyzer/_state.py` | FFTW wisdom 管理、STFT 缓存 (LRU maxsize=8)、`_max_reduce_with_carry` |
| `analyzer/spectrum.py` | `_SpectrumMixin` — STFT、多分辨率、相位重分配、mel、MFCC、流式渲染 |
| `analyzer/quality.py` | `_QualityMixin` — 削波、过采样检测、DR、LUFS、true peak |
| `analyzer/core.py` | `AudioAnalyzer` 门面类，继承两个 mixin，保留 load/waveform/info |

所有外部导入 `from analyzer.core import AudioAnalyzer` 无需改动。

### 2.2 音频加载 — `analyzer/load.py`

**设计策略**
- 主解码器：**PyAV** (libav) — 支持 FLAC, OPUS, WAV, MP3, M4A, AAC, WMA, APE, OGG, TTA, AIFF
- PyAV 解码失败时有 ffmpeg 子进程回退（容错兜底，不保证所有格式）

**关键设计决策**
- 所有格式统一输出 `(numpy.ndarray, sample_rate)` 形式，shape 为 `(channels, samples)`
- 整数格式用 `frame.format.bits` 做通用归一化，覆盖 s16/s24/s32/s64 等所有位深
- ffmpeg 回退固定输出 `48kHz, 立体声`，避免硬依赖

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
```

#### 关键算法

**多分辨率 STFT** — 三频段重叠拼接：低频 0–320Hz (n_fft=8192)、中频 280–3200Hz (n_fft=2048)、高频 2800Hz–Nyquist (n_fft=512)，固定 hop=512。重叠区用 `np.diff` + boolean mask 向量化去重，保留严格递增频率子集

**相位重分配频谱图** — iZotope RX 风格（Auger-Flandrin, IEEE TASSP 1995）。三路 STFT：原始 `S`、时间导数 `S_t`（信号乘时间斜坡）、频率导数 `S_f`（窗函数乘频率斜坡）。从 `S` 的 real/imag 一次推导 `S_sq` 和 `mag`，通过 `ω_corr = Im(S_t·S*/S_sq)` 和 `τ_corr = Re(S_f·S*/S_sq)` 计算瞬时频率和群延迟修正量，`np.searchsorted` + `np.bincount` 将能量重分配到修正坐标

**削波检测** — flat-top 检测，`np.diff` 边缘检测，MIN_FLAT=1（单样本峰值也报削波）。硬/软分类用二阶导数（曲率）：平坦=硬削波，弯曲=软削波

**高频截止检测** — 在 1.5s 随机段（3–8 段）上做批量 FFT，映射到 128 个 log-spaced 频率箱取中位数，高斯平滑（σ=1.5）。噪底估计为高频端 10% bin 的中位数，从高频向低频扫描找能量上升 >6dB 的转折点。confidence 基于信号段与噪底对比度（>40dB → 1.0，20–40dB 线性，<20dB → 0）

**动态范围** — P95-P10 帧 RMS 差值（TT DR Meter 标准），stride 视图零拷贝，帧长 4096 / hop 2048

**LUFS (EBU R128)** — `pyloudnorm`，降采样保护：`sr > 12000 * 1.5` 才做 decimate

#### 性能优化策略

分析模块内有多处计算等价优化，不影响输出精度：

- **相位重分配**：`S_sq` 和 `mag` 从 `S` 的实部/虚部一次性推导（`real² + imag²`），省去重复的 `np.abs` 调用
- **多分辨率 STFT 频率去重**：用 `np.diff` + boolean mask 向量化替代 Python 逐元素循环
- **高频截止检测**：多段 FFT 拼成 2D 数组，单次 `np.fft.rfft(axis=1)` 批量计算，减少 Python→C 调度开销
- **True Peak 去重**：`analyze_quality()` 计算一次 true peak 后传入 `_measure_loudness()` 复用，避免重复的 4x 重采样

### 2.5 配色方案 — `analyzer/palette.py`

纯数据模块（零依赖），12 种配色方案。GLSL shader 从 256×3 的 LUT 纹理查色。

### 2.6 渲染器 — `ui/spectrogram_widget.py`

#### SpectrogramGLWidget (OpenGL)
- GPU-accelerated via `QOpenGLWidget`
- dB 矩阵上传为 `GL_R32F` 2D 纹理
- GLSL fragment shader 做 y 轴映射 + colormap LUT 查询
- Shader 文件外置：`ui/shaders/spectrogram.vert` / `spectrogram.frag`
- 流式加载：texture 初始化为 `-120.0` dB（噪声底），`GL_NEAREST` 过滤，软边界过渡
- 3-tap 垂直 box filter 消除频率混叠
- **视图状态**：`_view_t0/_view_t1`（时间窗口）、`_view_f0/_view_f1`（频率窗口），通过 shader uniform `u_t_start/u_t_end/u_fview_min/u_fview_max` 实现 GPU 端缩放
- **光标信息**：`setMouseTracking(True)`，`mouseMoveEvent` 发射 `cursor_info(time, freq, dB, px)` 信号
- **滚轮缩放**：`wheelEvent` 以光标位置为中心缩放时间轴，Shift+滚轮缩放频率轴，双击重置
- **光标竖线**：hover 时跟随鼠标；播放中鼠标离开声谱区时跟随播放进度；鼠标回到声谱区立即切回跟随鼠标
- **GL 资源管理**：`_cleanup_gl(need_context)` 释放纹理/program/VAO，`initializeGL` 传 `need_context=False`（Qt 已持有上下文），`closeEvent` 传 `True`
- **LUT 缓存**：`build_lut` / `build_lut_np` 按配色名缓存结果，避免重复计算

#### 坐标轴组件
- `_YAxisWidget` — 频率轴（左），支持 `view_f0/view_f1` 参数，过滤视窗外的刻度
- `_XAxisWidget` — 时间轴（下），支持 `view_t0/view_t1` 参数，刻度只显示整分钟（短文件显示整秒）
- `_ColorBarWidget` — dB 色条（右），渐变条宽度 7px
- 三个组件 `pad_top=0, pad_bot=0`，与声谱图完全对齐

### 2.7 音频播放 — `ui/playback_engine.py`

- 基于 `sounddevice.OutputStream`，WASAPI 共享模式（低延迟 ~10ms）
- 回调帧计数器追踪位置（无 DAC time 抖动），`_cb_frame`/`_start_frame` 统一在 `_cb_lock` 内更新
- 播放/暂停/停止/Seek + 拖拽跟踪 (`track_position`)
- `pause()` 在 `_close_stream()`（可能触发 `_on_stream_finished` 覆盖计数器）之后恢复保存的位置
- 启动时探测 WASAPI 设备默认采样率，`load()` 时自动重采样（soxr 优先，scipy 回退）
- `latency='high'` 使用较大缓冲区避免 underrun

### 2.8 播放进度条 — `_PlaybackSlider`（`ui/main_window.py`）

- 自定义 QWidget，位于声谱图与 X 轴之间（grid row 2, col 1）
- 轨道 + 进度填充 + 可拖拽圆形滑块
- 拖拽时实时跳转播放位置（`sliderPressed`/`sliderReleased`/`valueChanged` 信号）
- `resizeEvent` 中向左右各扩展 `_PAD=8px`，滑块在端点不裁切，轨道与声谱图等宽对齐
- 播放时滑块自动跟随（`_on_playback_tick` 更新 value），停止时归零

---

## 3. UI 组件树

```
MainWindow (QMainWindow)
├── toolbar
│   ├── brand_label "Spectra"
│   ├── open_btn
│   ├── play_label + play_btn (▶/‖ toggle)
│   ├── palette_label + palette_combo
│   ├── mode_label + mode_combo
│   ├── fft_label + fft_combo
│   ├── yscale_label + yscale_combo
│   ├── save_btn
│   └── lang_btn
├── central_widget
│   ├── left
│   │   ├── wave_card (margins 36/0/36/0 — aligned with spectrogram)
│   │   │   └── WaveformWidget
│   │   └── spec_card
│   │       └── QGridLayout
│   │           ├── filename_widget (row 0, col 0-2)
│   │           ├── YAxisWidget (row 1, col 0, width=36)
│   │           ├── SpectrogramGLWidget (row 1, col 1, stretch)
│   │           ├── ColorBarWidget (row 1, col 2, width=36)
│   │           ├── PlaybackSlider (row 2, col 1, height=20)
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

## 5. 技术选型原因

| 技术 | 原因 |
|------|------|
| **PyQt6** | 跨平台桌面，OpenGL 集成良好，支持高 DPI |
| **PyAV (libav)** | 原生解码，避免 Python 封装层损耗 |
| **librosa** | STFT/mel/MFCC 标准库（延迟加载，quality.py 已不依赖） |
| **pyloudnorm** | EBU R128 响度标准 |
| **mutagen** | 纯 Python，多格式元数据 |
| **pyfftw** | FFTW Python 绑定，替代 numpy.fft |
| **sounddevice** | 轻量 PortAudio 绑定，WASAPI 共享模式播放 |
| **soxr** | 高质量重采样（WASAPI 采样率适配，优先于 scipy） |

---

## 6. 文件依赖图

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
  ├── analyzer/palette.py (PALETTE)
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

## 7. 崩溃诊断与 PyInstaller 打包

### 崩溃日志

`main.py` 启动时自动初始化 file logger，写入 `%USERPROFILE%\.spectra\crash.log`（RotatingFileHandler，5 MB × 3）。

### safe_slot 装饰器

所有 Qt signal-slot 主线程回调使用 `@safe_slot`，捕获异常后记日志，防止 PyQt6 因 slot 异常调用 `abort()` 静默退出。

### PyInstaller spec 注意事项

- **`numba` 不能加入 `excludes`**：librosa 懒加载依赖
- **`hiddenimports`**：pyloudnorm、pyfftw.interfaces、scipy.signal、sklearn.utils._cython_blas（验证过 scipy 1.15 / sklearn 1.7 仍有效）、soxr
- **`console=False`**（生产）+ 文件日志兜底
- **`upx_exclude`**：`.pyd` 和 numpy/scipy DLL 不压缩
- Shader 文件需打入 datas：`ui/shaders/spectrogram.vert`、`spectrogram.frag`

---

## 8. 启动优化

### 延迟加载策略

重量级库（librosa、pyfftw、scipy、pyloudnorm）不在模块顶层导入，而是延迟到首次使用时加载：

- `analyzer/_state.py`：`import pyfftw` 移入 `_ensure_wisdom()` / `_flush_wisdom()`
- `analyzer/spectrum.py`：`import librosa` 移入各方法内部（stft、spectrogram_db 等 10 个方法）
- `analyzer/quality.py`：`import librosa` 移入 `_measure_dynamic_range()`，`import pyloudnorm` 移入 `_measure_loudness()`
- `analyzer/core.py`：`_ensure_librosa()` 在 `load()` 中调用，含 FutureWarning 抑制
- `ui/main_window.py`：`AudioAnalyzer` 导入推迟到 `_LoadWorker.run()` / `_BatchWorker.run()`，`_PreloadWorker` 后台预热

### i18n 系统

- `lang.t("中文", "English")` 统一翻译入口
- `on_lang_change(callback)` 注册回调，返回 `unsubscribe()` 函数防泄漏
- `MetadataPanel._retranslate()` 使用 `set_texts()` 原地更新，避免 `deleteLater()` 导致的语言切换崩溃

---

## 9. 扩展点

1. **新格式支持** — 扩展 `SUPPORTED_EXTENSIONS` + PyAV
2. **新配色方案** — 在 `_PALETTE_STOPS` / `PALETTE` 加条目
3. **新分析指标** — 在 `_QualityMixin.analyze_quality()` 添加
4. **多语言扩展** — `lang.t()` 或 gettext

---

> 最后更新: 2026-06-02 (性能优化更新)
> 基于文件: main.py, ui/main_window.py, analyzer/core.py, analyzer/_state.py, analyzer/spectrum.py, analyzer/quality.py, analyzer/load.py, analyzer/metadata.py, analyzer/batch.py, analyzer/palette.py, ui/spectrogram_widget.py, ui/metadata_panel.py, ui/waveform_widget.py, ui/playback_engine.py, ui/batch_dialog.py, ui/styles.py, ui/shaders/spectrogram.vert, ui/shaders/spectrogram.frag, lang.py, spectra.spec
