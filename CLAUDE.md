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
│  │  │  [YAxis] [SpectrogramGLWidget] [ColorBar] [XAxis]   │  │  │
│  │  │          ↑ playhead overlay (shared with waveform)  │  │  │
│  │  ├─────────────────────────────────────────────────────┤  │  │
│  │  │  MetadataPanel (right sidebar)                      │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                             │  │
│  PlaybackEngine (sounddevice OutputStream)                     │  │
│  — audio playback with playhead sync                           │  │
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

**多分辨率 STFT** — 三频段（8192/2048/512），拼接去重

**相位重分配频谱图** — iZotope RX 风格（Auger-Flandrin），瞬时频率 + 群延迟一阶导数修正

**削波检测** — flat-top 检测，`np.diff` 边缘检测替代 while 循环

**动态范围** — 向量化 `librosa.feature.rms` 替代逐帧循环

**LUFS (EBU R128)** — `pyloudnorm`，降采样保护：`sr > 12000 * 1.5` 才做 decimate

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

#### 坐标轴组件
- `_YAxisWidget` — 频率轴（左）
- `_XAxisWidget` — 时间轴（下）
- `_ColorBarWidget` — dB 色条（右），渐变条宽度 7px

### 2.7 音频播放 — `ui/playback_engine.py`

- 基于 `sounddevice.OutputStream`，回调帧计数器追踪位置（无 DAC time 抖动）
- 播放/暂停/停止/Seek + 拖拽跟踪 (`track_position`)
- `_cb_frame` 读写加锁，跨线程安全
- `pause()` 在 `_close_stream()`（可能触发 `_on_stream_finished` 覆盖计数器）之后恢复保存的位置

### 2.8 Playhead 同步

- playhead 位置由 `main_window` 单一管理
- `WaveformWidget` 和 `SpectrogramGLWidget` 通过 `playhead_pos` 属性被动绘制
- 拖拽时 `_on_playhead_drag` 回调直接写两个 widget 的 `playhead_pos` 并 repaint
- 点击判定宽度 ±20px
- 不播放时 playhead 始终显示在最左端

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
│   │           ├── filename_widget (row 0)
│   │           ├── YAxisWidget (row 1, col 0, width=36)
│   │           ├── SpectrogramGLWidget (row 1, col 1, stretch)
│   │           ├── ColorBarWidget (row 1, col 2, width=36)
│   │           └── XAxisWidget (row 2)
│   └── right
│       └── MetadataPanel (width=310)
└── status_bar
```

### 关键 UI 设计模式

**i18n 系统**
- `lang.t("中文", "English")` 统一翻译入口
- `on_lang_change(callback)` 注册回调，返回 `unsubscribe()` 函数防泄漏

**样式系统 (`ui.styles`)**
- 全局 CSS token：`BG_BASE`, `BG_SURFACE`, `TEXT_PRI`, `ACCENT` 等
- 深色主题一致性

**safe_slot 装饰器**
- 所有 Qt signal-slot 主线程回调使用 `@safe_slot` 装饰器

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
| **librosa** | STFT/mel/MFCC 标准库 |
| **pyloudnorm** | EBU R128 响度标准 |
| **mutagen** | 纯 Python，多格式元数据 |
| **pyfftw** | FFTW Python 绑定，替代 numpy.fft |
| **sounddevice** | 轻量 PortAudio 绑定，音频播放 |

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

## 8. 扩展点

1. **新格式支持** — 扩展 `SUPPORTED_EXTENSIONS` + PyAV
2. **新配色方案** — 在 `_PALETTE_STOPS` / `PALETTE` 加条目
3. **新分析指标** — 在 `_QualityMixin.analyze_quality()` 添加
4. **多语言扩展** — `lang.t()` 或 gettext

---

> 最后更新: 2026-05-27
> 基于文件: main.py, ui/main_window.py, analyzer/core.py, analyzer/_state.py, analyzer/spectrum.py, analyzer/quality.py, analyzer/load.py, analyzer/metadata.py, analyzer/batch.py, analyzer/palette.py, ui/spectrogram_widget.py, ui/metadata_panel.py, ui/waveform_widget.py, ui/playback_engine.py, ui/batch_dialog.py, ui/styles.py, ui/shaders/*.glsl, lang.py, spectra.spec
