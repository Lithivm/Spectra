# Spectra — 音频分析工具架构文档

> iZotope RX-style audio analysis desktop application built on PyQt6, PyAV, and librosa.

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    main_window.py (entry)                        │
│  PyQt6 QMainWindow + drag-drop + central area                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Central area: waveform + spectrogram + metadata panel    │  │
│  │                                                         │  │
│  │  [ SpectrogramGLWidget ]                                │  │
│  │  [ WaveformWidget ]                                     │  │
│  │  [ MetadataPanel ] (right sidebar)                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                             │  │
│  Actions: batch analysis, palette/yscale toggles, language   │  │
│           export, clear, exit                               │  │
└─────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         load_audio()    AudioAnalyzer  Analyzer
         (load.py)       (core.py)     (core.py, batch.py)
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                      Analyzer.run()
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      _analyze_spectrum() _analyze_quality()
         │                   │
         ▼                   ▼
    multi-band STFT     quality metrics
    + reassigned         (clipping, upsampling,
    GLSL render          DR, LUFS, LRA)
```

---

## 2. 核心模块

### 2.1 音频加载 — `analyzer/load.py`

**设计策略**
- 主解码器：**PyAV** (libav) — 支持 FLAC, OPUS, WAV, MP3, M4A, AAC, WMA, APE, OGG, TTA, AIFF
- PyAV 解码失败时有 ffmpeg 子进程回退（容错兜底，不保证所有格式）

**关键设计决策**
- 所有格式统一输出 `(numpy.ndarray, sample_rate)` 形式，shape 为 `(channels, samples)`
- 整数格式自动归一化到 `[-1.0, 1.0]`，输出 `float32`
- ffmpeg 回退固定输出 `48kHz, 立体声`，避免硬依赖

**支持的格式**
| 格式 | 扩展名 | 解码器 |
|------|--------|--------|
| FLAC | `.flac` | PyAV |
| OPUS | `.opus` | PyAV |
| WAV | `.wav` | PyAV |
| MP3 | `.mp3` | PyAV |
| M4A/ALAC | `.m4a`, `.mp4` | PyAV |
| AAC | `.aac` | PyAV |
| WMA | `.wma` | PyAV |
| APE | `.ape` | PyAV |
| OGG Vorbis | `.ogg` | PyAV |
| TTA | `.tta` | PyAV |
| AIFF | `.aiff` | PyAV |

### 2.2 元数据解析 — `analyzer/metadata.py`

基于 **mutagen** 的多格式元数据提取：
- MP3 (ID3v2): `TIT2` → 标题, `TPE1` → 艺术家, `TALB` → 专辑 等
- MP4 (MP4 metadata): `©art` → 艺术家, `©alb` → 专辑 等
- FLAC/Vorbis: 标准 vorbis comment 字段
- ASF (WMA): Windows Media 元数据
- APE (Monkey's Audio): APEv2 tags

**映射设计**
- 内部统一使用 `str → Any` 字典返回
- 字段名保持英文键名（如 `TIT2`），供 `metadata.py` 映射到中文标签
- 返回的元数据字典包含通用字段：`format`, `bitrate`, `duration`, `channels`, `sample_rate`

### 2.3 音频分析 — `analyzer/core.py`

#### AudioAnalyzer 对象模型
```python
class AudioAnalyzer:
    filepath: Path        # 原始文件路径
    filename: str         # 文件名
    format: str           # 音频格式
    sample_rate: int      # 采样率
    duration: float       # 时长（秒）
    channels: int         # 声道数
    data: np.ndarray | None  # (channels, samples) float32
    metadata: dict | None     # 解析后的元数据
```

#### 关键算法

**多分辨率 STFT (`_multi_resolution_stft`)**
- 三个频段分别用不同 FFT 大小：
  - `0–320 Hz`: 8192 → 高频率分辨率（低频细节）
  - `280–3200 Hz`: 2048 → 均衡分辨率
  - `2800 Hz–Nyquist`: 512 → 高时间分辨率（高频瞬态）
- 拼接结果并消除重复 bin，得到统一频率轴
- 使用 `librosa.amplitude_to_db` 输出 dB

**相位重分配频谱图 (`_reassigned_spectrogram`)**
- iZotope RX 风格算法（Auger-Flandrin, IEEE TASSP 1995）
- 对每个 STFT bin 的相位求一阶导数，修正时频坐标：
  - **瞬时频率校正**：`ω̂ = ω + Im{S_t · S*} / |S|² / 2π`
  - **群延迟校正**：`τ̂ = t + Re{S_f · S*} / |S|²`
- 将幅度分配到修正后的坐标 bin 上

**频谱质心 & ZCR**
- `spectral_centroid()`：`librosa.feature.spectral_centroid`
- `zcr()`：`librosa.feature.zero_crossing_rate`

#### 质量分析（`analyze_quality`）

| 指标 | 方法 | 实现 |
|------|------|------|
| **削波** | 检查 `|audio| >= 0.999` 的连续段 | `_detect_clipping` |
| **过采样检测** | FFT 幅度下降 25 dB 处找截止频率 | `_detect_high_freq_cutoff` |
| **动态范围** | 分段 RMS 的最大值–均值 | `_measure_dynamic_range` |
| **峰值电平** | 全局 `|max|` → `20 log10` | `_compute_peak` |
| **RMS** | `sqrt(mean(x²))` | `_compute_rms` |
| **LUFS (EBU R128)** | `pyloudnorm` 库 | `_measure_loudness` |
| **True Peak** | 4× 过采样 BS.1770-4 | `_true_peak` |

#### LUFS 计算细节
```python
meter = pyln.Meter(sr)
integrated = meter.integrated_loudness(audio_st)   # 全曲
# Short-term: 3s 非重叠块
st_vals = [meter.integrated_loudness(block) for block in blocks]
short_term = max(st_vals)
# LRA: P10 到 P95 的间距
lra = sorted(st_vals)[int(0.95*n)] - sorted(st_vals)[int(0.1*n)]
```

### 2.4 配色方案 — `analyzer/palette.py`

纯数据模块（零依赖），只有 `PALETTE: dict[str, str]` 字典（name → label），共 12 种配色方案。GLSL shader 直接从 256×3 的 LUT 纹理查色，不需要 matplotlib colormap。

### 2.5 渲染器 — `ui/spectrogram_widget.py`

#### SpectrogramGLWidget (OpenGL)
- **GPU-accelerated** spectrogram rendering via `QOpenGLWidget`
- 核心数据流：
  1. dB 矩阵上传为 `GL_R32F` 2D 纹理
  2. GLSL fragment shader 做 y 轴映射（log/linear）+ colormap LUT 查询
  3. 纯 CPU 预处理 = 零（upload 后完全 GPU 运算）

**GLSL Shader 关键逻辑**
```glsl
// Y-axis: log or linear
float y;
if (u_log_scale) {
    float f_min_log  = log(20.0);
    float f_max_log  = log(22050.0);
    float f_val_log  = f_min_log + uv.y * (f_max_log - f_min_log);
    float f_norm = (exp(f_val_log) - 20.0) / (22050.0 - 20.0);
    y = clamp(f_norm, 0.0, 1.0);
} else {
    y = uv.y;
}
// dB normalization → colormap lookup
float db = texture(u_spec, vec2(uv.x, y)).r;
float t = clamp((db - u_vmin) / (u_vmax - u_vmin), 0.0, 1.0);
fragColor = texture(u_colormap, vec2(t, 0.5));
```

---

## 3. iZotope RX 配色方案

### 三频段 dB→亮度曲线

| 频段 | dB 范围 | 变换 | 视觉效果 |
|------|---------|------|----------|
| 噪声底 | DB_MIN (~-90) → NOISE_DB (~-75) | `^4.0 × 0.01` | 压向黑底，隐藏底噪 |
| 中频 | NOISE_DB → KNEE_DB (~-45) | `0.01 + s^1.8 × 0.44` | 缓慢出现，保持暗区 |
| 信号区 | KNEE_DB → 0 dB | `0.45 + 0.55 × s^0.5` | 快速提亮，音乐信号凸显 |

**Gamma = 1.0（线性）**，形状完全由曲线段控制。

### 调色板色标
```
0%     → #000000 (black)
15%    → #00051E (near-black blue)
30%    → #003359 (deep blue)
48%    → #008C99 (cyan ← RX primary)
62%    → #33994C (cyan-green)
72%    → #B38000 (orange)
82%    → #E64800 (orange-red)
91%    → #CC0D00 (deep red)
97%    → #FF6633 (bright orange)
100%   → #FFFFFF (white)
```

---

## 4. UI 组件树

```
MainWidget (QMainWindow)
├── menu_bar (action_bar)
│   ├── batch_analyze → BatchProgressDialog (QDialog)
│   ├── export_csv → CSV export via analyzer/batch.py
│   ├── clear → clears audio, resets widgets
│   ├── exit
│   └── language toggle (LANG = "zh" ↔ "en")
│
├── central_widget (QSplitter)
│   ├── left: waveform_widget (WaveformWidget)
│   ├── right: split vertically
│   │   ├── spectrogram (SpectrogramGLWidget) — top
│   │   │   ├── y-axis: _YAxisWidget (QWidget, QPainter)
│   │   │   ├── x-axis: _XAxisWidget (QWidget, QPainter)
│   │   │   └── colorbar: _ColorBarWidget (QWidget, QPainter)
│   │   └── metadata: metadata_panel (MetadataPanel)
│   │       ├── technical section (info from analyzer)
│   │       ├── tags section (metadata tags)
│   │       └── analysis section (quality metrics)
│   │
│   └── toolbar (palette, y_scale, freq labels)
```

### 关键 UI 设计模式

**i18n 系统**
- `lang.t("中文", "English")` 统一翻译入口
- `on_lang_change(callback)` 注册回调，切换语言时自动刷新
- 所有 `QLabel` 文本使用 `t()` 封装，切换语言时重新设置

**样式系统 (`ui.styles`)**
- 全局 CSS 变量定义：`BG_BASE`, `BG_SURFACE`, `BG_RAISED`, `TEXT_PRI`, `TEXT_SEC`, `ACCENT`, `ACCENT_GRN`, `ACCENT_RED` 等
- 所有 widget 内联 `setStyleSheet()` 引用统一色板
- 保证深色主题一致性

**进度/状态指示**
- 进度条：`QProgressBar` 渐变金色 chunk
- 状态点：`QLabel("●")` 用颜色指示（绿=正常/红=异常/琥珀=警告）

---

## 5. 批量分析

### 架构

```
Analyzer.run() (background thread)
  └── analyze_file(filepath)
        ├── load_audio(filepath)
        ├── metadata = get_metadata(filepath)
        └── qa = analyzer.analyze_quality()
```

### CSV 导出 (`analyzer/batch.py`)

```python
BATCH_COLUMNS = [
    "filename", "filepath", "format", "duration",
    "sample_rate", "channels", "bitrate",
    "title", "artist", "album", "year", "genre", "track",
    "peak_db", "rms",
    "integrated_lufs", "short_term_lufs", "lra_lu", "true_peak_db",
    "clipping_count", "clipping_longest_ms",
    "cutoff_hz", "upsampling_ok",
    "dynamic_range_db",
]
```

`flatten_analysis()` 把 metadata + quality analysis 合并为单行。

---

## 6. 技术选型原因

| 技术 | 原因 |
|------|------|
| **PyQt6** | 跨平台桌面，OpenGL 集成良好，支持高 DPI |
| **PyAV (libav)** | 原生解码，避免 Python 封装层损耗 |
| **ffmpeg 子进程** | 覆盖 PyAV 不支持的格式 (DSD) |
| **librosa** | STFT/mel/MFCC 标准库，社区成熟 |
| **pyloudnorm** | EBU R128 响度标准，BSD 许可 |
| **mutagen** | 纯 Python，多格式元数据支持 |
| **pyfftw** | FFTW 的 Python 绑定，替代 numpy.fft 提升性能 |

---

## 7. 数据流总结

### 单次文件分析

```
用户拖入文件
  ↓
main_window._handle_drag(file)
  ↓
analyzer = AudioAnalyzer(file)
  ├─ load_audio(file) → float32 ndarray
  ├─ metadata = get_metadata(file)
  └─ _analyze_spectrum() → multi-band STFT
  └─ run background: _analyze_quality() → clipping + LUFS + LRA + DR
  ↓
waveform_widget.set_audio(analyzer.data)
spectrogram_widget.set_audio(analyzer.data)  → GLSL render
metadata_panel.load_metadata(analyzer)       → display
metadata_panel.load_analysis(qa)             → display
```

### 批量分析

```
user → batch button
  ↓
progress_dialog.show()
  ↓
thread:
  for each file:
    analyzer = AudioAnalyzer(file)
    qa = analyzer.analyze_quality()
    result = flatten_analysis(metadata, qa, file)
    results.append(result)
  ↓
progress_dialog.finish()
  ↓
user → export button → dest file
  ↓
export_batch_csv(results, dest)
```

---

## 8. 关键算法要点

### 多分辨率 STFT
- 低频用大 FFT (8192) → 高频率分辨率（看清低频细节）
- 高频用小 FFT (512) → 高时间分辨率（捕捉瞬态）
- 拼接频段时去重，避免 bin 重叠

### 相位重分配
- 利用瞬时频率和群延迟的一阶导数修正 STFT 坐标
- 把能量"搬运"到真实时频位置，消除谱图模糊
- 实现上：`S_t` (时间乘 ramp 再做 STFT), `S_f` (频域乘 ramp 做 STFT)

### GLSL 频谱图着色
- dB → [-90, 0] → 标准化 [0,1] → LUT 查询
- 三频段非线性映射：噪声底→中频→信号区，分别用 `^4`, `s^1.8`, `s^0.5` 控制
- Gamma = 1.0（纯曲线控制亮度）

---

## 9. 文件依赖图

```
main_window.py
  ├── analyzer/core.py (AudioAnalyzer)
  ├── analyzer/load.py (load_audio, is_audio_file)
  ├── analyzer/metadata.py (get_metadata)
  ├── analyzer/batch.py (flatten_analysis, export_batch_csv)
  ├── ui/spectrogram_widget.py (SpectrogramGLWidget)
  ├── ui/metadata_panel.py (MetadataPanel)
  ├── ui/waveform_widget.py (WaveformWidget)
  ├── ui/batch_dialog.py (BatchProgressDialog)
  ├── analyzer/palette.py (PALETTE)
  ├── lang.py (t, toggle_lang)
  └── ui/styles.py (color constants)
```

---

## 10. 扩展点

1. **新格式支持**：扩展 `SUPPORTED_EXTENSIONS` + PyAV 原生支持即可
2. **新配色方案**：在 `_PALETTE_STOPS` / `PALETTE` 字典加新条目
3. **新分析指标**：在 `AudioAnalyzer.analyze_quality()` 添加新方法
4. **GPU 加速渲染**：当前 GLSL shader 已是完整 GPU 管线
5. **更多格式元数据**：扩展 `_MAPS` 字典
6. **多语言扩展**：扩展 `lang.t()` 或改用 gettext

---

## 11. 崩溃诊断与 PyInstaller 打包

### 崩溃日志

`main.py` 启动时自动初始化 file logger，写入 `%USERPROFILE%\.spectra\crash.log`（RotatingFileHandler，5 MB × 3）。同时注册：

- `sys.excepthook`：主线程未捕获异常 → `logger.critical`
- `threading.excepthook`：QThread 未捕获异常 → `logger.critical`

所有 `logger = logging.getLogger(__name__)` 的输出也会进入同一日志文件。

### safe_slot 装饰器 (`ui/main_window.py`)

所有通过 Qt signal-slot 关联的主线程回调均使用 `@safe_slot` 装饰器，捕获 `Exception` → `logger.exception` → 不再向上抛出。防止 PyQt6 在 windowed 模式下因 slot 异常调用 `abort()` 导致静默退出。

### Workder 异常保护

- `_SpectrumWorker.run()` 外层 try/except + `logger.exception("SpectrumWorker failed")`
- `_QualityWorker.run()` 的 `except:` 改为 `except Exception:` + `logger.exception("QualityWorker failed")`

### PyInstaller spec 注意事项

- **`numba` 不能加入 `excludes`**：librosa 懒加载依赖 numba，排除后 STFT 计算抛 ModuleNotFoundError → 闪退
- **`console=False`**（生产）+ 文件日志兜底，不再依赖 console 输出调试
- **`upx_exclude`**：`.pyd` 和 numpy/scipy DLL 不压缩（UPX 易损坏 native 库）
- **`hiddenimports`**：PyInstaller 自动分析不到的子模块需显式列出（pyloudnorm、pyfftw.interfaces、scipy.signal、sklearn.utils._cython_blas、soxr）

---

> 生成时间: 2026-05-22
> 最后更新: 2026-05-22（numba 闪退修复 + 崩溃日志系统 + 删除 analyzer/spectrogram.py）
> 基于文件: main.py, ui/main_window.py, analyzer/core.py, analyzer/load.py, analyzer/metadata.py, analyzer/batch.py, analyzer/palette.py, ui/spectrogram_widget.py, ui/metadata_panel.py, ui/waveform_widget.py, ui/batch_dialog.py, ui/styles.py, lang.py, spectra.spec
