# Spectra

> A modern OpenGL-rendered spectrogram analyzer for music enthusiasts and HiFi listeners
> 
> 一款面向音乐爱好者与 HiFi 发烧友的现代化频谱分析工具

![screenshot](demo.jpg)

---

## Features · 功能特性

| English | 中文 |
|---------|------|
| **OpenGL Spectrogram** — fragment shader with real-time colormap switching (12 palettes) | **OpenGL 频谱图** — 片段着色器渲染，实时切换 12 种色板 |
| **Audio Playback** — play/pause with synchronized playhead on waveform + spectrogram | **音频播放** — 播放/暂停，波形与频谱同步进度指示 |
| **Draggable Playhead** — click or drag to seek, always visible at current position | **可拖拽进度线** — 点击或拖拽跳转，始终显示当前位置 |
| **Quality Analysis** — clipping detection, upsampling check, dynamic range (DR), LUFS (EBU R128), true peak, LRA | **质量分析** — 削波检测、升频检测、动态范围、响度 (LUFS)、真峰值 |
| **Multi-resolution STFT** — standard, multi-band, and phase-reassigned (iZotope RX style) | **多分辨率 STFT** — 标准、多频段、相位重分配三种模式 |
| **Flexible Y-axis** — linear, logarithmic, mel, and bark frequency scales | **灵活 Y 轴** — 线性、对数、mel、bark 四种频率刻度 |
| **Waveform Preview** — down-sampled envelope aligned with spectrogram | **波形预览** — 降采样包络，与频谱上下对齐 |
| **Drag & Drop** — WAV, FLAC, MP3, M4A, OGG, AAC, OPUS, APE, AIFF, WMA, TTA | **拖放加载** — 支持 11 种常见音频格式 |
| **Bilingual UI** — Chinese / English one-click toggle | **双语界面** — 工具栏一键切换中文 / 英文 |
| **Batch Analysis** — process folders, export to CSV | **批量分析** — 批量处理文件夹，导出 CSV |
| **Screenshot Export** — save spectrogram view as PNG | **截图导出** — 保存当前频谱视图为 PNG |

## Supported Formats · 支持格式

| Format | Extension |
|--------|-----------|
| FLAC | `.flac` |
| WAV | `.wav` |
| MP3 | `.mp3` |
| M4A / ALAC | `.m4a` `.mp4` |
| AAC | `.aac` |
| OGG Vorbis | `.ogg` |
| OPUS | `.opus` |
| Monkey's Audio | `.ape` |
| WMA | `.wma` |
| True Audio | `.tta` |
| AIFF | `.aiff` |

All formats decoded via PyAV (libav). No external dependencies required.

全格式通过 PyAV (libav) 解码，无需外部依赖。

## Quality Metrics · 质量指标

| Metric | Description |
|--------|-------------|
| **Clipping** | Flat-top detection — hard/soft clip count, longest duration |
| **Hi-freq Cutoff** | Multi-segment median spectrum analysis — detects upsampled/low-passed sources |
| **Dynamic Range** | Segmented RMS max-to-mean ratio |
| **LUFS (I)** | EBU R128 integrated loudness |
| **LUFS (S)** | Short-term loudness (3 s blocks, maximum) |
| **LRA** | Loudness range (P10–P95 spread) |
| **True Peak** | 4× oversampling BS.1770-4 |
| **Peak / RMS** | Global sample peak and RMS level |

## Quick Start · 快速启动

### From source · 从源码运行

```bash
git clone https://github.com/Lithivm/Spectra.git
cd Spectra
pip install -r requirements.txt
python main.py
```

Drag an audio file into the window, or click **Open File** (打开文件).

拖入音频文件，或点击 **打开文件** 按钮。Click **▶** (播放) to preview audio.

### Pre-built · 打包版本

Download `Spectra-v0.2.0-windows.zip` from [Releases](https://github.com/Lithivm/Spectra/releases), extract, and run `Spectra.exe`.

从 Releases 下载 `Spectra-v0.2.0-windows.zip`，解压后运行 `Spectra.exe`。

## Build · 构建

```bash
pip install pyinstaller
pyinstaller spectra.spec --noconfirm
```

Output is in `dist/Spectra/`. See `spectra.spec` for packaging details.

输出在 `dist/Spectra/`，打包配置见 `spectra.spec`。

## Colormaps · 色板

| Name | Description |
|------|-------------|
| `rx` | iZotope RX style — black → cyan → orange → white |
| `inferno` | Perceptually uniform, warm (default) |
| `viridis` | Perceptually uniform, blue-green-yellow |
| `plasma` | Perceptually uniform, purple-orange-yellow |
| `magma` | Perceptually uniform, dark-purple-yellow |
| `cividis` | Perceptually uniform, blue-yellow (colorblind-safe) |
| `hot` | Black → red → yellow |
| `coolwarm` | Cool blue → warm red (diverging) |
| `seismic` | Blue → white → red (diverging) |
| `ice` | Dark blue → cyan → white |
| `fire` | Black → orange → yellow → white |
| `aurora` | Dark teal → green → warm white |

## Architecture · 架构

```
Spectra/
├── main.py                     # Entry point + crash logger
├── lang.py                      # Bilingual (zh/en) toggle
├── analyzer/
│   ├── core.py                  # AudioAnalyzer facade class
│   ├── _state.py                # FFTW wisdom + STFT cache (LRU)
│   ├── spectrum.py              # _SpectrumMixin — STFT, reassigned, mel
│   ├── quality.py               # _QualityMixin — clipping, LUFS, DR
│   ├── load.py                  # PyAV multi-format decoder
│   ├── metadata.py              # Tag extraction via mutagen
│   ├── palette.py               # Colormap registry (zero-dependency)
│   └── batch.py                 # CSV export for batch analysis
├── ui/
│   ├── main_window.py           # Main window, toolbar, workers, safe_slot
│   ├── spectrogram_widget.py    # QOpenGLWidget renderer + axis/colorbar
│   ├── waveform_widget.py       # Waveform envelope + playhead
│   ├── metadata_panel.py        # File info + quality analysis panel
│   ├── playback_engine.py       # Audio playback via sounddevice
│   ├── batch_dialog.py          # Batch progress dialog
│   ├── styles.py                # Color tokens for dark theme
│   └── shaders/
│       ├── spectrogram.vert     # GLSL vertex shader
│       └── spectrogram.frag     # GLSL fragment shader
└── assets/
    ├── logo.png
    └── logo.ico
```

## Tech Stack · 技术栈

| Library | Purpose |
|---------|---------|
| **PyQt6** | Desktop UI + OpenGL widget |
| **librosa** | STFT, spectral features |
| **pyFFTW** | FFTW bindings (faster FFT) |
| **pyloudnorm** | EBU R128 loudness |
| **PyAV** | Audio decoding (libav) |
| **sounddevice** | Audio playback (PortAudio) |
| **mutagen** | Metadata extraction |
| **scipy** | Signal processing |
| **numpy** | Numerical arrays |
| **numba** | JIT (librosa dependency) |

## Troubleshooting · 故障排查

If the app crashes, check `%USERPROFILE%\.spectra\crash.log` for details.
All uncaught exceptions and worker failures are logged there.

如果应用崩溃，查看 `%USERPROFILE%\.spectra\crash.log` 获取详细错误信息。

## License · 许可证

MIT

---

*Built with PyQt6, OpenGL, librosa, pyFFTW, and sounddevice.*
