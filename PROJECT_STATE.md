# Spectra

## 当前功能
- 项目骨架已创建（目录、`requirements.txt`、`pyproject.toml`）
- 核心依赖已定义（PyQt6、mutagen、librosa、numpy、matplotlib、soundfile）
- 音频分析引擎已实现 — 波形/频谱/MFCC/RMS/质心/过零率
- 元数据提取 — mutagen + 中文映射
- 频谱渲染 — matplotlib (Qt5Agg)
- **质量分析**（2026-04-28） — `analyze_quality()` 返回削波/疑似升频/动态范围三项指标
- UI — 主窗口 + 拖放 + 元数据面板 + 频谱面板
- 入口 — `main.py`
- 打包 — `spectra.spec`

## 文件结构

```
spectra/
├── main.py                     # ✅ 已创建 — 入口
├── pyproject.toml              # ✅ 已创建 — 项目元数据
├── requirements.txt            # ✅ 已创建 — 依赖列表
├── analyzer/
│   ├── __init__.py             # ✅ 已创建，暴露 AudioAnalyzer
│   ├── core.py                 # ✅ 已创建 — AudioAnalyzer
│   ├── metadata.py             # ✅ 已创建 — 元数据提取 + 中文映射
│   └── spectrogram.py          # ✅ 已创建 — STFT + matplotlib 渲染
├── ui/
│   ├── __init__.py             # ✅ 已创建，暴露 MainWindow
│   ├── main_window.py          # ✅ 已创建 — QMainWindow + 拖放
│   ├── metadata_panel.py       # ✅ 已创建 — 元数据标签面板
│   ├── waveform_widget.py      # ✅ 已创建 — 波形绘制
│   └── spectrogram_widget.py   # ✅ 已创建 — 频谱图面板
└── spectra.spec         # ✅ 已创建 — PyInstaller 打包配置
```

## TODO

### 已实现
- [x] 项目骨架（目录、配置文件）
- [x] `analyzer/metadata.py` — mutagen 元数据提取 + 中文字段映射
- [x] `analyzer/core.py` — AudioAnalyzer 类（音频解码 + 元数据读取）
- [x] `analyzer/spectrogram.py` — STFT 计算 + matplotlib 渲染
- [x] `ui/waveform_widget.py` — 波形绘制
- [x] `ui/metadata_panel.py` — 元数据面板 + 质量分析（ANALYSIS section）
- [x] `ui/spectrogram_widget.py` — 频谱图面板
- [x] `ui/main_window.py` — 主窗口 + 拖放 + 布局
- [x] `main.py` — 入口
- [x] `spectra.spec` — PyInstaller 打包配置
- [x] **音频质量分析** — 削波检测/疑似升频/动态范围（numpy/scipy，无新依赖）

### 音频质量分析实现 (2026-04-28)
- 在 `analyzer/core.py` 新增 `analyze_quality()` 方法，返回 `dict` 含削波/疑似升频/动态范围三项指标
- 纯 numpy + scipy 实现，无新依赖
- 在 `ui/metadata_panel.py` 滚动区域底部新增 "ANALYSIS" section，显示绿色✓（正常）或黄色⚠（警告）
- `MetadataPanel.load_metadata()` 调用 `analyze_quality()` 并渲染结果
- 所有结论措辞用"疑似"，不用"确认"

## 已修复的问题 (已完成)
- `_load_file` 中方法调用参数不匹配 → 改为正确的 API
- `_load_file` 中使用不存在的属性/方法 (`_metadata`, `set_metadata`) → 改为正确 API
- `info()` 中冗余字段 → 已移除
- 冗余的局部导入 → 已清理
- `_current_palette` 未定义错误 → 改为正确初始化方式

## 修复历史

### _update_theme 方法 (已完成)
- 旧代码引用了一个不存在的 `_update_theme` 方法，在 `set_palette` 和 `set_theme` 中调用
- 修复方法：删除所有 `_update_theme` 引用，使用实际的颜色设置代码替代

### palette_combo 初始化和位置问题 (已完成)
- `_current_palette` 初始化在 `_spec` 之前，导致 None 报错
- 修复：移除 `_current_palette`，改用 `_on_palette_changed("magma")` 初始化
- `palette_menu.addAction` 在循环内被调用了多次（每次都创建新菜单）
- 修复：先创建 `palette_menu`，循环内统一添加 actions
- 初始配色设置放在了正确的位置（`_create_widgets` 末尾）

### _load_file 修复 (已完成)
- `_load_file` 中 `self._meta.set_metadata(data, path)` 改为 `self._meta.load_metadata(self._analyzer)`（`MetadataPanel` 没有 `set_metadata` 方法）
- `_load_file` 中 `self._wave.render(data["waveform"], ...)` 改为 `self._wave.render(self._analyzer.waveform, ...)`
- 移除了 `_load_file` 中不再使用的 `self._metadata` 属性

### info() 方法清理 (已完成)
- 移除了 `info()` 中不再需要的 `waveform` 和 `sample_rate` 字段
- 移除 `_load_file` 中对 `self._metadata = self._analyzer.info()` 的冗余赋值

## _update_theme 修复 (已完成)
- 旧代码引用了一个不存在的 `_update_theme` 方法，在 `set_palette` 和 `set_theme` 中调用
- 修复方法：删除所有 `_update_theme` 引用，使用实际的颜色设置代码替代

## palette_combo 初始化和位置问题 (已完成)
- `_current_palette` 初始化在 `_spec` 之前，导致 None 报错
- 修复：移除 `_current_palette`，改用 `_on_palette_changed("magma")` 初始化
- `palette_menu.addAction` 在循环内被调用了多次（每次都创建新菜单）
- 修复：先创建 `palette_menu`，循环内统一添加 actions
- 初始配色设置放在了正确的位置（`_create_widgets` 末尾）

## _load_file 修复 (已完成)
- `_load_file` 中 `self._meta.set_metadata(data, path)` 改为 `self._meta.load_metadata(self._analyzer)`（`MetadataPanel` 没有 `set_metadata` 方法）
- `_load_file` 中 `self._wave.render(data["waveform"], ...)` 改为 `self._wave.render(self._analyzer.waveform, ...)`
- 移除了 `_load_file` 中不再使用的 `self._metadata` 属性

## info() 方法清理 (已完成)
- 移除了 `info()` 中不再需要的 `waveform` 和 `sample_rate` 字段
- 移除 `_load_file` 中对 `self._metadata = self._analyzer.info()` 的冗余赋值

## 下一步入口
1. 安装: `pip install PyQt6 mutagen librosa numpy scipy matplotlib soundfile`
2. 运行: `python main.py`
3. 打包: `pyinstaller --clean spectra.spec`
4. 验证质量分析: 打开音频文件，观察元数据面板底部新增的 "ANALYSIS" section

## 修复记录 (2026-04-28)

### palette_combo 类型错误 (`main_window.py` line 67)
- `Renderer.PALETTES` 不存在 — `Renderer` 是 `SpectrogramRenderer` 类，palette 是 `PALETTE` 模块级 dict
- 修复：改为 `Renderer.PALETTE`

### 未定义的 `load_font_families()` 函数 (`main.py` line 16)
- `load_font_families()` 从未定义，导致 `NameError`
- 修复：移除调用

### `__init__.py` 缺少 re-export
- `analyzer/__init__.py` 未导出 `Analyzer`, `AudioFile`, `AudioInfo`
- `ui/__init__.py` 未导出 `SpectrogramWidget`
- 修复：补充所有 re-export

## UI 风格升级 (2026-04-29)

### 已完成
- **深色主题** — 从浅色调色板 (Breeze Light) 切换到现代深色风格
  - 背景: `#121212` — 接近 Claude 风格的深灰
  - 卡片: `#1E1E1E` — 类似 GitHub dark 的主题
  - 主色: `#4C9AFF` (蓝色系)
  - 辅助色: `#00C9B7` (蓝绿)
  - 文字: `#E6E6E6` (主) / `#9E9E9E` (中灰)
  - 边框: `#2B2B2B` (低对比度)

- **样式表升级** (`main_window.py` — `set_style`)
  - 主窗口: 大圆角 `20px` + 微渐变
  - Dock 标题: `#222222` 背景 + 大圆角
  - GroupBox: 大圆角 `14px` + 内边距 `16px`
  - LineEdit/TextEdit: 圆角 `10px` + 边框 `1.5px solid #2B2B2B`
  - 下拉列表: 圆角 `12px` + 卡片背景
  - 工具栏: 无边框 + 悬停态 `#00C9B715`
  - 按钮: 主色 `#4C9AFF` 背景 + 圆角 `10px`
  - 按钮组: 渐变背景 + 圆角 `8px`
  - 菜单: 卡片风格圆角
  - 标签 (`QLabel#meta-card`): 半透明背景 `#1E1E1E90` + 圆角 `8px`
  - 滚动条: 6px 宽 + 圆角 + 半透明轨道
  - TreeView: 卡片背景 + 半透明选中态
  - 标签页: 无边框 + 圆角 `8px`
  - Tab 标签: 半透明背景 + 圆角 + 底部指示线
  - QDockWidget 标题: 大圆角 + 半透明背景
  - QDockWidget 关闭/浮动按钮: 简化布局
  - QHeaderView::section: 背景 `#1E1E1E` + 大圆角 + 粗体
  - QToolButton: 圆形 `32px` + 无边框 + 渐变色
  - QToolBar QToolButton: 渐变色 + 悬停发光
  - QMenuBar: 圆角 + 半透明 + 无边框
  - QMenuBar::item: 选中态大圆角 + 渐变

- **QPalette 设置** (`_set_dark_palette`)
  - 深色调色板，替代旧 `QPalette()` 浅色调色板
  - Window: `#121212`, WindowText: `#E6E6E6`
  - Base: `#181818`, AlternateBase: `#222222`
  - Text: `#E6E6E6`, Button: `#222222`
  - ButtonText: `#FFFFFF`
  - Highlight: `#00C9B7` 主色高亮

- **QMainWindow 背景**: 设置 `#121212` `QWidget` 背景色

- **Dock 面板**: 添加 `setStyle` 和 `_set_dark_palette` 调用
- **主窗口**: 添加 `setStyle` 和 `_set_dark_palette` 调用

## 修复记录 (2026-04-29, 续)

### QPushButton 未导入 (`main_window.py`)
- 使用了 `QPushButton` 但未在 `from PyQt6.QtWidgets import` 中导入
- 修复：在 `QtWidgets` 的 import 中添加了 `QPushButton`

### TODO
- 旧测试文件已损坏（依赖了已重构的 API）
- 重写 `tests/test_main_window.py` — 使用 `unittest.mock` 模拟 `QApplication`
- 新增测试：
  - `_is_audio` 接受/拒绝
  - 拖放接受/拒绝
  - 拖放后创建 analyzer
  - 调色板切换 (magma/inferno/viridis)
  - 仅选中动作改变
  - 调色板变化通知 spectrogram
- 全部通过 ✅

## 声谱渲染重构 — iZotope RX 风格 (2026-04-30)

### 目标
参考 iZotope RX 的声谱渲染方案，重构项目的 STFT 分析与渲染管线，使画面背景深邃、只有音乐信号亮起。

### 已完成

#### 1. STFT 分析引擎重构 (`analyzer/core.py`)
- **默认 FFT Size: 4096** — 频率分辨率 ~10.7 Hz @ 44.1kHz
- **75% Overlap** — `hop_length = n_fft // 4`，消除时间轴条纹
- **移除零填充** — `win_length = n_fft`，使用完整 FFT 窗
- **`spectrogram_db()` 新增 `mode` 参数**:
  - `"standard"` — 单分辨率 STFT（默认）
  - `"multi"` — **三频段多分辨率 STFT**（低 8192 / 中 2048 / 高 512），模拟心理声学
  - `"reassign"` — **相位重分配声谱**（Auger-Flandrin 方法），谐波超清晰
- **多分辨率 STFT** 使用统一 `hop_length` 对齐各频段时间帧，缝合为单一声谱
- **重分配** 通过相位导数计算瞬时频率与群延迟，将能量重新定位到真实时频坐标

#### 2. 渲染管线重写 (`ui/spectrogram_widget.py`)
- **色彩映射**:
  - `DB_MIN = -100` — 纯黑底色（噪底）
  - `KNEE_DB = -20` — 音乐信号在此之上才亮起
  - 三段式 dB→亮度曲线: 幂律压缩(-100~-80) → sigmoid 渐变(-80~-20) → 指数上升(-20~0)
  - **Gamma = 1.5** 全局压制背景底噪
- **Y 轴插值**:
  - 低频 (< 3kHz) 使用 **Cubic 插值**消除马赛克
  - 高频使用线性插值保留瞬态细节
  - 支持四种频率刻度: Log / Mel / Bark / Linear
- **QImage 渲染** — 替代逐像素 `drawRect()`，构建 RGBA uint8 数组后一次性 `drawImage()` blit
- **新增调色板**: ice（蓝青白）、fire（黑红黄白）、aurora（绿蓝紫）

#### 3. UI 更新 (`ui/main_window.py`)
- **Mode 选择器**: Standard / Multi-Resolution / Reassigned
- **Scale 选择器**: Log / Mel / Bark / Linear
- FFT 选择器在 multi 模式下自动灰化
- 默认 FFT 改为 4096

#### 4. 调色板同步 (`analyzer/spectrogram.py`)
- 新增 ice / fire / aurora 调色板
- 导入 `matplotlib.colors` 用于 `LinearSegmentedColormap`

### 当前问题
- 渲染效果不理想，需要进一步调试色彩映射和插值参数
- 重分配模式计算量大，大文件可能较慢
- 多分辨率模式频段缝合处可能有伪影

### 下一步
1. 联调渲染参数，确保声谱画面符合预期
2. 验证三种模式的实际视觉效果
3. 可能需要调整 cubic 插值的 crossover 频率或插值方法
4. 考虑为多分辨率模式添加频段间的 crossfade
