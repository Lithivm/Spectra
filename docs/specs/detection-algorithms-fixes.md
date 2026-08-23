# Spec — 检测算法修复规划

> 来源：2026-08 代码审查（`analyzer/quality.py`、`analyzer/spectrum.py`、`analyzer/core.py`）。
> 每项含「现状 / 根因 / 目标 / 方案 / 涉及文件 / 测试与验收 / 优先级·工作量」。
> 实测数据来自合成信号验证（无音频文件依赖），复现脚本见各条「现状」。

## 0. 范围与总览

| # | 问题 | 类别 | 优先级 | 工作量 | 里程碑 |
|---|------|------|--------|--------|--------|
| F1 | 峰值 / True Peak / LUFS 只读第 0 声道 | 正确性 bug | **P0** | M | M1 |
| F2 | 削波检测对孤立满幅峰误报 | 正确性 bug | **P0** | S | M1 |
| F3 | 高频截止检测：`cutoff_hz` 不可靠 + 语义需如实 | 数值可信度 | P1 | S–M | M2 |
| F4 | DR 与 TT DR Meter 不符（docstring 夸大） | 标准符合度 | P1 | S–M | M2 |
| F5 | 相位重分配：量纲/归一化与 Auger-Flandrin 不符 | 算法保真度 | P2 | L | M3 |
| F6 | True Peak 全文件 4× float64 上采样，内存峰值高 | 健壮性 | P3 | S | M4 |
| F7 | `_mono` 命名误导 + 声谱图只画左声道 | 正确性/体验 | P3 | S | M4 |
| F8 | short-term loudness 非重叠块 / LRA 简化 | 标准符合度 | P3 | S | M4 |
| F9 | DR `as_strided` 极短音频越界 view（UB） | 健壮性 | P3 | XS | M4 |

**里程碑顺序**：M1（P0，用户直接可见的错误）→ M2（P1，数值可信度）→ M3（P2，算法保真，验证先行）→ M4（P3，健壮性/命名）。每个 F 项独立可交付、独立可回归。

## 1. 统一测试策略（所有 F 项共用）

- **全部用合成信号**，不依赖音频文件，沿用 `tests/test_analyzer.py` 现有模式（`_make_analyzer_with_audio`）。
- 扩展 fixture：新增立体声 / 多声道构造 helper（当前只有单声道），因为 F1/F7 的核心就是多声道。
- **回归测试要求**：每个 P0/P1 项的测试必须在「当前代码上失败、修复后通过」（先红后绿）。
- 验证命令：`py -m pytest tests/test_analyzer.py -q`（注意本机 `python`/`python3` 是商店占位符，用 `py`）。

---

## F1 — 峰值 / True Peak / LUFS 只读第 0 声道（P0）

**现状（实测）**：L 幅度 0.10、R 幅度 0.90 的立体声：
```
reported peak_db      = -20.0   （读的是 L）
reported true_peak_db = -20.0
reported integrated   = -20.1 LUFS
manual max-over-ch    =  -0.9   （正确值）
```
偏差达 ~19 dB。

**根因**：`core.py` 中 `self._mono = self.data[0]`（左声道，非 mixdown），而 `analyze_quality()`（在 `quality.py`）把 `audio = self._mono` 传给 `_compute_peak`、`_true_peak`、`_measure_loudness`。**M1 实现时进一步发现：DR / rms / zero_crossing / upsampling 也全用 channel-0**（不只 peak/loudness）——即除 clipping 外几乎全是单声道。M1 按测试范围只修了 peak/true-peak/loudness 三项；DR/rms/zcr 的声道问题并入 M2（F4）。

**目标**：三个指标反映完整混音/全声道，而非单声道。

**方案**：
1. `peak_db`：对 `self.data` 所有声道取 `max |x|`（逐声道 argmax 再取全局）。
2. `_true_peak`：BS.1770 true peak 本就是「逐声道取最大」。把完整 `(samples, n_ch)` 传入，`resample_poly(..., axis=0)` 已支持多声道，取全局 max。移除 `np.column_stack([mono, mono])` 的假立体声 hack。
3. `_measure_loudness`：把完整 `self.data.T`（`(samples, n_ch)`）传给 `pyloudnorm.Meter`——它自带 R128 声道增益，直接得到正确混音 integrated loudness。移除「mono 复制成假立体声」逻辑。
4. `self._mono`（channel 0）**仅保留给 STFT/声谱图可视化**（见 F7）。

**涉及文件**：`analyzer/quality.py`（`analyze_quality`、`_true_peak`、`_measure_loudness`；`_compute_peak` 保留供测试，生产路径改为内联全声道 peak）。

> **M1 实现状态**：已完成。`analyze_quality` 现用 `self.data.T`（全声道）算 peak/true-peak/loudness；`_measure_loudness` 改 `(samples, n_ch)` 契约 + `decimate(axis=0)`，去掉 fake-stereo hack；DR/rms/zcr/upsampling 暂仍 channel-0（见上）。回归测试 `TestChannelHandling` + `TestClippingIsolation`，29 项全绿。实证：L=0.1/R=0.9 → peak -20.0→**-0.9**、true_peak -20.0→**-0.9**、integrated -20.1→**-4.0**。

**测试与验收**：
- 立体声 R 明显比 L 响 → `peak_db`/`true_peak_db`/`integrated_lufs` 反映较响声道 / 正确混音（不再等于 L 的 -20 dB）。
- 单声道文件结果不变（回归保护）。
- 双声道等幅 → integrated ≈ 单声道 + 3 dB 左右（R128 双声道增益），作为 sanity。

---

## F2 — 削波检测对孤立满幅峰误报（P0）

**现状（实测）**：192 个相互隔离的单样本 1.0 尖峰（无 flat-top）→ `ok=False, count=192`。真 hard clip（flat-top run）也能正确抓到（`count=4000`）。

**根因**：`_detect_clipping` 把单样本满幅峰也计入 `total_count`，只要 `total_count > 0` 就 `ok=False`。**满幅峰值 ≠ 削波**——干净母带里瞬态尖峰孤立触及 0 dBFS 会被判为 clip。

**目标**：只有连续 flat-top run 才判 clip；孤立满幅峰单独作为信息项，不触发 `ok=False`。

**方案**：
- 引入 `MIN_RUN = 2`（常量，可配）：只有长度 ≥ `MIN_RUN` 的连续 run 计为 clip、计入 `total_count`、决定 `ok`。
- `single_sample_count` 保留为信息字段（「满幅峰值数量」），不再影响 `ok`。
- **阈值合理性论证**（为何 MIN_RUN=2 能干净分离）：`CLIP_THRESH=0.999` ≈ -0.0087 dBFS，极紧。48k/1kHz 正弦周期 48 样本、相位步进 0.1309 rad/样本，只有距峰 <0.34 样本的采样才 ≥0.999 → 干净满幅正弦只产生**孤立单样本**；而 amp=1.5 削波到 ±1.0 的正弦每周期顶部有 ~13 个连续满幅样本。故 MIN_RUN=2 恰好分离两者。

**涉及文件**：`analyzer/quality.py`（`_detect_clipping`）。

**测试与验收**：
- 孤立单样本满幅峰 → `ok=True`（或至少不判 clip），`single_sample_count` 正确计数。
- flat-top run（≥2）→ `ok=False`，`count`/`hard_clips` 正确。
- **干净满幅正弦**（amp=1.0，无削波）→ `ok=True`（当前代码会误报，修复后通过——关键回归）。

---

## F3 — 高频截止检测：`cutoff_hz` 不可靠 + 语义需如实（P1）

**现状（实测，Nyquist=24k，阈值线 0.85·nyq=20.4k）**：
```
full-band noise        -> ok=True  cutoff=24000  （ok 正确；cutoff 正确）
gentle LP@15k (4阶)    -> ok=True  cutoff=22260  （ok 正确，温和滚降不判）
sharp brickwall@15k    -> ok=False cutoff=16963  （ok 正确；cutoff 略偏高，应 ~15000）
phone band-limit@3.4k  -> ok=False cutoff=15768  （ok 合理；但 cutoff 值错，应 ~3400）
bass-only 80-5k        -> ok=False cutoff=17666  （ok 合理；cutoff 值错，应 ~5000）
```

**根因**：`cutoff_hz` 不可靠——高→低扫描在 floor 区遇到 Welch PSD 的单 bin 估计噪声尖峰就停下，报出近 Nyquist 的值（15–21k），而非真实频带边缘（3.4–10k）。slope 还固定测「上 1/3 频段（≥8kHz）」，对 8k 以下的真实边缘不敏感。

**自我修正说明**：初版 spec 把「bass-only 被判 ok=False」当作误报、并计划让 phone=判 / bass-only=不判。**这不成立**——phone band-limit 与 bass-only 在频谱上无法区分（都是「低频有内容、高频 floor」），任何纯谱形判据都无法只判其一。故撤销「bass-only 误报」：检测器报告「内容带宽低于 Nyquist」是**事实正确**的（band-limited = flagged）；是否为"缺陷"取决于文件预期带宽，工具无从得知。

**目标**：
- (a) `cutoff_hz` 可靠——落在真实频带边缘，而非 floor 区噪声尖峰。
- (b) 语义如实：`ok=False` = 「检出内容带宽上限（低于 Nyquist）」；文档说明它不区分"人工滤波"与"天然低频内容"。
- （可选）加 `expected_min_bandwidth` 参数让用户自设阈值，抑制低价值告警——把"是否缺陷"的判断交回用户，而非假装能自动区分。

**方案**：
1. **稳健化边缘定位**：PSD 中值平滑（跨若干 bin）消除单 bin 噪声尖峰 + 滞回（连续 K 个 bin 高于 `floor+threshold` 才认定越过），使 `cutoff_hz` 落在真实频带边界。
2. **局部陡度**（可选增强）：边缘 ±1 八度局部斜率用于 confidence，替代固定上 1/3。
3. **文档/输出**：docstring 与返回字段如实说明语义；`ok=False` = 「内容带宽上限低于 Nyquist」。

**涉及文件**：`analyzer/quality.py`（`_detect_high_freq_cutoff`）。

**✅ 已实现（M2）**：
- **稳健化边缘定位**：PSD 经 `scipy.signal.medfilt` 中值平滑（~250 Hz 窗口，奇数核）消除单 bin 估计尖峰；高→低扫描加滞回——连续 K 个（~150 Hz）平滑后 bin 高于 `floor+6 dB` 才认定越过边缘。`cutoff_hz` 由此落在真实频带边界。
- **局部陡度**：slope 改在检测到的边缘 ±~1 八度局部测量（替代固定上 1/3 ≥8kHz），confidence/slope/gibbs 保留为信息字段。
- **决策简化 + 语义如实**：`ok` 由稳健后的边缘位置驱动——`cutoff_hz < 0.85·nyq`（内容明确在 Nyquist 下方停止）即判 band-limited；docstring 如实说明它不区分"人工滤波"与"天然低频内容"。
- **before→after**（dur=1.0、内容+−18 dB 全带噪声底，跨 12 seed 取极值；旧代码 cutoff 在 5.8k–23.9k 乱跳且多数误判 ok=True）：

| 信号 | 旧 cutoff / ok | 新 cutoff / ok |
| --- | --- | --- |
| full-band | 24000 / True | 24000 / **True** |
| brickwall@15k | ~15064 / False | 14936–14941 / **False** |
| phone@3.4k | 5836–23877 / 多 True（漏报） | **3340（恒定）** / **False** |
| bass@5k | ~19752 / False | **4939–4945** / **False** |
| gentle LP@15k(4) | 22477 / True | 22477 / **True**（温和滚降不判） |

- **测试**：`TestHighFreqCutoff`（5 例，内容+噪声底、dur=1.0、固定 seed）。RED 验证：旧代码上 `test_phone_band_3k4`/`test_bass_only_5k` 失败（bass cutoff=19752 vs ~5000），新代码全绿。

**测试与验收**（cutoff 值贴近真实边缘；ok 反映"是否 band-limited below Nyquist"，基本不变）：
```
full-band noise        -> ok=True,  cutoff≈nyq
gentle LP@15k (4阶)    -> ok=True,  cutoff≈nyq（温和滚降无硬边）
sharp brickwall@15k    -> ok=False, cutoff≈15000±容差
phone band-limit@3.4k  -> ok=False, cutoff≈3400±容差   （值修正）
bass-only 80-5k        -> ok=False, cutoff≈5000±容差   （值修正；flag 合理，非 bug）
```

---

## F4 — DR 与 TT DR Meter 不符（P1）

**现状**：`_measure_dynamic_range` docstring 写 "Matches the TT DR Meter convention"，但实现是**无 K-weighting 的帧 RMS、~90ms 帧**取 P95-P10。TT DR Meter 用 R128 **short-term loudness（K-weighted，~1s 块）**百分位差。两者数值不一致（RMS 版通常偏高）。

**根因**：指标定义与文档声明不符。

**目标**：DR 与 EBU R128 一致、更接近 TT DR Meter，且文档如实。

**方案（首选 A）**：
- **A（推荐）**：复用 `_measure_loudness` 已算的 short-term loudness 数组，`DR = P95 - P10`（LU）。好处：与 R128 一致、更接近 TT、统一响度管线、且天然覆盖全混音声道（比当前「逐声道 max」更正确）。
- **B（备选，改动最小）**：保留 RMS 法，仅把 docstring 改为 "RMS-based dynamic range estimate（非 TT 合规）"。

**涉及文件**：`analyzer/quality.py`（`_measure_dynamic_range`、与 `_measure_loudness` 的 short-term 计算协调——算一次共享）。

**测试与验收**：
- AM 信号 → `dr > 0`；静音 → `dr == 0`。
- DR 现应随「响度范围」变化（与 LRA 同向），而非纯 RMS 离散度。
- docstring 与实际方法一致（A 或 B 任一，二选一落地）。

**✅ 已实现（M2）**：方案 A 落地。
- `_measure_dynamic_range(st_vals)` 改为取 R128 short-term loudness 数组的 P95-P10（LU），与 LRA 共用同一 `_loudness_percentile_range` helper → **DR ≡ LRA**、口径一致。
- 新增 `_short_term_loudness_values()`：全混音声道一次算出 3s 块 short-term loudness，供 LRA 与 DR 共享（昂贵的 meter 积分只做一遍）；`analyze_quality` 先算一次 `st_vals` 再分别喂给两者。
- **顺带修复**：原 LRA/DR 对全静音会因 `-inf` 算出 `nan`，现 `_loudness_percentile_range` 过滤非有限值（<3 个有限样本→0）。
- **实证**（12s 立体声、分块响度交替、R 通道主导）：DR = LRA = 24.1 LU；integrated = -4.7 LUFS（全混音，R 主导）；静音 → DR/LRA = 0；稳态正弦 → DR = 0。
- **注意**：3s 非重叠块 → 需 ≥ ~9s 音频才有 ≥3 个样本，更短则 DR=0（与现有 LRA 行为一致）。声道问题中 **DR 已改全混音**；`rms`/`zero_crossing`/`upsampling` 仍读 channel-0，并入 F7。

---

## F5 — 相位重分配：量纲/归一化与 Auger-Flandrin 不符（P2）

**现状**：`_reassigned_spectrogram` 声称 iZotope RX / Auger-Flandrin (IEEE TASSP 1995) 风格，但几处对不上标准公式：
- **归一化 ramp**：时间用 `linspace(-1,1,N)`、频率用归一化 bin index `(arange(n_fft)-n_fft/2)/(n_fft/2)`，均非物理 time/freq 单位 → correction 幅度被缩放常数倍。
- **time / freq ramp 归一化不一致**（一个按 `N`、一个按 `n_fft`），两个 correction 项尺度互不匹配。
- **量纲可疑**：`t_new = frame_idx + tau_corr * (hop/sr)`——把 `tau_corr·(hop/sr)` 加到帧索引上，单位对不上（若 tau_corr 是秒，帧数应是 `tau_corr/(hop/sr)`）。
- **snap 回粗 grid**：reassigned 坐标最后 `searchsorted` 吸附回原始粗频率网格，削弱亚 bin 锐化收益。

**目标**：要么让它数学上符合 Auger-Flandrin（并证明），要么如实降级为「锐化渲染模式」。首选前者。

**方案（验证先行）**：
1. **先建验证 harness**：线性 chirp + 若干已知频率纯音；断言重分配后能量落在真实瞬时频率脊线上（chirp 逐帧峰值跟踪其频率，误差 ≤ ±1 bin；tone 留在精确 bin）。对照参考 reassignment 实现。
2. **修量纲**：时间 ramp = 采样索引 n（或秒）、频率 ramp = 角频率（rad/sample）或 Hz，二者一致，使 `omega_corr`/`tau_corr` 符合 Auger-Flandrin；修正 tau 项单位使 `t_new` 正确落在帧坐标。
3. **亚网格重分配**：把能量累加到更细的网格（时/频各 4–8× 过采样）或插值，再降采样，真正兑现锐化收益，而非 snap 回粗 grid。

**涉及文件**：`analyzer/spectrum.py`（`_reassigned_spectrogram`）。

**测试与验收**：
- chirp 脊线跟踪误差 ≤ ±1 bin；tone 落在精确 bin。
- 视觉/数值锐化 ≥ 标准 STFT（同信号对比熵或峰值集中度）。
- 若验证后仍无法对齐参考 → 改 docstring 为「启发式锐化渲染」，不宣称 Auger-Flandrin。

---

## F6 — True Peak 内存峰值高（P3）

**现状**：`_true_peak` 对全文件做 `resample_poly(4×)` float64。10min/48k 立体声峰值 ~1GB+。

**方案**：分块上采样（如 1–2s 窗口），滚动维护全局 max，限制内存；或评估 float32 是否足够（true peak 精度需求）。

**涉及文件**：`analyzer/quality.py`（`_true_peak`）。
**验收**：长文件 true peak 结果不变，峰值内存显著下降（可加 `tracemalloc` 断言上界）。

## F7 — `_mono` 命名误导 + 声谱图只画左声道（P3）

**现状**：`self._mono = self.data[0]` 是 channel 0；STFT/声谱图可视化只用它 → 右声道独有瞬态在图上看不到。

> **范围补充（M2/F4 后确认）**：除可视化外，`analyze_quality` 里 `rms`、`zero_crossing`、`upsampling`（高频截止检测）也仍读 channel-0。这三项各自需定聚合语义（RMS/ZCR 取逐声道 max；截止检测取「最可能过采样」的声道/混音），非 F4 能顺手重定义，故并入本项统一处理。

**方案**：
- 重命名 `_mono` → `_first_channel`，或新增真正的 `_mixdown`（`(L+R)/2` 或 `sum/√n`）属性。
- 声谱图/STFT 可视化改用 mixdown（让 pan 内容可见）。此为「显示什么」的有意决策，落地时在 UI 层确认。

**涉及文件**：`analyzer/core.py`、`analyzer/spectrum.py`。
**验收**：仅右声道有内容的信号，声谱图可见其能量（当前不可见）。

## F8 — short-term loudness 非重叠块 / LRA 简化（P3）

**现状**：short-term 用非重叠 3s 块（R128 是 ~75% overlap 重叠窗）；LRA 简化为 short-term 的 P95-P10（非严格 R128 LRA）。

**方案**：
- short-term 改 R128 重叠窗（block 3–4s、step = 0.75×duration）。
- LRA：实现更贴近 R128 的算法，或明确文档为「近似值」。

**涉及文件**：`analyzer/quality.py`（`_measure_loudness`）。
**验收**：与 `pyloudnorm`/参考对一段已知响度变化的素材对比，偏差在可接受范围；docstring 如实。

## F9 — DR `as_strided` 极短音频越界 view（P3）

**现状**：`n < frame_len` 时 `as_strided(shape=(1, frame_len))` 构造越界 view（UB）。当前结果被丢弃不会崩，但属未定义行为。

**方案**：`n < frame_len` 时提前 `return 0.0`（或按实际长度处理），不构造越界 view。

**涉及文件**：`analyzer/quality.py`（`_measure_dynamic_range._dr_single_channel`）。
**验收**：极短音频（<1 帧）返回 0 且不触发越界；现有 DR 测试不回归。

---

## 交付顺序与依赖

- **M1 = F1 + F2**（P0）：互不依赖，可并行；先做，用户直接受益。
- **M2 = F4 + F3**（P1）：F4 方案 A 需让 `_measure_loudness` 暴露/返回 short-term loudness 数组供 DR 复用（当前内部算完即弃）；这与 F8 的重叠窗改进相互独立，不必先做 F8。F3 独立。
- **M3 = F5**（P2）：验证先行，独立。
- **M4 = F6 + F7 + F8 + F9**（P3）：小改动收尾。

## 风险与开放问题

1. **F1 LUFS 多声道布局**：`pyloudnorm` 对标准立体声/单声道无歧义；>2 声道（5.1 等）的 R128 增益需确认声道映射，落地时以双声道为主、多声道标注「按 R128 标准布局」。
2. **F2/F3 阈值调参**：F2 的 `MIN_RUN`、F3 的边缘陡度/K 滞回数需以边界矩阵实测校准，避免为过测试而过度拟合——用「语义正确」而非「刚好过线」定阈值。
3. **F5 是否值得 L 工作量**：若产品上 reassign 只是「好看」的渲染模式，可降级为方案 B（如实文档化），把 L 省下来。需产品判断。
4. **测试环境**：本机 `python`/`python3` 是商店占位符，CI/本地统一用 `py`；spec 内命令已按此写。
