# 跨 prompt 共同 latent preference：定义、估计与共享 cyclic component 的识别（方案 v2，Fan 已定四项）

日期：2026-10-13。状态：v2，已按 Fan 的决定更新（特征两套都做、embedding 先出；S4 的 J_a 两种都报；第一轮只做纯 latent 拟合、ρ 扫描留第二轮；identity 与 logit 两种 link 都算）。第 4 点（latent cyclic oracle）未授权，保留为条件性后续。Handoff 见同目录 `HANDOFF_TO_YU_HE_2026-10-13.md`。依据：10-13 会议纪要（Idea A/B、P0 todo）、latent-preference note（Prop L1–L2、Thm L3）、partial-sharing note、hodge_diagnostics 的 09-25 报告与 `hodge/coherence.py`、LLM_Sampling 的 oracle/非 oracle 协议。

## 0. 一句话

把"跨 prompt 共同 latent preference"定义为特征空间上的一个反对称双线性形式 Ā（Prop L1），用 pooled 的 pairwise label 以 skew-bilinear Bradley–Terry 估计它（Prop L2），共享的 cyclic component 就是 Ā 的 attribute 块 Ā_cyc 在 latent Fisher 度量下的主旋转平面（Thm L3 的 G^{-1/2} S G^{-1/2} 的主特征对）；用 within-prompt shuffle 与 cross-fitting 防止"最大化 cycle 选到噪声"；然后把已存 checkpoint 的 logits 投影到同一 latent 坐标上看它是否沿这个平面旋转、极化或静止。

## 1. 定义：什么叫"共同的 latent preference"

设每个 response y 有特征 φ(y) = (ψ(y), 1) ∈ R^m，ψ ∈ R^{m−1} 是与 prompt 无关的属性坐标（简洁/详尽/自信……，或 embedding 的低维投影）。

- **Latent preference**：反对称矩阵 Ā ∈ Skew_m；对任意两条回答（无论来自哪个 prompt），latent flow 为 A(y, y′) = φ(y)ᵀ Ā φ(y′)。
- **Latent Hodge 分解**（Prop L1）：Ā 的 intercept 行 = latent potential ū_lat（一个线性打分函数），attribute 块 = latent cycle Ā_cyc。每个 prompt 的 panel 流 A_a = Φ_a Ā Φ_aᵀ，其 potential u_a = Φ̃_a(ū_lat + Ā_cyc ψ̄_a)，其 cycle C_a = Φ̃_a Ā_cyc Φ̃_aᵀ，**只依赖 Ā_cyc**；K_a = 2 时 C_a ≡ 0。
- **Prompt-specific 部分**：A_a = Φ_a Ā Φ_aᵀ + E_a，E_a 是不能被共享特征解释的残差（正确性、语境）。会议里说的 global + specific 就是这个分解，"相加"发生在 flow（Ψ(P) − Ψ(1/2)）的尺度上，不是概率尺度。
- **Gauge**：平移属性原点 ψ → ψ + t 把 (ū_lat, Ā_cyc) 变成 (ū_lat − Ā_cyc t, Ā_cyc)，所有 A_a 不变。所以"每个 prompt 的质心"和"latent potential"是同一个东西；只有 Ā_cyc 是 gauge 不变的，这正是我们要找的共享对象。

**为什么这是正确的聚合坐标。** 会议反对"直接平均 response-index 矩阵"是对的：index 没有跨 prompt 含义。Prop L1 说明正确的聚合是把每个 panel 的 flow 通过它自己的 Φ_a 拉回到 Skew_m，这恰好是最小二乘的 normal equation（见 §2），不是任何手工对齐。

## 2. 估计：skew-bilinear BT（Prop L2 的 GLM 形式）

数据：pooled 比较 (a, i, j, label)，每条 label ∈ {0,1} 或软概率，来自任何 prompt、任意稀疏。

- **模型**：P(i ≻ j | a) = σ(φ_iᵀ Ā φ_j) = σ(⟨Ā, M_ij⟩)，M_ij = (φ_iφ_jᵀ − φ_jφ_iᵀ)/2。参数是 Skew_m 的 m(m−1)/2 个自由度；是一个标准 logistic 回归，设计行为 vec(M_ij)。恒等 link 时是对 label 的最小二乘。
- **识别**（Prop L2，referee 修正后）：Ā 在 span{M_ij} 上可识别；全识别 ⇔ span = Skew_m。one-hot 角色 + intercept 会把 cycle 与 potential 混淆（每条观测 = Ā_rs + ū_r − ū_s），**所以特征必须是连续的或至少在同一属性上跨 prompt 变化**；dense 特征下 60 个 prompt 各 1 对即可到机器精度。
- **与 prompt-specific 残差分离**：拟合 full 模型 σ(φ_iᵀĀφ_j + ε_{a,ij})，ε 反对称、每 prompt 一块，用 group-lasso/ridge 罚 ε（这就是 partial-sharing 的 ρ）；ρ 扫描给出"共享部分能解释多少 flow 能量"的曲线。ρ→∞ 就是纯 latent 模型。
- **Ā_cyc 的取法**：拟合后取 attribute 块；为避免 gauge 伪影，属性先按 pooled 均值中心化（这只改 ū_lat，不改 Ā_cyc）。

## 3. 找"共享的 cyclic component"：三个层次

1. **静态能量**：‖Ā_cyc‖_F² / ‖Ā‖_F²（latent cyclic share），以及每个 prompt 暴露的 share ‖C_a‖²/‖A_a‖²。K=2 的 panel share 为 0，但它们仍然贡献 Ā 的估计（这是 Appendix D 的反转）。
2. **动力学相关的方向**（会议 Idea B 的 PCA 类比的正确版本）：Thm L3 说共享策略的乘子由 S_eff = G^{-1/2} S G^{-1/2} 决定，G = Σ w_a Φ̃_aᵀ J_a Φ̃_a 是 latent Fisher，S = Σ w_a G_a Ā_cyc G_a（共 Fisher 时 S = G Ā_cyc G）。S_eff 是实反对称阵，特征值 ±iω̃_j；**最大的 ω̃_1 和它的 2 维不变平面就是"最一致的循环方向"**，latent frontier 是 α² + (βλ ω̃_1)² < 1。这不是 PCA（PCA 找对称阵的主方向），而是反对称阵的主旋转平面；它自动按各 prompt 的采样权重 J_a 加权，所以"某属性在很多 prompt 上都有变化且策略在那里还有质量"才算数。
3. **可解释投影**：把主旋转平面投到人类属性（简洁/详尽/自信）上看载荷；若载荷集中在 2–3 个属性，就能用会议 Idea A 的语言描述（"模型在简洁与自信之间来回"）。

**防止选到噪声**（会议明确担心）：

- 空分布：within-prompt label shuffle（保留每个 |C_a| 与 label 集合，只打掉跨 prompt 对齐；09-25 报告已用）+ 随机翻转每个 prompt 的 orientation（X4 的随机方向模型，期望 Ā_cyc = 0）+ transitive surrogate（每 panel 单一 BT 分数过同一 link 后重估，给出纯传递序因离散化/link 产生的 ω̃_1 基线；09-25 已用）。
- Cross-fitting：在 prompt 的一半上估 Ā_cyc 与主平面，在另一半上算该平面的 ω̃ 与 cyclic share；只报告 held-out 值。
- Bootstrap over prompts 给 ω̃_1 的置信区间。

## 4. 与 Yu He 现有实验的衔接

现状（会议报告，未核验）：500 个 HelpSteer prompt × 4 responses，两个 oracle 的 BT 概率混合（等权或 0.6/0.4），只 1 个显式序环；打印的 ‖C‖_2 与曲线波动没有明显对应；每轮 checkpoint 已存；per-step win rate 未算。

09-25 报告里已经有一个 one-hot 版本：`hodge/coherence.py` 用 model identity（UltraFeedback）或 length rank（HelpSteer）作共享特征，算 G、S、ω̃，并有 shuffle null 与精确共享动力学。结论是对齐真实但小（ω̃ ≈ 0.014–0.018 对 per-prompt RMS 0.10–0.18），且主要由 label 离散化产生。**这个工具就是 L3 的 K=one-hot 特例，直接可扩展到连续特征。**

具体步骤（全部 CPU，不需要新 GPU）：

- **S0 固定数据对象。** 从 Yu He 的 500 panel 导出 (prompt, i, j, P_ij) 表 + 两个 oracle 各自的 P_ij（不混合）+ 混合权重；记录 link（identity=IPO、logit=DPO 两种都算）。区分三个东西：oracle 偏好结构（固定）、每轮生成样本上的估计矩阵（随 checkpoint 变）、模型风格分布（随 checkpoint 变）。
- **S1 特征。** 三套并行：(a) 可解释属性：用一个 judge（Nemotron 或 GPT 类）给 2000 条 response 打 简洁/详尽/自信/礼貌/结构化 等 5–8 维连续分；(b) 数据驱动：response embedding 的 PCA 前 8–16 维；(c) 现成：长度、长度 rank、句数（已有）。属性 pooled 中心化。
- **S2 估计 Ā。** 对每套特征、每个 link、每个 oracle 单独及混合：skew-bilinear BT；输出 Ā_cyc、latent cyclic share、ρ 扫描的共享解释比例。
- **S3 共享 cyclic 方向。** 在 uniform 策略与 initial checkpoint 的实际策略两处算 G、S_eff、ω̃_1 及主平面，附 shuffle/flip 空分布与 cross-fit 值；算 latent frontier β*=√(1−α²)/(λω̃_1) 并与实验用的 β（DPO β=0.1 ⇒ 10；IPO τ）对照。**判据**（已精确化）：主策略取 initial checkpoint 的实际 panel 概率（uniform 为次），null（within-prompt shuffle、orientation flip）与 transitive-surrogate 对照以及 ω̃_1 都在 cross-fit 的 held-out 半边上算（各 500 次，95 分位）；β 范围取该 run 每种 link 实际用的 β（DPO β = 1/β_train，IPO β = 1/τ）。若 held-out ω̃_1 高于两个 null 的 95 分位和 transitive surrogate、且 β* 落在该范围内，说明当前数据有可测的共同循环信号；否则当前构造本来就不会显示循环（会议的条件性解读）。
- **S4 checkpoint 投影（Idea A 的 metric）。** 对每个 checkpoint t、同一 prompt 集：用 logits x_a^{(t)} 做 L3 的受限回归 θ^{(t)} = argmin Σ_a w_a ‖Φ̃_a θ − x̃_a^{(t)}‖²_{J_a}（这是 latent 坐标下的策略），再投到 S3 的主平面得 (p_t, q_t)。三种结局可分辨：旋转（相位单调推进、半径有界）、极化（半径发散、相位冻结）、静止/噪声（半径小、相位随机）。同时报告 S4 残差 ‖x̃_a − Φ̃_a θ‖ 的占比，说明 latent 坐标解释了多少策略运动。这一步不需要重新评分，只需要 checkpoint 上 4 个 response 的 logprob。
- **S5 两个 oracle 的分解。** 分别估 Ā^{(1)}, Ā^{(2)} 与混合，两种 link 都算。单个 BT oracle 在 logit link 下残差为零、identity link 下为正（09-25 报告）：单 oracle 在 identity 下有 Ā_cyc 而 logit 下没有 ⇒ link mismatch；单 oracle 在 logit 下就有 ⇒ 该 oracle 自身在特征坐标下不是 BT 传递的；只有混合在 logit 下有 ⇒ 两个 oracle 的异质性。再加 transitive-surrogate 对照（每 panel 拟合单一 BT 分数，过同一 link 后重估 Ā）剔除离散化/link 伪影。这直接回答"cycle 从哪来"。（2026-10-13 文字报告校对后修正）

## 5. 预期与决策树

- 若 S3 信号显著且 S4 的主平面相位推进：这是 Paper B Section 7 最强的实证层，latent 框架写进正文。
- 若 S3 信号显著但 S4 静止：按 partial-sharing 定理，模型容量（ρ 小）把循环"各存各的"；可用 LoRA 跨 prompt 梯度重叠估计实际 ρ，并设计 shared-LoRA vs per-group adapter 对照。
- 若 S3 信号不显著：当前两 oracle 构造不产生共同循环；要么换成有已知 Ā_cyc 的 latent cyclic oracle（在属性空间定义 Ā，对所有 prompt 一致，是 cyclic-history pilot 的连续版），要么接受 HelpSteer 类数据是"concentration without oscillation"的负对照。

## 6. Fan 的决定（2026-10-13）

1. 特征：可解释属性与 embedding 两套都做，embedding 先出结果。
2. S4：initial-checkpoint 的 J_a 与逐 checkpoint 的 J_a 都报。
3. 第一轮只做 ρ→∞ 的纯 latent 拟合；ρ 扫描（共享/私有分解）放第二轮。
4. latent cyclic oracle（新训练）：未授权；只有 S3 不显著时再议。
5. identity（IPO）与 logit（DPO）两种 link 都算。

## 7. 不在本方案内

不改训练流程；不新开 GPU；不把 09-25 的 one-hot 结果当作本方案结果；Ā 的估计误差对动力学的影响（Ā 视为已知）留作后续。


---
Shared-copy note (2026-10-13): read `EXECUTION_CLARIFICATIONS.md` before execution. It takes precedence for statistical tests, attribution and scope. This is a candidate research plan/background, not a new training authorization or certified manuscript theorem. Source-local companion scripts mentioned in background are not included in this plan bundle.
