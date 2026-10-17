# Partial sharing: a theorem interpolating between isolated prompts and a fully shared policy

Date: 2026-10-13. Written by Claude for Fan Yao after the objection that an LLM can condition on the prompt and need not pool preferences. Numerical companion: `partial_sharing_check.py` (NumPy, run with the Codex-runtime interpreter). Status: derivation and numerics by Claude; refereed adversarially by an independent Opus agent (`referee_report.md`, 2026-10-13); the ten issues it raised are applied in this version (marked [ref]).

## 0. Summary

A policy shared across prompts is modeled as a shared component plus prompt-specific components whose use costs ρ per unit (the cost summarizes how expensive prompt-conditional directions are: data sparsity per condition, the KL anchor, low-rank adapters). The exact restricted population round becomes a consensus-penalized KL projection. Results:

1. ρ = 0 is the paper's isolated model; ρ → ∞ is the common-role pooled model of result X1 (exact reduction). The family interpolates continuously.
2. The coupling is **state dependent and anisotropic**: at a stationary point, prompt a keeps a fraction K_a = (J_a + ρΠ)⁻¹J_a of its own deviation from the consensus, where J_a is its Fisher matrix. A prompt whose policy has concentrated on a few responses keeps most of its autonomy inside the support of its policy (eigenvalues of K_a up to (1/2)/(1/2+ρ)) and follows the consensus in the directions it never samples. Freezing J_a at the uniform value gives a scalar κ = (1/K)/(1/K + ρ); on the toy this surrogate loses stability already at κ ≈ 0.913 (ρ ≈ 0.032), about 60 times smaller than the exact threshold ρ* = 1.94 [ref]. The surrogate misjudges the coupling direction by direction: too strong along directions a prompt samples heavily, too weak along directions it rarely samples.
3. The round map's Jacobian at a nondegenerate fixed point is (𝓗_ρ)⁻¹ · blockdiag(w_a J_a DF_a) with 𝓗_ρ the penalized Hessian (implicit function theorem). Verified against finite differences to 1.7·10⁻¹⁰.
4. On the sparse three-prompt toy (α = 0.9, λ = 1) the spectral radius is a continuous, non-monotone function of ρ: mild sharing *stabilizes* (0.90 → 0.80 at ρ = 0.2), strong sharing destabilizes; the radius crosses 1 exactly once on the scanned range, at ρ* = 1.94 (κ* = 0.146) for β = 10 and ρ* = 1.30 (κ* = 0.204) for β = 20 [ref: single crossing confirmed by a grid scan over ρ ∈ [0.005, 40] and by branch continuation]. Simulations converge for ρ ≤ 1 and oscillate for ρ ≥ 2, matching the radii.

Interpretation: the cross-prompt instability of X3 is not an artifact of forcing one policy on all prompts; it appears as soon as prompt-conditional capacity is scarcer than a computable threshold, and stronger updates (larger β) need more conditional capacity to stay stable.

## 1. Setting

Common-role coordinates as in X1: K aligned roles, centered logits x_a ∈ 1^⊥ for prompt a = 1..N, weights w_a > 0 summing to 1, per-prompt ordinary maps F_a(x) = αx + b_a + gC_a softmax(x) (coverage term inside b_a), g = βλ, 0 ≤ α < 1. J(p) = Diag(p) − ppᵀ, Π = I − 11ᵀ/K.

**Policy class.** x_a = s + t_a with shared s ∈ 1^⊥ and prompt-specific t_a ∈ 1^⊥. The restricted population round with targets z_a = F_a(x_{a,t}) is
  (s, t) ← argmin Σ_a w_a KL(softmax(s + t_a) ‖ softmax(z_a)) + (ρ/2) Σ_a w_a ‖t_a‖²,  ρ ∈ [0, ∞].

**Lemma 1 (elimination of the shared component).** For fixed x = (x_a), min over s of Σ_a w_a‖x_a − s‖² is attained at s = x̄ := Σ_a w_a x_a. Hence the round is equivalently
  x ← argmin_x L_ρ(x; z),  L_ρ(x; z) = Σ_a w_a KL(softmax(x_a) ‖ softmax(z_a)) + (ρ/2) Σ_a w_a ‖x_a − x̄‖²,
with s = x̄ and t_a = x_a − x̄ at any minimizer.
*Proof.* Expand ‖x_a − s‖² and differentiate in s; the weighted average is the unique minimizer. Substituting gives the penalty Σ_a w_a‖x_a − x̄‖², which equals Σ_a w_a‖x_a‖² − ‖x̄‖², a quadratic form with matrix L_w ⊗ I, L_w = Diag(w) − wwᵀ (the weighted Laplacian, positive semidefinite with kernel the consensus direction). ∎

## 2. Theorem (partial sharing)

**(i) Limits.** (a) ρ = 0: the minimizers are exactly x_a = z_a for all a; the N prompts evolve by their isolated ordinary recursions. (b) ρ → ∞: for ρ ≥ ρ₀ (depending on z) minimizers exist and stay in a bounded set, and every limit point as ρ → ∞ is the consensus point x_a = z̄ := Σ_a w_a z_a for all a, i.e. the common-role round of X1. [ref: existence for small ρ > 0 is not claimed.]

**(ii) Stationarity and anisotropic coupling.** Every minimizer (indeed every critical point) satisfies, for each a,
  J(x_a)(x_a − z_a) + ρ(x_a − x̄) = 0  on 1^⊥.   (S)
Equivalently x_a = x̄ + K_a (z_a − x̄) with K_a := (J(x_a) + ρΠ)⁻¹J(x_a) (inverse on 1^⊥), whose eigenvalues are λ_j/(λ_j + ρ) ∈ (0, 1) for ρ > 0 (K_a = I at ρ = 0), where λ_j ∈ (0, 1/2] are the eigenvalues of J(x_a) on 1^⊥, and the consensus is fixed by Σ_a w_a K_a (z_a − x̄) = 0. In particular t_a = x_a − x̄ = −ρ⁻¹J(x_a)(x_a − z_a).

**(iii) Jacobian.** Let x* be a fixed point of the round map Φ_ρ (x* = Φ_ρ(x*)), e_a = x_a* − F_a(x_a*), T_a = T_{p_a*}(e_a) = Diag(J_a e_a) − (J_a e_a)p_a*ᵀ − p_a*(J_a e_a)ᵀ, and
  𝓗_ρ = blockdiag_a( w_a (J_a + T_a) ) + ρ L_w ⊗ I   (restricted to the centered stacked space (1^⊥)^N).
If 𝓗_ρ ≻ 0, then the strict-local-minimizer branch through x* given by the implicit function theorem is C^∞; defining Φ_ρ as that branch near x* [ref: uniqueness of the global inner minimizer is not proved, so the round map is this branch],
  DΦ_ρ(x*) = 𝓗_ρ⁻¹ · blockdiag_a( w_a J_a (αI + gC_aJ_a) ).
At ρ = 0 and at a fixed point of the isolated maps, e_a = 0, T_a = 0, and DΦ_0 = blockdiag(αI + gC_aJ_a): the paper's local spectrum, prompt by prompt.

**(iv) Crossing of the stability boundary.** Assume (a′) ρ ↦ x*(ρ), ρ ∈ [0, ∞), is a continuous branch of fixed points of the round-map branch of (iii) with 𝓗_ρ ≻ 0 along it; (b′) at ρ = 0 the branch consists of linearly stable isolated fixed points (r(0) < 1; e.g. robust winners with β > β₀ of X3 Theorem A, or verified numerically as on the toy, where each prompt has a tie and Theorem A does not apply [ref]); (c′) x*(ρ) converges as ρ → ∞ to a common-role fixed point (all coordinates equal) that is linearly unstable (β > β_J, X3 Proposition B). Then r(ρ) is continuous with r(0) < 1 < lim r(ρ), so {ρ : r(ρ) = 1} is nonempty; with ρ*₋ its infimum and ρ*₊ its supremum, partial sharing is locally stable for ρ < ρ*₋ and linearly unstable for ρ > ρ*₊. The radius need not be monotone, so several crossings are possible in general [ref]; on the toy there is exactly one.

### Proofs

*(i)(a).* L_0 is a weighted sum of KL divergences, each ≥ 0 with equality iff softmax(x_a) = softmax(z_a), iff x_a = z_a on 1^⊥.

*(i)(b).* Let κ_z := Σ_a w_a KL(softmax(z̄) ‖ softmax(z_a)) = Σ_a w_a lse(z_a) − lse(z̄) ≥ 0 (X1, logarithmic-opinion-pool lemma: Σ_a w_a KL(p‖softmax z_a) = KL(p‖softmax z̄) + κ_z for every p). Evaluating L_ρ at the consensus point gives L_ρ(z̄,…,z̄) = κ_z, so any minimizer x^ρ has (ρ/2)Σ_a w_a‖x_a^ρ − x̄^ρ‖² ≤ κ_z and deviations O(ρ^{-1/2}). Write q = softmax(z̄) and f(y) = KL(softmax(y)‖q) on 1^⊥; f vanishes only at y = z̄. *Compact sublevel sets* [ref]: as ‖y‖ → ∞ on 1^⊥, min_i softmax(y)_i → 0 uniformly, so softmax(y) enters the compact set B_ε = {p ∈ Δ : min_i p_i ≤ ε}; on B_ε the continuous function p ↦ KL(p‖q) is positive (q is interior) and attains a positive minimum m(ε), with m(ε) → −log(1 − q_min) as ε → 0, q_min = min_i q_i [ref: along a ray the limit is −log q(S) for the argmax face S, so the sharp threshold is −log(1−q_min), not min_i log(1/q_i)]. Hence for η < m(ε) the sublevel set {f ≤ η} is contained in a ball, i.e. compact. *Limit*: by uniform continuity of y ↦ KL(softmax(y)‖q′) on bounded sets (softmax is 1/2-Lipschitz and p ↦ KL(p‖q′) is continuous on the closed simplex), Σ_a w_a KL(softmax(x_a^ρ)‖softmax z_a) ≥ f(x̄^ρ) + κ_z − ε(δ_ρ), δ_ρ = max_a‖x_a^ρ − x̄^ρ‖ → 0; with L_ρ(x^ρ) ≤ κ_z this gives f(x̄^ρ) ≤ ε(δ_ρ) → 0, so x̄^ρ stays in a compact sublevel set and converges to z̄. *Existence and boundedness for ρ ≥ ρ₀*: choose ρ₀ with ε((2κ_z/ρ₀)^{1/2}) < m(ε₀) for a fixed ε₀; then {L_ρ ≤ κ_z} is bounded (deviations and consensus both bounded by the two inequalities), hence compact, and the continuous L_ρ attains its minimum on it, which is global since every point outside has L_ρ > κ_z. ∎

*(ii).* ∂L_ρ/∂x_a = w_a[J(x_a)(x_a − z_a) + ρ(x_a − x̄)] (the gradient of KL(softmax(x)‖q) is J(x)(x − log q + const), X1 Prop 3; the penalty gradient is ρ w_a (x_a − x̄) because Σ_b w_b ∂(x_b − x̄)/∂x_a contributions cancel: ∂/∂x_a Σ_b w_b‖x_b − x̄‖² = 2w_a(x_a − x̄) − 2w_a Σ_b w_b(x_b − x̄) = 2w_a(x_a − x̄)). Setting it to zero gives (S). On 1^⊥, J(x_a) is positive definite for interior policies (lem:softmax), so J + ρΠ is invertible for ρ ≥ 0 and (S) rearranges to x_a = (J+ρ)⁻¹(J z_a + ρx̄) = x̄ + (J+ρ)⁻¹J(z_a − x̄). The eigenvalue claim is the spectral mapping λ ↦ λ/(λ+ρ); λ_max(J(p)) ≤ 1/2 because vᵀJ(p)v = Var_p(v) ≤ (max v − min v)²/4 ≤ ‖v‖²/2 for centered v. Summing w_a(x_a − x̄) = 0 gives the consensus equation. ∎

*(iii).* Stationarity defines Φ_ρ implicitly through G(x; x_prev) := ∇_x L_ρ(x; F(x_prev)) = 0, with ∂G/∂x = 𝓗_ρ (the Hessian of KL(softmax(x_a)‖softmax z_a) is J_a + T_{p_a}(x_a − z_a), X1 Prop 3; the penalty Hessian is ρL_w ⊗ I) and ∂G/∂x_prev = −blockdiag(w_a J_a DF_a(x_{a,prev})) (chain rule through z_a = F_a(x_{a,prev}): ∂/∂z_a of J(x_a)(x_a − z_a) is −J(x_a)). At a fixed point the gradient vanishes; if 𝓗_ρ ≻ 0, the implicit function theorem gives a C^∞ branch with DΦ_ρ = −(∂G/∂x)⁻¹ ∂G/∂x_prev, which is the formula; positivity of the Hessian makes the branch the unique strict local minimizer nearby (as in X5 Theorem 1). At ρ = 0, fixed points are isolated fixed points with e_a = 0, so T_a = 0 and 𝓗_0 = blockdiag(w_aJ_a), giving DΦ_0 = blockdiag(J_a⁻¹J_a DF_a) = blockdiag(DF_a). ∎

*(iv).* Along the branch, ρ ↦ DΦ_ρ(x*(ρ)) is continuous (𝓗_ρ ≻ 0 and its inverse vary continuously), and so is the spectral radius; r(0) < 1 by (b′) and (iii). For the limit [ref: the terms w_aT_a do not vanish individually; they cancel only after projection]: split the stacked centered space into the consensus subspace V_c = {(v,…,v)} and the deviations V_d; the penalty ρL_w ⊗ I is ρΛ with Λ ≻ 0 on V_d and 0 on V_c. Writing A = blockdiag(w_a(J_a+T_a)) in this decomposition, the Schur complement gives 𝓗_ρ⁻¹ → A_cc⁻¹ ⊕ 0 as ρ → ∞, where A_cc = Σ_a w_a(J_a+T_a) on V_c ≅ 1^⊥ (A_cc ≻ 0 along the branch since 𝓗_ρ ≻ 0). At the common-role limit all x_a* coincide, so J_a = J̄* and the residuals e_a = x̄* − F_a(x̄*) satisfy Σ_a w_a e_a = x̄* − F̄(x̄*) = 0; T_p(e) is linear in e, so Σ_a w_a T_a = 0 and A_cc = J̄*. Hence DΦ_ρ(x*(ρ)) converges to the matrix with consensus block J̄*⁻¹ Σ_a w_a J̄*(αI + gC_aJ̄*) = αI + gC̄J̄*, the common-role Jacobian, and zero deviation block; its radius exceeds 1 by (c′). The intermediate value theorem gives at least one crossing; the first and last define ρ*₋ ≤ ρ*₊. ∎

## 3. Remarks

- **Why the uniform-Fisher surrogate misleads.** Freezing J_a ≡ Π/K turns (S) into x_a = z̄ + κ(z_a − z̄) with scalar κ = (1/K)/(1/K + ρ). In that surrogate the consensus coordinate obeys x̄_{t+1} = Σ_a w_a F_a(x_{a,t}) for every κ, so the pooled cyclic feedback is not attenuated by κ; on the toy its radius is continuous in κ and crosses 1 at κ ≈ 0.913 (ρ ≈ 0.032): 0.9817 at κ = 0.95, 0.9973 at 0.92, 1.0013 at 0.91, 1.0047 at 0.90 [ref]. The exact coupling K_a = (J_a + ρ)⁻¹J_a is anisotropic: a prompt concentrated on the responses it was labeled on keeps most of its autonomy inside their span (eigenvalues up to (1/2)/(1/2+ρ)) and follows the consensus in the directions it rarely samples (λ_j ≈ 0); relative to the surrogate the exact coupling is weaker in the former directions and stronger in the latter. A heuristic, not a theorem: the cyclic rotation has to pass through the rarely sampled directions, where the exact round pins each prompt to the consensus, which is why its threshold is sixty times larger.
- **Non-monotonicity.** On the toy the radius first falls (0.90 → 0.80 at ρ = 0.2) before rising: mild sharing acts as damping, since the consensus mixes prompts whose cyclic modes are out of phase. This is the X2 cancellation effect appearing inside a partially shared model.
- **What ρ is in an LLM.** ρ is the marginal cost of moving a prompt-conditional direction relative to the shared one. Heuristically, with features [shared style directions; prompt-group indicator × style], the shared direction receives the pooled gradient of all N prompts while each conditional direction receives one prompt's, so the effective ρ grows with N and with the KL anchor and shrinks with adapter capacity per group. Measuring it is the experiment: gradient overlap across prompts on the labeled pairs (JVP on LoRA parameters), or shared LoRA versus per-group adapters.
- **Scope.** Exact population inner optimization; common-role coordinates; nothing about sampled losses or nonlinear features beyond X5's remarks. The theorem's part (iv) assumes a continuous nondegenerate fixed-point branch; on the toy it was verified numerically but not proved.

## 4. Numerical verification (toy: K = 3, N = 3, labeled pairs concise≻detailed, detailed≻confident, confident≻concise, flow 0.3, α = 0.9, λ = 1; `partial_sharing_check.py`)

Exact penalized round (Newton inner solve to 10⁻¹³; outer fixed point by Newton on the exact round map; Jacobian by implicit function theorem (iii)):

| ρ | κ_eq = (1/3)/(1/3+ρ) | spectral radius | prompt-1 policy at fixed point | simulated tail (last 200 of 1500 rounds) |
|---|---|---|---|---|
| 0 | 1.000 | 0.9000 | (0.50, 0.00, 0.50) | converged |
| 0.02 | 0.943 | 0.8909 | (0.65, 0.02, 0.32) | converged |
| 0.05 | 0.870 | 0.8734 | (0.69, 0.04, 0.27) | converged |
| 0.1 | 0.769 | 0.8326 | (0.67, 0.07, 0.26) | converged |
| 0.2 | 0.625 | 0.7966 | (0.62, 0.11, 0.28) | converged |
| 0.3 | 0.526 | 0.8044 | (0.57, 0.14, 0.29) | converged |
| 0.5 | 0.400 | 0.8498 | (0.51, 0.18, 0.31) | converged |
| 1 | 0.250 | 0.9362 | (0.44, 0.24, 0.32) | converged |
| 2 | 0.143 | 1.0020 | (0.39, 0.28, 0.33) | oscillates, ‖x‖ ∈ [0.86, 0.93] |
| 5 | 0.062 | 1.0438 | (0.36, 0.31, 0.33) | oscillates, [1.37, 1.72] |
| 20 | 0.016 | 1.0632 | (0.34, 0.33, 0.33) | oscillates, [1.46, 1.83] |

Threshold by bisection on the analytic radius: **ρ* = 1.9435 (κ* = 0.1464) at β = 10**, prompt-1 policy there (0.39, 0.28, 0.33); **ρ* = 1.3013 (κ* = 0.2039) at β = 20**. The referee's independent continuation reproduces 1.94353 and 1.30127, finds the branch continuous with 𝓗_ρ ≻ 0 down to ρ = 10⁻⁶ (logits growing like log(1/ρ)), exactly one crossing on ρ ∈ [0.005, 40] at each β, and the large-ρ limit 1.0693 [ref]. Implicit-function Jacobian versus central finite differences of the exact round map: max entry difference 1.7·10⁻¹⁰ (ρ = 0.1), 6.3·10⁻¹¹ (ρ = 1), 7.4·10⁻¹¹ (ρ = 5). Uniform-Fisher surrogate for comparison: radius 0.9817 (κ = 0.95), 0.9973 (0.92), 1.0013 (0.91), 1.0047 (0.90), rising to 1.0693 at κ = 0; its threshold is κ ≈ 0.913, i.e. ρ ≈ 0.032 [ref].

Limits check: ρ = 0 reproduces the isolated fixed points and radius 0.9 (X3-type); ρ = 20 reproduces the common-role radius 1.069 and tail [1.46, 1.83] of the pooled toy (FACTS of the toy deck).


---
Shared-copy note (2026-10-13): read `../EXECUTION_CLARIFICATIONS.md` before execution. It takes precedence for statistical tests, attribution and scope. This is a candidate research plan/background, not a new training authorization or certified manuscript theorem. Source-local companion scripts mentioned in background are not included in this plan bundle.
