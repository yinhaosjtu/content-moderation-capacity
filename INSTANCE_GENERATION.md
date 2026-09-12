# Instance Generation Methodology

**Last Updated:** September 2026  
**Paper Reference:** "Joint Capacity Commitment and Automated Detection Configuration for Content Moderation under Correlated Operational Uncertainty"

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Public Data Anchors](#public-data-anchors)
4. [Parameter Specification](#parameter-specification)
5. [Structural Validation](#structural-validation)
6. [Reproducibility](#reproducibility)

---

## Overview

This document describes how the test instances used in our study are constructed. While the instances do not use proprietary platform data, they are **structurally grounded** in publicly documented moderation patterns and **parametrically anchored** to ranges disclosed in platform transparency reports.

### Why Structured Instances?

Real content moderation data is confidential and subject to strict privacy and security constraints. However, we can construct **realistic test cases** by:

1. **Extracting structural patterns** from public enforcement disclosures (nested qualifications, inspection duties, state-dependent outsourcing)
2. **Anchoring parameter ranges** to publicly reported operating conditions (traffic volumes, handling times, error rates)
3. **Validating mechanisms** against documented platform behaviors (correlation patterns, capacity constraints, compliance requirements)

This approach produces instances that are:
- ✅ **Operationally realistic**: parameters span documented ranges
- ✅ **Structurally faithful**: reflect observed moderation architectures
- ✅ **Fully reproducible**: every instance can be regenerated from seeds
- ✅ **Transparent**: all generation rules are open-source

---

## Design Philosophy

### What We Model

Our instances capture the **structural mechanisms** that govern joint capacity-detection planning:

| Mechanism | Source | How We Model It |
|-----------|--------|-----------------|
| **Nested qualifications** | Platforms organize moderators into skill tiers (Meta CSER 2025, Roberts 2019) | Three-tier hierarchy with category-dependent minimum qualification |
| **Inspection duties** | DSA Article 35 mandates internal quality checks | Senior tier performs inspections at mandated rates |
| **State-dependent outsourcing** | BPO providers raise prices or reduce capacity during demand spikes (Gillespie 2018) | Outsourced availability contracts with demand and detection shocks |
| **Detection-capacity coupling** | Stricter thresholds send more content to review (Gorwa et al. 2020) | Operating points jointly determine automated errors and review workload |
| **Multilingual operations** | Platforms operate across 100+ languages (TikTok CGER Q1 2026) | Four language pools with heterogeneous resources |
| **Policy categories** | Platforms enforce 6-8 major policy areas (Meta CSER 2025) | Six categories with severity- and expressiveness-indexed parameters |

### What We Abstract

Our instances intentionally simplify:

- **Backlog dynamics**: We assume daily clearing. Real platforms manage multi-day queues.
- **Reviewer attrition**: We fix headcount. Real platforms face turnover and burnout.
- **Strategic user behavior**: We treat item distributions as exogenous. Real users adapt to enforcement.
- **Detector retraining**: We model detection quality as a shock. Real platforms continuously retrain models.

These abstractions **do not invalidate** the core mechanisms we study. Joint planning value, correlation effects, and VSS > EVPI are structural results that survive these simplifications.

---

## Public Data Anchors

Our parameter ranges are **anchored to publicly disclosed values**. Below we document the sources and how we use them.

### 1. Traffic Volumes

**Source:** TikTok Community Guidelines Enforcement Report (CGER) Q1 2026

| Metric | Disclosed Value | How We Use It |
|--------|----------------|---------------|
| Global video removals | 184 million in Q1 2026 | ≈ 2.04M removals/day |
| Removal rate | 0.5% of uploads | Implies ≈ 408M uploads/day globally |
| Automation rate | 96.7% | Informs our L0/L1 escalation rates |

**Our baseline traffic ladder:** 8.0×10⁶ to 1.2×10⁵ items/day across four language pools.  
**Justification:** Scales down from TikTok's global volume to a representative multi-language platform. The geometric spread (factor of ~67 between largest and smallest pool) reflects observed disparities in language-pool resources (English vs. low-resource languages).

### 2. Handling Times

**Source:** Industry surveys (Steiger et al. 2021, Roberts 2019)

| Content Type | Reported Range | Our Range |
|--------------|----------------|-----------|
| Simple policy violations (spam, nudity) | 10-30 seconds | 15-45 seconds |
| Context-dependent (hate speech, harassment) | 30-90 seconds | 45-120 seconds |
| High-severity (terrorism, CSAM) | 60-180 seconds | 90-240 seconds |

**Justification:** Our ranges include the reported values plus a safety margin for translation, note-taking, and escalation.

### 3. Detection Quality

**Source:** Academic audits (Jiang et al. 2023, Bandy & Diakopoulos 2021)

| Detector Type | Reported Accuracy | Our Modeling |
|---------------|------------------|--------------|
| NSFW/Adult content | 85-95% | Δ⁰ = 0.8-1.2, Δ¹ = 1.2-1.8 (ROC separation) |
| Hate speech | 60-80% | Δ⁰ = 0.4-0.7, Δ¹ = 0.8-1.2 |
| Misinformation | 50-70% | Δ⁰ = 0.3-0.6, Δ¹ = 0.6-1.0 |

**Our detection interface:** We model detectors through latent-score ROC separations rather than point accuracies. This abstracts from specific model architectures while preserving the **trade-off between precision and recall** that threshold selection governs.

### 4. Outsourcing Patterns

**Source:** Gillespie (2018), Roberts (2019), BPO industry reports

| Observation | Source | How We Model It |
|-------------|--------|-----------------|
| Outsourced capacity contracts during crises | Gillespie 2018 | $\bar{v}_{ld\omega} = \bar{v}^0_{ld}\,(1 + \zeta_1 (Z_{d\omega}-1)^{+} + \zeta_2 \delta_{d\omega})^{-1}$ |
| BPOs charge 20-40% premium for English | Industry surveys | Wage multipliers by language pool |
| Outsourcing limited to frontline work | Roberts 2019 | Only tier-1 capacity can be outsourced |

**Justification:** The exponential contraction formula captures the documented pattern that third-party capacity becomes scarce precisely when platforms need it most (high traffic + detector degradation).

### 5. Compliance Requirements

**Source:** EU Digital Services Act (DSA) Article 35

| Requirement | DSA Provision | Our Implementation |
|-------------|---------------|-------------------|
| Internal oversight | Article 35(1) | Senior tier performs inspections |
| Sampling rate | Not specified | $\underline{\eta}^{\mathrm{insp}}_{lk} \sim \mathrm{U}(0.02, 0.08)$ |
| Content retention | Article 34(3) | Not modeled (orthogonal to planning) |

**Justification:** The DSA mandates internal quality checks but does not prescribe rates. We set inspection floors in the 2-8% range, consistent with industry practice (Meta CSER 2025).

### 6. Error Costs

**Source:** Meta CSER (2025), academic literature on moderation harms

| Error Type | Qualitative Impact | Our Cost Range |
|------------|-------------------|----------------|
| False negative (missed violation) | User harm, regulatory risk | $1.5 - 200$ per item (severity-indexed) |
| False positive (wrongful removal) | User frustration, speech chilling | $3 - 22$ per item (expressiveness-indexed) |

**Justification:** These are **not actual platform costs** (which are unknowable). They are **decision weights** that reflect the relative severity of different error types, indexed by policy category. High-severity categories (terrorism) have much higher FN costs than low-severity ones (spam).

---

## Parameter Specification

This section documents the complete generation rules. Mathematical notation follows the paper.

### Instance Dimensions

| Tier | $|L|$ | $|C|$ | $|K|$ | $|D|$ | $|\mathcal{P}|$ | $|\Omega|$ | $n_{\mathrm{disc}}$ | $n_{\mathrm{block}}$ |
|------|-------|-------|-------|-------|----------------|-----------|---------------------|---------------------|
| Small | 4 | 4 | 2 | 7 | 6 | 400 | 20 | 2,800 |
| Medium | 5 | 4 | 3 | 8 | 10 | 400 | 35 | 4,000 |
| Large | 6 | 6 | 3 | 14 | 8 | 400 | 46 | 5,600 |

**Policy categories:** spam/manipulation, adult content, harassment/bullying, hate speech, violence/incitement, terrorism/extremism.  
**Free vs. forced dispositions:** Categories 1-4 are free ($C^{\mathrm{fr}}$); terrorism→forced-remove ($C^{\mathrm{fc}}$), hate speech→forced-outsource ($C^{\mathrm{fo}}$).

### Volume and Scope

| Parameter | Generation Rule | Justification |
|-----------|----------------|---------------|
| $\bar{a}_{ld}$ | Geometric ladder from $8.0\times10^6$ to $1.2\times10^5$ items/day, with lognormal jitter $\exp\{\mathcal{N}(0, 0.08^2)\}$ and deterministic weekly/seasonal factors | Anchored to TikTok CGER Q1 2026 global scale, scaled down to representative platform |
| $\psi_{lc}$ | $\sum_c \psi_{lc} \sim \mathrm{U}(0.25, 0.55)$, split across categories by severity-weighted Dirichlet | Reflects that 0.5% removal rate (TikTok) suggests < 1% prevalence of violations |
| $r_l$ | $r_l = (\log \bar{a}_l - \log \bar{a}_{\min}) / (\log \bar{a}_{\max} - \log \bar{a}_{\min}) \in [0,1]$ | Resource index linking pool scale to detector quality and reviewer accuracy |

### Detection Coefficients

| Parameter | Generation Rule | Justification |
|-----------|----------------|---------------|
| $\Delta^0_{lc}, \Delta^1_{lc}$ | Layer separations increasing in $r_l$, decreasing in $s_c$; $\Delta^1 > \Delta^0$ | High-resource pools get better detectors; high-severity content is harder to detect |
| $\pi^0_{lc}$ | Log-interpolated from ≈6% (low severity) to ≈0.4% (high severity), confined to [0.001, 0.150] | Consistent with 0.5% platform-wide removal rate |
| $\rho_{lc}$ | Score correlation $\sim \mathrm{U}(0.35, 0.75)$ | L0 and L1 detectors share features but L1 adds refinement |
| $\kappa_{lc}, b_{lc}$ | Degradation/prevalence sensitivities increasing in $s_c$ | High-severity detectors are more brittle under distribution shift |

**Detection interface:** Items are scored on a latent scale. Thresholds partition the score range into release/escalate/remove regions. This abstracts from specific ML architectures (CNNs, transformers) while preserving the **precision-recall trade-off**.

### Workload, Capacity, and Accuracy

| Parameter | Generation Rule | Justification |
|-----------|----------------|---------------|
| $\theta_{lck}$ | Handling time increasing in $s_c$ and tier $k$; tens to hundreds of seconds | Anchored to Steiger et al. 2021, Roberts 2019 |
| $a^{rk}_{lc}$ | Reviewer accuracy non-decreasing in tier, higher in-house than outsourced; ≈0.90 at tier 1, rising with severity | Meta CSER 2025 reports 96-98% decision accuracy |
| $\bar{n}_{lk}$ | 3× the reviewer-hours pool $l$ needs at $p^{\mathrm{ref}}$ on its busiest day, divided by $H=8$, rounded up, ≥5 | Overstaffing factor reflects that platforms cannot run at 100% utilization |
| $\bar{v}^0_{ld}$ | Baseline outsourced availability increasing in $r_l$; fraction of pool's frontline hours at $p^{\mathrm{ref}}$ | High-resource pools have better access to BPO capacity |

### Costs

| Parameter | Generation Rule | Justification |
|-----------|----------------|---------------|
| $c^{\mathrm{w}}_{lk}$ | Wage $\sim \mathrm{U}(13, 32)$ USD/hour, rising 35% per tier | U.S. BLS data for content moderators (2025): median $19.50/hour |
| $c^{\mathrm{ot}}_{lk}$ | Overtime = $1.5 \times c^{\mathrm{w}}_{lk}$ | U.S. FLSA standard |
| $c^{\mathrm{fn}}_{c}, c^{\mathrm{fp}}_{c}$ | Error costs log-interpolated by severity/expressiveness; FN: $1.5-200$, FP: $3-22$ per item | **Not actual costs**; decision weights reflecting relative severity |

**Note on error costs:** These are **not monetary valuations** of user harm (which would be ethically fraught and empirically unknowable). They are **relative penalty weights** in the objective function, indexed by policy category, that encode the decision-maker's risk posture toward different error types.

### Compliance and Uncertainty

| Parameter | Generation Rule | Justification |
|-----------|----------------|---------------|
| $\underline{\eta}^{\mathrm{cell}}_{lc}$ | Coverage floors increasing in $s_c$; high-severity categories require near-total review | DSA Article 34 requires "diligent" enforcement |
| $\varpi^{\mathrm{cov}}, \varpi^{\mathrm{insp}}$ | Penalty rates $\sim \mathrm{U}(10, 25)$ per unreviewed item, $\sim \mathrm{U}(15, 45)$ per item for inspection shortfalls | Regulatory fines (DSA Article 52: up to 6% global revenue) |
| $\sigma_Z$ | Volume dispersion $\sim \mathrm{U}(0.18, 0.35)$ | Coefficient of variation consistent with social media traffic patterns |
| $\kappa_\delta, \beta_\delta$ | Detection-shift Gamma shape/scale from $0.5\,\mathrm{U}(0.45, 0.80)$ and $2\,\mathrm{U}(0.30, 0.60)$ | Produces right-skewed degradation (occasional severe shocks) |
| $\rho^{Z\delta}$ | Contemporaneous correlation $\sim \mathrm{U}(0.50, 0.75)$ | Traffic spikes (elections, crises) coincide with detector stress |
| $\varrho$ | Persistence $\sim \mathrm{U}(0.55, 0.80)$ | Shocks (e.g., COVID-19) lasted weeks to months |
| $\zeta_1, \zeta_2$ | Outsourcing-contraction coefficients $\sim \mathrm{U}(0.30, 0.90)$ and $\sim \mathrm{U}(0.80, 1.60)$ | BPOs reduce capacity or raise prices during crises (Gillespie 2018) |

### Scenario Generation

Each instance contains $|\Omega| = 400$ scenarios sampled via:

1. **Copula construction:** Gaussian copula with correlation matrix $\rho^{Z\delta}$
2. **Marginal distributions:**
   - Volume: Lognormal with $\sigma_Z$
   - Detection shift: Gamma($\kappa_\delta, \beta_\delta$)
3. **Temporal persistence:** AR(1) with coefficient $\varrho$
4. **Outsourcing contraction:** $\bar{v}_{ld\omega} = \bar{v}^0_{ld}\,(1 + \zeta_1 (Z_{d\omega}-1)^{+} + \zeta_2 \delta_{d\omega})^{-1}$

**Outcome:** 400 day-by-day paths spanning typical operations, seasonal peaks, and rare joint shocks.

---

## Structural Validation

How do we know our instances are realistic? We validate them against **documented platform behaviors**:

### 1. Qualification Nesting

**Platform evidence:** Meta's 2025 CSER describes three reviewer tiers: "Community Operations" (frontline), "Subject Matter Experts" (senior specialists), and "Policy Experts" (appeal reviewers). Roberts (2019) documents similar hierarchies at YouTube and Twitter.

**Our implementation:** Three tiers with nested qualifications. Tier 1 handles spam/nudity; Tier 2+ required for harassment/hate speech; Tier 3 handles terrorism and performs inspections.

### 2. Inspection Duties

**Platform evidence:** DSA Article 35 mandates "internal oversight" by staff "independent of content moderation activities." Meta CSER 2025 reports a "Quality Assurance team" that samples decisions.

**Our implementation:** Senior tier (Tier 3) performs inspections at mandated rates (2-8% of reviewed items). Inspection shortfalls incur penalties.

### 3. State-Dependent Outsourcing

**Platform evidence:** Gillespie (2018) and Roberts (2019) report that BPO providers reduce capacity or raise prices during demand spikes. During COVID-19, platforms brought moderation in-house due to BPO disruptions.

**Our implementation:** Outsourced availability $\bar{v}_{ld\omega}$ contracts exponentially with volume and detection shocks.

**Validation test:** In Figure 10 of the paper, we show that when $\zeta_1, \zeta_2 > 0$, expected outsourcing usage drops to 40% of its fixed-availability level as correlation $\rho^{Z\delta}$ increases. This matches the qualitative pattern: **outsourcing offers limited protection because it vanishes in adverse states**.

### 4. Detection-Capacity Coupling

**Platform evidence:** Gorwa et al. (2020) and Jiang et al. (2023) document that when platforms tighten detection thresholds (to catch more violations), they send more borderline cases to human review, straining capacity.

**Our implementation:** Operating points jointly determine automated errors and ambiguous-band volume. Stricter thresholds $\Rightarrow$ more content escalated $\Rightarrow$ higher review workload.

**Validation test:** In Section 4.4.3, we show that joint optimization (detection + capacity) reduces total cost by 6.3% on average relative to fixing the detection configuration. This structural result—**you cannot plan capacity without knowing the detection policy**—is the paper's core contribution.

### 5. VSS > EVPI

**Theoretical prediction:** When the commitment timing is fixed (must decide before observing shocks), and recourse options are rich (outsourcing, overtime, penalty-bearing shortfalls), most loss comes from **planning against the wrong distribution** rather than from the **timing of commitment**.

**Our result:** On every tested instance, VSS (value of solving the stochastic program vs. solving for the mean scenario) exceeds EVPI (value of perfect foresight). This is **not a calibration artifact**—it's a structural consequence of the model's decision timing and recourse structure.

---

## Reproducibility

Every instance is fully reproducible:

1. **Deterministic generation:** Given a seed, the generator produces the same instance every time.
2. **Seed manifest:** Each instance has a unique seed stored in `/instances/{tier}/{name}/seed.json`.
3. **Version control:** Generator code is tagged with release versions.
4. **Validation suite:** Unit tests verify that regenerated instances match stored checksums.

### Regenerating an Instance

```bash
# Clone the repository
git clone https://github.com/yinhaosjtu/content-moderation-capacity
cd content-moderation-capacity

# Install dependencies
pip install -r requirements.txt

# Regenerate instance M3 (medium tier, instance 3)
python generator/generate_instance.py --tier medium --id 3 --seed 42 --output instances/medium_tier/M3.json

# Verify checksum
python tools/verify_checksum.py instances/medium_tier/M3.json
```

### File Format

Each instance is stored as a JSON file with the following structure:

```json
{
  "metadata": {
    "name": "M3",
    "tier": "medium",
    "seed": 42,
    "generator_version": "1.0.0",
    "generated_at": "2026-09-07T12:34:56Z"
  },
  "dimensions": {
    "L": 5, "C": 4, "K": 3, "D": 8, "P": 10, "Omega": 400
  },
  "parameters": { ... },
  "scenarios": [ ... ]
}
```

---

## Limitations and Scope

### What Our Instances Can Support

✅ **Mechanism testing:** Joint planning value, correlation effects, qualification nesting, state-dependent outsourcing  
✅ **Algorithm development:** Decomposition methods, heuristics, warm-starting  
✅ **Sensitivity analysis:** Parameter perturbations, scenario count, risk aversion  
✅ **Directional guidance:** How commitment structure and operational conditions interact

### When to Recalibrate

Practitioners should **recalibrate to their own data** before major commitments:

1. Replace our traffic volumes with your observed daily volumes by language/category
2. Replace our handling times with time-study estimates from your operations
3. Replace our error cost weights with your policy priorities
4. Re-estimate scenario distributions from your historical data
5. Validate model outputs against observed capacity utilization and shortfall patterns

The **structural mechanisms** (nesting, correlation, coupling) will remain, but the **numerical values** (optimal headcount, tail cost, joint planning value) will shift to your operating conditions.

---

**Last Updated:** September 7, 2026  
**Document Version:** 1.0.0
