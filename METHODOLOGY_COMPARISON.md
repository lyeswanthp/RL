# AI Clinician Methodology Comparison
## Original (Komorowski et al. 2018) vs Your Implementation

---

## 📋 EXECUTIVE SUMMARY

### Key Finding
Your current implementation is **MORE PERMISSIVE** than the original AI Clinician, which could be **beneficial** for retaining critical sparse features (lactate, ABGs) but requires careful validation.

---

## 1. MISSING DATA THRESHOLD

| Method | Threshold | Rationale |
|--------|-----------|-----------|
| **Original AI Clinician (2018)** | **Drop if >70% missing** | Conservative approach - only keep features with >30% data availability |
| **Your Current Implementation** | **Drop if >90% missing** | Liberal approach - retain very sparse but clinically critical features |
| **Recommendation** | **Use 70-75%** initially, then compare | Follow benchmark for reproducibility, then ablate |

---

## 2. MISSING DATA IMPUTATION PIPELINE

### Original AI Clinician (3-Stage Pipeline)

```
Stage 1: Sample-and-Hold (SAH)
├─ Forward-fill last observed value within each patient
├─ Applied TWICE in preprocessing pipeline
└─ Purpose: Handle sparse temporal measurements

Stage 2: Linear Interpolation (fixgaps.m)
├─ Applied ONLY to features with <5% missing
├─ Uses linear interpolation between time points
└─ Purpose: Fill small gaps in frequently measured vitals

Stage 3: K-Nearest Neighbors Imputation
├─ kNN with k=1, Standardized Euclidean distance
├─ Applied in chunks of 10,000 records
├─ Imputes based on similar patient states
└─ Purpose: Fill remaining gaps using similar patients
```

**Key Insight:** They use **sophisticated KNN imputation** based on patient similarity, not just median.

---

### Your Current Implementation (2-Stage Pipeline)

```
Stage 1: Forward Fill (within each patient stay)
├─ Group by stay_id
├─ Sort by time_window
├─ Apply forward fill (.ffill())
└─ Purpose: Temporal continuity

Stage 2: Global Median Imputation
├─ Compute median from training set
├─ Fill all remaining NaNs
└─ Purpose: Conservative fallback
```

**Key Insight:** You use **simpler median imputation**, not patient-similarity-based.

---

### Your Proposal Specification

From your PDF (Page 4, Section III.B):
> "All continuous state variables will be normalized using **z-scoring** (transforming to mean 0, standard deviation 1). The mean and standard deviation for this transformation will be computed only from the 70% training set and then persistently applied to the validation and test sets to prevent data leakage."

**Normalization:** Z-score (matches your current implementation ✓)

**Missing Data Strategy** (Page 4, config excerpt):
> `strategy: "forward_fill_then_median"`

**Matches your current implementation ✓**

---

## 3. FEATURE SET COMPARISON

### Original AI Clinician Features (48 total)

**From their MATLAB code:**
```matlab
% Demographics (4)
- Age, Gender, Weight, Re-admission

% Vital Signs (10)
- HR, SysBP, MeanBP, DiaBP, RR, Temp, SpO2, FiO2, GCS, MechVent

% Labs - Chemistry (11)
- K+, Na+, Cl-, Glucose, BUN, Creatinine, Mg2+, Ca2+, SGOT, SGPT, Bilirubin

% Labs - Hematology (6)
- Hb, WBC, Platelets, PTT, PT, INR

% Labs - Blood Gas (6)
- pH, paO2, paCO2, Base Excess, HCO3, Lactate

% Fluid Balance (4)
- Input_total, Input_4h, Output_total, Output_4h

% Derived Scores (5-7)
- SOFA, SIRS, Shock Index, PaO2/FiO2 ratio, Cumulative balance
```

### Your Proposal Features (48 total)

**From your PDF (Page 2, Section II.A):**
> "The patient state $s$ at each 4-hour timestep is represented by a vector of **48 physiological and demographic variables**. This set is adopted from the benchmark study and includes 34 continuous variables (e.g., vital signs, lab values) and 14 binary indicators (e.g., ventilation status, comorbidities)."

**Your config.yaml** lists the same 48 features ✓

---

## 4. OUTLIER DETECTION

### Original AI Clinician

**Method:** Hard physiological cutoffs
```matlab
% Examples from their code:
- Lactate > 30 mmol/L → DELETE
- Hemoglobin > 20 g/dL → DELETE
- WBC > 500 × 10⁹/L → DELETE
- Temperature outliers removed via deloutabove/deloutbelow functions
```

**No statistical outlier detection (IQR/Z-score)**

### Your Implementation

**Method 1:** Physiological ranges (similar to original ✓)
```python
VALID_RANGES = {
    'Arterial_lactate': (0.1, 30),  # Matches original
    'Hb': (2, 25),                   # Matches original
    'WBC_count': (0, 500),           # Matches original
    ...
}
```

**Method 2:** IQR statistical outlier detection (additional)
```python
# Q1 - 3*IQR, Q3 + 3*IQR
# More aggressive than original
```

**Analysis:** Your approach is **more aggressive** - you apply both physiological AND statistical outlier removal.

---

## 5. NORMALIZATION

### Original AI Clinician

**From code analysis:**
- No explicit z-score normalization found in MATLAB preprocessing
- They may normalize within the discretization/clustering step
- **Unclear from available code**

### Your Implementation

**Z-score normalization:**
```python
# From normalization.py
# Continuous features: (x - mean) / std
# Log-transform first: SpO2, BUN, Creatinine, etc.
# Binary features: subtract 0.5
```

**From your proposal (Page 4):**
> "All continuous state variables will be normalized using z-scoring"

**This is GOOD** - z-score normalization is standard for linear Q-learning.

---

## 6. KEY DIFFERENCES SUMMARY

| Aspect | Original AI Clinician | Your Implementation | Assessment |
|--------|----------------------|---------------------|------------|
| **Missing threshold** | 70% | **90%** | ⚠️ More permissive - may retain too-sparse features |
| **Imputation method** | Forward-fill → Interpolation → **KNN** | Forward-fill → **Median** | ⚠️ Simpler approach - may lose patient similarity info |
| **Normalization** | Unclear | **Z-score** | ✅ Good - needed for linear Q-learning |
| **Outlier removal** | Physiological only | Physiological + **IQR** | ⚠️ More aggressive - may remove valid extreme values |
| **Feature set** | 48 features | **48 features** (same) | ✅ Matches benchmark |
| **Time window** | 4 hours | **4 hours** | ✅ Matches benchmark |

---

## 📝 CRITICAL RECOMMENDATIONS

### 🚨 RECOMMENDATION 1: Align Missing Data Threshold with Original

**Current:** 90% threshold
**Original:** 70% threshold
**Action:** Change `max_missing_ratio: 0.70` in `config.yaml`

**Rationale:**
- Benchmark reproducibility
- Avoid features with <30% data availability
- Critical features (lactate, WBC, platelets) likely have 30-70% availability and will be retained

---

### 🚨 RECOMMENDATION 2: Consider Adding KNN Imputation (Optional)

**Current:** Forward-fill → Median
**Original:** Forward-fill → Interpolation → KNN
**Action:** Add KNN imputation as optional strategy

**Rationale:**
- KNN uses patient similarity (more sophisticated than median)
- Original AI Clinician showed this works well
- Can compare median vs KNN in ablation study

**Code change needed:**
```python
# In data_cleaning.py, add KNN option
elif self.strategy == 'knn':
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=1, metric='nan_euclidean')
    features[numeric_cols] = imputer.fit_transform(features[numeric_cols])
```

---

### 🚨 RECOMMENDATION 3: Verify Outlier Strategy

**Current:** Physiological + IQR (double filtering)
**Original:** Physiological only
**Action:** Make IQR optional, default to physiological-only

**Rationale:**
- IQR may remove clinically valid extreme values (e.g., WBC=450 in severe sepsis)
- Original AI Clinician only used physiological ranges
- Can enable IQR as optional aggressive cleaning

---

### 🚨 RECOMMENDATION 4: Document Feature Missingness

**Action:** Before re-running preprocessing, analyze actual missing percentages

**Why:** Understand which critical features will be:
- Retained at 70% threshold
- Retained at 90% threshold
- Dropped at both thresholds

**Critical features to check:**
- `Arterial_lactate` (gold standard for septic shock)
- `max_dose_vaso` (your RL action/target)
- `WBC_count`, `Platelets_count` (SOFA components)
- ABGs: `Arterial_pH`, `paCO2`, `paO2`

---

## 🎯 PROPOSED WORKFLOW

### Phase 1: Reproduce Benchmark (70% threshold)
```yaml
# config.yaml
missing_data:
  strategy: "forward_fill_then_median"
  max_missing_ratio: 0.70  # Match original
```

**Goal:** Establish baseline comparable to Komorowski et al.

---

### Phase 2: Ablation Study (if needed)
Test different thresholds and imputation strategies:

| Config | Threshold | Imputation | Purpose |
|--------|-----------|------------|---------|
| **Baseline** | 70% | Forward-fill + Median | Reproduce benchmark |
| **Permissive** | 90% | Forward-fill + Median | Retain sparse critical features |
| **Sophisticated** | 70% | Forward-fill + KNN | Match original method |
| **Aggressive** | 90% | Forward-fill + KNN | Best of both worlds |

Evaluate using validation set WDR-OPE (from your proposal Section V).

---

## 📚 REFERENCES

**Original AI Clinician:**
- Komorowski et al. (2018). *Nature Medicine*. DOI: 10.1038/s41591-018-0213-5
- GitHub: https://github.com/matthieukomorowski/AI_Clinician
- MATLAB preprocessing: `AIClinician_sepsis3_def_160219.m`

**Your Implementation:**
- Proposal: "A Robust and Interpretable Framework for Sepsis Treatment Policy Optimization"
- Code: `/home/user/RL/`
- Config: `configs/config.yaml`

---

## ✅ ALIGNMENT CHECK: Your Proposal vs Implementation

| Proposal Requirement | Current Implementation | Status |
|---------------------|------------------------|--------|
| 48 state features | ✓ 48 features in config | ✅ |
| 4-hour time windows | ✓ 4-hour windows | ✅ |
| Z-score normalization | ✓ Implemented | ✅ |
| Forward-fill + Median | ✓ Implemented | ✅ |
| 70/15/15 split | ✓ Implemented | ✅ |
| Missing threshold | ⚠️ Using 90%, proposal doesn't specify | ⚠️ Should use 70% for benchmark alignment |
| KNN imputation option | ❌ Not implemented | ⚠️ Optional but recommended |

---

## 🔬 NEXT STEPS

1. **Decide on threshold:** 70% (benchmark) vs 90% (permissive)
2. **Run preprocessing** with chosen threshold
3. **Analyze feature retention:** Which critical features are kept/dropped
4. **Compare with benchmark:** Expected ~48 features after cleaning
5. **Validate clinically:** Ensure lactate, vasopressors retained
6. **Document decision:** Justify any deviations from original

---

*Last updated: 2025-11-16*
