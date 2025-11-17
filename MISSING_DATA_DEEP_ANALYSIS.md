# Missing Data Deep Analysis & Solutions
## Comprehensive Guide to Resolving the 75% vs 90% Threshold Confusion

*Last Updated: 2025-11-17*

---

## 📋 EXECUTIVE SUMMARY

**The Key Issue:** You're confused about why the baseline (AI Clinician) works with 75% threshold but you struggle with 90% threshold. This comprehensive analysis explains the confusion, compares methodologies, and provides clear solutions.

**Quick Answer:**
1. **Original AI Clinician used 70% threshold** (drop features with >70% missing = keep features with >30% data)
2. **Your current config uses 90% threshold** (drop features with >90% missing = keep features with >10% data)
3. **The 90% threshold is MORE permissive** - it keeps MORE features (including very sparse ones)
4. **Zero-filling is NOT recommended** - use forward-fill + median or KNN imputation instead

---

## 🔍 UNDERSTANDING THE CONFUSION

### Why Does 75% Work for Baseline but Not for You?

This is a **conceptual misunderstanding**. Let's clarify:

| Threshold | What It Means | Features Kept | Features Dropped |
|-----------|---------------|---------------|------------------|
| **70% missing** | Drop if >70% missing | Features with ≥30% data | Very sparse features |
| **75% missing** | Drop if >75% missing | Features with ≥25% data | Extremely sparse features |
| **90% missing** | Drop if >90% missing | Features with ≥10% data | Only the most sparse features |

**The Reality:**
- **Lower threshold (70%)** = MORE strict = FEWER features retained = EASIER to impute remaining
- **Higher threshold (90%)** = MORE permissive = MORE features retained = HARDER to impute (many very sparse features)

**Why you're struggling:**
- At 90% threshold, you're keeping many features that are 80-90% missing
- These features are VERY difficult to impute accurately
- Simple median imputation doesn't work well for such sparse data
- You end up with features that are mostly imputed values (unreliable)

---

## 📊 ORIGINAL AI CLINICIAN METHODOLOGY (2018)

### 1. Missing Data Threshold

**Source:** Komorowski et al., Nature Medicine 2018 + MATLAB code analysis

```matlab
% Original AI Clinician approach:
% Drop features with >70% missing values
threshold = 0.70;  % 70% missing = 30% data availability required
```

**Features Required:** 48 total features
**Typical Retention:** ~45-48 features after 70% threshold

### 2. Three-Stage Imputation Pipeline

The original AI Clinician used a sophisticated **3-stage approach**:

```
STAGE 1: Sample-and-Hold (SAH)
├─ Forward-fill last observed value within each patient
├─ Applied TWICE in preprocessing pipeline
└─ Purpose: Handle sparse temporal measurements in ICU

STAGE 2: Linear Interpolation (fixgaps.m)
├─ Applied ONLY to features with <5% missing (after SAH)
├─ Uses linear interpolation between time points
└─ Purpose: Fill small gaps in frequently measured vitals

STAGE 3: K-Nearest Neighbors Imputation
├─ kNN with k=1, Standardized Euclidean distance
├─ Applied in chunks of 10,000 records (memory management)
├─ Imputes based on similar patient states
└─ Purpose: Fill remaining gaps using similar patients
```

**Key Point:** They use **patient-similarity-based KNN imputation**, NOT simple median!

### 3. Features Typically Retained

Based on MIMIC-III sepsis cohort analysis:

**Well-measured features (typically >70% complete):**
- Demographics: age, gender, weight
- Vital signs: HR, BP, RR, Temp, SpO2, GCS
- Basic labs: Sodium, Potassium, Glucose, Creatinine, BUN
- Fluid balance: inputs, outputs

**Moderately sparse features (30-70% complete):**
- WBC, Platelets, Hemoglobin
- Some liver enzymes (SGOT, SGPT, Bilirubin)
- Coagulation (PT, PTT, INR)

**Very sparse features (typically <30%, DROPPED at 70% threshold):**
- Arterial blood gases (pH, pO2, pCO2, BE) - often 40-60% missing
- Lactate - typically 50-70% missing
- Some advanced labs

---

## 🎯 YOUR CURRENT IMPLEMENTATION

### Configuration (config.yaml:229)

```yaml
missing_data:
  strategy: "forward_fill_then_median"
  max_missing_ratio: 0.90  # Drop features with >90% missing
```

### What This Means

1. **You're being MORE permissive than the baseline**
   - Baseline: Drop if >70% missing
   - You: Drop if >90% missing
   - Result: You retain features with 70-90% missing (very sparse!)

2. **You're using simpler imputation**
   - Baseline: Forward-fill → Interpolation → KNN
   - You: Forward-fill → Median
   - Result: Less sophisticated handling of sparse data

3. **Why this creates problems:**
   - Features that are 80% missing → 80% of values will be median
   - These features carry little actual information
   - They can mislead the RL algorithm
   - Training becomes less stable

---

## 🔬 RESEARCH FINDINGS: BEST PRACTICES (2024)

### Latest Evidence on Imputation Methods

Based on systematic reviews and MIMIC-specific studies:

#### Performance Ranking (2024 studies)

| Method | Performance | Best For | Issues |
|--------|-------------|----------|--------|
| **MissForest** | ★★★★★ Best | Complex patterns | Computationally expensive |
| **MICE** | ★★★★☆ Excellent | Medical data | Slow for large datasets |
| **KNN** | ★★★★☆ Very Good | ICU time-series | Needs careful distance metric |
| **Linear Interpolation** | ★★★☆☆ Good | Temporal gaps | Only works for time-series |
| **Median** | ★★☆☆☆ Fair | Simple cases | Ignores relationships |
| **Forward Fill (LOCF)** | ★☆☆☆☆ Poor | Last resort | Highest RMSE/MAE |
| **Zero Filling** | ☆☆☆☆☆ Not Recommended | Almost never | Introduces bias |

#### Key Finding for ICU Data

**Best approach:** Combine forward-fill (temporal) + KNN (cross-sectional)

```python
# State-of-the-art approach for ICU data:
1. Forward fill within patient (captures temporal continuity)
2. Linear interpolation for features with <5% missing
3. Weighted KNN for remaining gaps (uses patient similarity)
```

### Specific Findings for MIMIC Sepsis Data

From recent studies (2024-2025):

1. **Lactate missingness:** ~28-30% in MIMIC-IV sepsis cohorts
2. **ABG missingness:** ~40-60% (pH, pO2, pCO2)
3. **Typical threshold:** Most studies use 20-30% data requirement (= 70-80% missing threshold)
4. **Common practice:** Some studies require ≥95% completeness (very strict!)

---

## ❓ ANSWERING YOUR SPECIFIC QUESTIONS

### Q1: Why does baseline use 75% and work, but I can't work with 90%?

**Answer:** This is a misunderstanding. Let me clarify:

1. **Baseline actually uses 70% (not 75%)**
   - Source: Komorowski et al. MATLAB code
   - 70% missing threshold = keep features with ≥30% data

2. **Your 90% threshold is MORE permissive, not less**
   - 90% threshold = keep features with ≥10% data
   - You're keeping MORE features (including very sparse ones)
   - These sparse features are HARDER to impute

3. **Why you're struggling:**
   - You're trying to work with features that are 80-90% missing
   - Simple median imputation doesn't work well for such sparse data
   - The baseline DROPPED these features (smarter approach)

**Solution:** Use a LOWER threshold (like 70%) to match the baseline!

---

### Q2: What features did the baseline use?

**Original AI Clinician Features (48 total):**

See complete list in METHODOLOGY_COMPARISON.md, but key categories:

✅ **Well-measured features (retained at 70% threshold):**
- Demographics (4): age, gender, weight, readmission
- Vital signs (10): HR, BP, RR, Temp, SpO2, GCS, etc.
- Basic chemistry (8-10): Na, K, Glucose, Creatinine, BUN, etc.
- Hematology (4-5): WBC, Hb, Platelets, PT/PTT
- Fluid balance (4): inputs/outputs
- Derived scores (4): SOFA, SIRS, shock index, PaO2/FiO2

❌ **Typically DROPPED (>70% missing in many cohorts):**
- Some arterial blood gases (if >70% missing)
- Lactate (if >70% missing - though this is borderline at ~50-60%)
- Some liver enzymes in certain cohorts
- INR (if sparse)

**Important Note:** Which features are retained depends on your specific cohort's measurement patterns!

---

### Q3: What features do you require?

**Your config specifies 48 features (same as baseline):**

```yaml
# From config.yaml
state_features:
  demographics: 4
  vitals: 10
  labs_chemistry: 11
  labs_hematology: 6
  labs_blood_gas: 6
  fluid_balance: 4
  derived: 5
Total: 48 features
```

**Critical features for sepsis RL:**
- **Lactate** (Arterial_lactate) - Gold standard for septic shock
- **Vasopressor dose** (max_dose_vaso) - Your treatment target
- **SOFA score** - Reward function component
- **Blood pressure** (SysBP, MeanBP) - Treatment effectiveness
- **Fluid balance** - Treatment component
- **WBC, Platelets** - SOFA components

**Reality Check:** You need to verify which of these 48 features are actually available in YOUR data with sufficient completeness (>30% at 70% threshold).

---

### Q4: If I take all features without threshold, how to fill missing values? Is zero-filling the right option?

**Short Answer: NO! Zero-filling is NOT recommended.**

#### Why Zero-Filling is Bad

1. **Introduces systematic bias:**
   ```
   Heart Rate: normal range 60-100, median ~80
   If you fill missing values with 0:
   - 0 is physiologically impossible (patient would be dead)
   - Completely distorts the distribution
   - RL agent learns wrong associations
   ```

2. **Better alternatives exist** (see ranking above)

3. **Special cases where 0 might make sense:**
   - Binary indicators (e.g., "received vasopressor: 0=no, 1=yes")
   - But even then, forward-fill is usually better

#### Recommended Approach If You Must Keep All Features

**Option 1: Sophisticated Imputation (Recommended)**

```python
# Multi-stage approach
1. Forward fill within patient (temporal continuity)
2. KNN imputation (patient similarity)
3. Median for any remaining gaps (last resort)
```

**Option 2: Hybrid Strategy (Practical)**

```python
# Different strategies for different sparsity levels
def impute_by_sparsity(feature, missing_pct):
    if missing_pct < 0.20:  # <20% missing
        return forward_fill_then_interpolate(feature)
    elif missing_pct < 0.50:  # 20-50% missing
        return forward_fill_then_knn(feature)
    elif missing_pct < 0.80:  # 50-80% missing
        return forward_fill_then_median(feature)
    else:  # >80% missing
        # Consider dropping this feature!
        return forward_fill_then_median_with_flag(feature)
```

**Option 3: Missingness Indicators (Advanced)**

```python
# Add a binary "was_missing" indicator for very sparse features
for feature in sparse_features:
    data[f'{feature}_was_missing'] = data[feature].isnull().astype(int)
    data[feature] = impute(data[feature])  # Then impute

# This lets the RL agent learn that imputed values are less reliable
```

---

### Q5: Do we have number of missing values for each feature?

**Answer:** Not yet - you haven't run preprocessing on actual data!

#### How to Get This Information

**Step 1: Run the analysis script (after fixing pandas import)**

```bash
# Install pandas first if needed
pip install pandas

# Run missing data analysis
python analyze_missing.py
```

This will show:
- Missing percentage for each feature
- Which features would be kept at 75% vs 90% threshold
- Critical features that would be recovered

**Step 2: During preprocessing**

The `DataCleaner` class has a method:

```python
# In data_cleaning.py
cleaner = DataCleaner(config)
missing_summary = cleaner.get_missing_summary(data)
print(missing_summary)
```

This generates a detailed report of missing values.

**Step 3: Create a visualization**

I can help you create a script to visualize missing data patterns once you have the data.

---

## 🎯 RECOMMENDED SOLUTIONS

### Solution 1: Match the Baseline (Recommended for Reproducibility)

**Goal:** Reproduce the original AI Clinician methodology

#### Step 1: Update config.yaml

```yaml
missing_data:
  strategy: "forward_fill_then_median"  # Keep for now
  max_missing_ratio: 0.70  # Change from 0.90 to 0.70 ← KEY CHANGE
```

#### Step 2: Implement KNN imputation option

```python
# In data_cleaning.py, modify _impute_missing method
elif self.strategy == 'knn':
    logger.info("  Using KNN imputation (k=1, matching AI Clinician)...")
    numeric_cols = features.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        # Use k=1 to match original AI Clinician
        imputer = KNNImputer(n_neighbors=1, metric='nan_euclidean')
        features[numeric_cols] = imputer.fit_transform(features[numeric_cols])
```

#### Step 3: Update config to use KNN

```yaml
missing_data:
  strategy: "forward_fill_then_knn"  # New strategy
  max_missing_ratio: 0.70
  knn_neighbors: 1  # Match original AI Clinician
```

**Pros:**
- ✅ Matches benchmark methodology
- ✅ More reproducible results
- ✅ Can compare with published results
- ✅ Only works with features that have ≥30% data

**Cons:**
- ❌ May drop some clinically important but sparse features (like lactate)
- ❌ Requires implementing KNN imputation

---

### Solution 2: Permissive Threshold with Advanced Imputation

**Goal:** Keep more features (including sparse ones) but impute them properly

#### Step 1: Keep 90% threshold

```yaml
missing_data:
  strategy: "forward_fill_then_knn"  # Use KNN, not median!
  max_missing_ratio: 0.90  # Keep permissive threshold
  knn_neighbors: 5  # Use more neighbors for very sparse data
```

#### Step 2: Add missingness indicators

```python
# New method in data_cleaning.py
def add_missingness_indicators(self, data, threshold=0.50):
    """
    Add binary indicators for features with >threshold missing
    This helps the RL agent know which values were imputed
    """
    for col in data.columns:
        missing_pct = data[col].isnull().mean()
        if missing_pct > threshold:
            data[f'{col}_was_missing'] = data[col].isnull().astype(int)
    return data
```

**Pros:**
- ✅ Keeps critical sparse features (lactate, ABGs)
- ✅ KNN imputation better than median for sparse data
- ✅ Missingness indicators help RL agent
- ✅ More information available

**Cons:**
- ❌ Less reproducible (deviates from benchmark)
- ❌ More complex methodology
- ❌ KNN imputation is slow for large datasets
- ❌ Imputed values for very sparse features are unreliable

---

### Solution 3: Hybrid Approach (Best of Both Worlds)

**Goal:** Use different thresholds for different feature types

#### Implementation

```yaml
# config.yaml
missing_data:
  strategy: "adaptive"  # New adaptive strategy
  thresholds:
    critical_features: 0.90  # Permissive for critical features
    standard_features: 0.70  # Strict for others

  critical_features:  # Must keep even if sparse
    - Arterial_lactate
    - max_dose_vaso
    - SOFA
    - SIRS
    - SysBP
    - MeanBP
    - WBC_count
    - Platelets_count

  imputation:
    temporal: "forward_fill"
    cross_sectional: "knn"
    last_resort: "median"
```

```python
# In data_cleaning.py
def _drop_high_missing_features_adaptive(self, data):
    """Drop features adaptively based on importance"""
    critical = self.config['missing_data']['critical_features']
    missing_ratios = data.isnull().sum() / len(data)

    dropped = []
    for feature in data.columns:
        missing_pct = missing_ratios[feature]

        if feature in critical:
            # Critical features: use permissive threshold
            if missing_pct > 0.90:
                dropped.append(feature)
        else:
            # Standard features: use strict threshold
            if missing_pct > 0.70:
                dropped.append(feature)

    return data.drop(columns=dropped), dropped
```

**Pros:**
- ✅ Keeps critical features even if sparse
- ✅ Drops non-critical sparse features (noise reduction)
- ✅ Balances reproducibility and information retention
- ✅ Clinically motivated

**Cons:**
- ❌ More complex configuration
- ❌ Requires domain knowledge to specify critical features
- ❌ Harder to explain in paper

---

## 📊 DECISION MATRIX

| Criteria | Solution 1: Match Baseline | Solution 2: Permissive | Solution 3: Hybrid |
|----------|---------------------------|------------------------|---------------------|
| **Reproducibility** | ★★★★★ | ★★☆☆☆ | ★★★☆☆ |
| **Information Retention** | ★★★☆☆ | ★★★★★ | ★★★★☆ |
| **Implementation Complexity** | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ |
| **Computational Cost** | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ |
| **Clinical Relevance** | ★★★★☆ | ★★★★☆ | ★★★★★ |
| **Justifiable in Paper** | ★★★★★ | ★★★☆☆ | ★★★★☆ |

---

## 🚀 RECOMMENDED ACTION PLAN

### Phase 1: Understand Your Data (FIRST!)

Before making any decisions, you MUST analyze your actual data:

```bash
# Step 1: Install dependencies
pip install pandas numpy scikit-learn

# Step 2: Run preprocessing up to feature extraction
# (Stop before cleaning to see raw missing percentages)

# Step 3: Run missing data analysis
python analyze_missing.py

# Step 4: Review the output
# - Which features have >70% missing?
# - Which critical features are at risk?
# - Are lactate and ABGs available with ≥30% data?
```

### Phase 2: Choose Strategy Based on Data

**If lactate and ABGs have ≥30% data:**
→ Use Solution 1 (Match Baseline)

**If lactate and ABGs have <30% but ≥10% data:**
→ Use Solution 3 (Hybrid Approach)

**If even with 90% threshold you have too few features:**
→ This indicates a serious data quality problem
→ May need to reconsider cohort selection

### Phase 3: Implement and Validate

```python
# Step 1: Implement chosen solution
# Step 2: Run preprocessing
# Step 3: Validate:
#   - Check number of features retained
#   - Check imputation quality
#   - Check for remaining missing values
#   - Visualize distributions before/after
```

### Phase 4: Document and Justify

For your paper, include:
1. Missing data analysis table (% missing per feature)
2. Justification for threshold choice
3. Comparison with original AI Clinician
4. Ablation study (if using different approach)

---

## 🔍 UNDERSTANDING THE ORIGINAL AI CLINICIAN'S APPROACH

### What Made Their Approach Work?

1. **Conservative threshold (70%)**
   - Only kept features with ≥30% data availability
   - Avoided very sparse features
   - More reliable imputation

2. **Sophisticated imputation**
   - Not just median!
   - Patient-similarity-based (KNN)
   - Temporal continuity (forward-fill)

3. **Large dataset**
   - MIMIC-III: ~20,000 sepsis ICU stays
   - More data → better KNN performance
   - Better statistical properties

4. **Feature engineering**
   - Derived features (SOFA, SIRS) aggregate multiple measurements
   - Less sensitive to individual feature missingness
   - More robust

---

## ⚠️ COMMON PITFALLS TO AVOID

### ❌ Pitfall 1: Using Too High a Threshold (>80%)

```python
# BAD:
max_missing_ratio: 0.95  # Keeps features with only 5% data!

# RESULT:
# - Features are 95% imputed values
# - Little actual information
# - RL agent learns from noise
```

### ❌ Pitfall 2: Simple Imputation for Sparse Data

```python
# BAD:
# Feature is 80% missing
# Simple median imputation
# → 80% of values are now the median!

# RESULT:
# - No variance in most values
# - RL agent can't learn from this feature
# - Waste of computational resources
```

### ❌ Pitfall 3: Zero-Filling

```python
# BAD:
data.fillna(0, inplace=True)  # NEVER DO THIS!

# RESULT:
# - Heart rate = 0 (impossible!)
# - Completely distorted distributions
# - RL agent learns wrong patterns
```

### ❌ Pitfall 4: Ignoring Temporal Structure

```python
# BAD:
# Global median across all patients and time points
data['HR'].fillna(data['HR'].median(), inplace=True)

# BETTER:
# Forward fill within each patient first
data = data.groupby('stay_id').ffill()
# Then use median for remaining gaps
```

---

## 📚 REFERENCES & FURTHER READING

### Key Papers

1. **Original AI Clinician:**
   - Komorowski et al. (2018). "The artificial intelligence clinician learns optimal treatment strategies for sepsis in intensive care." *Nature Medicine*, 24(11), 1716-1720.
   - GitHub: https://github.com/matthieukomorowski/AI_Clinician

2. **Missing Data Imputation in Medical Data:**
   - "Evaluating the state of the art in missing data imputation for clinical data" (2022). *Briefings in Bioinformatics*, 23(1).
   - "A Combined Interpolation and Weighted K-Nearest Neighbours Approach for the Imputation of Longitudinal ICU Laboratory Data" (2022). *PMC*.

3. **MIMIC Sepsis Studies:**
   - "MIMIC-Sepsis: A Curated Benchmark for Modeling and Learning from Sepsis Trajectories in the ICU" (2024).
   - "A flexible framework for sepsis prediction: Standardizing data management and imputation in time series using MIMIC-III" (2025).

### Best Practices Documents

- See: `METHODOLOGY_COMPARISON.md` for detailed comparison with baseline
- See: `PREPROCESSING_README.md` for implementation details
- See: `data_cleaning.py` for current imputation code

---

## 💡 KEY TAKEAWAYS

1. **Lower threshold = stricter** (70% missing = keep only features with ≥30% data)
2. **Your 90% threshold is MORE permissive, not less** (keeps very sparse features)
3. **Zero-filling is BAD** - use forward-fill + KNN or median instead
4. **Match the baseline first** (70% threshold) for reproducibility
5. **Analyze your actual data** before making decisions
6. **KNN imputation > median** for sparse data
7. **Document everything** for your paper

---

## 🤝 NEXT STEPS

Would you like me to:

1. **Implement Solution 1** (Match baseline with 70% threshold)?
2. **Implement Solution 3** (Hybrid approach with critical features)?
3. **Create a missing data visualization script** for when you have data?
4. **Modify the config.yaml** with your preferred settings?
5. **Add KNN imputation** to the DataCleaner class?

Let me know which direction you'd like to go!

---

*Created: 2025-11-17*
*Author: Claude Code Analysis*
*Project: Sepsis RL with AI Clinician Methodology*
