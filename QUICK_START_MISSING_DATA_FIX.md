# Quick Start Guide: Fixing the Missing Data Issue

This is a practical, step-by-step guide to resolve your missing data confusion and get preprocessing working properly.

---

## 🎯 THE PROBLEM (In Simple Terms)

**You think:** "90% threshold is stricter than 75%, so I should be able to handle it"

**Reality:** "90% threshold is MORE permissive - it keeps MORE features (including very sparse ones)"

**The confusion:**
```
Threshold of 90% = "Drop if MORE THAN 90% missing"
                  = "Keep if AT LEAST 10% present"
                  = PERMISSIVE (keeps many sparse features)

Threshold of 70% = "Drop if MORE THAN 70% missing"
                  = "Keep if AT LEAST 30% present"
                  = STRICT (drops sparse features)
```

---

## ✅ QUICK FIX (5 Minutes)

### Option A: Match the Baseline (Recommended)

**Step 1:** Edit `configs/config.yaml` line 229:

```yaml
# CHANGE THIS:
missing_data:
  strategy: "forward_fill_then_median"
  max_missing_ratio: 0.90  # ← Too permissive!

# TO THIS:
missing_data:
  strategy: "forward_fill_then_median"
  max_missing_ratio: 0.70  # ← Matches AI Clinician baseline
```

**Step 2:** Run preprocessing

```bash
python run_preprocessing.py
```

**Why this works:**
- Matches the original AI Clinician methodology
- Only keeps features with ≥30% data availability
- Easier to impute (less sparse data)
- More reproducible results

---

### Option B: Keep Critical Sparse Features (Advanced)

If you specifically need features like lactate or ABGs that might be <30% available:

**Step 1:** Create a new config section in `configs/config.yaml`:

```yaml
missing_data:
  strategy: "forward_fill_then_median"
  max_missing_ratio: 0.70  # Standard threshold

  # Exception: critical features can be up to 90% missing
  critical_features_threshold: 0.90
  critical_features:
    - Arterial_lactate  # Gold standard for septic shock
    - max_dose_vaso     # Your treatment target
    - SOFA              # Reward function
    - WBC_count         # SOFA component
    - Platelets_count   # SOFA component
    - Arterial_pH       # Important for severity
```

**Step 2:** Modify `src/preprocessing/data_cleaning.py` (I can help with this)

---

## 📊 UNDERSTANDING THE NUMBERS

### What Different Thresholds Mean

| Threshold | Meaning | Example |
|-----------|---------|---------|
| **50%** | Keep if ≥50% data | Very strict - drops many features |
| **70%** | Keep if ≥30% data | **AI Clinician baseline** ← Use this |
| **75%** | Keep if ≥25% data | Slightly more permissive |
| **90%** | Keep if ≥10% data | Very permissive - **current setting** |

### Real Example: Lactate Feature

Let's say lactate is measured in 40% of your observations:

```
At 50% threshold: ❌ DROPPED (need ≥50%, have 40%)
At 70% threshold: ✅ KEPT (need ≥30%, have 40%)
At 90% threshold: ✅ KEPT (need ≥10%, have 40%)
```

But if lactate is only measured in 20% of observations:

```
At 50% threshold: ❌ DROPPED (need ≥50%, have 20%)
At 70% threshold: ❌ DROPPED (need ≥30%, have 20%)
At 90% threshold: ✅ KEPT (need ≥10%, have 20%)
```

In the second case, if you use 90% threshold, lactate is 80% imputed values (unreliable!).

---

## ⚠️ ZERO-FILLING: DON'T DO IT!

### Why Zero is Bad

**Your question:** "If I take all features without threshold, how to fill missing values - is filling with 0 right?"

**Answer:** NO! Here's why:

```python
# Example: Heart Rate
Normal range: 60-100 bpm
Typical ICU patient: 80-120 bpm
Missing value filled with 0: IMPOSSIBLE (patient would be dead)

# What the RL agent learns:
"When heart rate = 0, the patient survived!"
→ Completely wrong association
→ Agent learns to ignore heart rate
→ Poor policy
```

### What to Use Instead

| Method | When to Use | Example |
|--------|-------------|---------|
| **Forward Fill** | Temporal data | Use patient's last known HR |
| **Median** | Static features | Use population median HR |
| **KNN** | Complex patterns | Use HR from similar patients |
| **Zero** | Almost never! | Only for counts that can be zero |

---

## 🔍 BEFORE YOU CHANGE ANYTHING

### Check Your Actual Data First!

You haven't run preprocessing yet, so you don't know which features are actually sparse in YOUR data.

**Do this first:**

```bash
# Step 1: Install pandas if needed
pip install pandas numpy scikit-learn

# Step 2: Run feature extraction only (to see raw data)
# Edit run_preprocessing.py to stop after feature extraction

# Step 3: Run missing data analysis
python analyze_missing.py
```

This will tell you:
- Actual missing percentages for each feature
- Which features would be kept/dropped at different thresholds
- Whether critical features like lactate are available

### Expected Output

```
================================================================================
MISSING VALUE ANALYSIS
================================================================================
Total observations: 50,000

--- Features by Missing Percentage ---

Arterial_lactate        : 55.20%  |  @75%: ✓ KEEP  |  @90%: ✓ KEEP
max_dose_vaso          : 45.30%  |  @75%: ✓ KEEP  |  @90%: ✓ KEEP
WBC_count              : 25.10%  |  @75%: ✓ KEEP  |  @90%: ✓ KEEP
Arterial_pH            : 68.40%  |  @75%: ✓ KEEP  |  @90%: ✓ KEEP
paO2                   : 72.20%  |  @75%: ❌ DROP |  @90%: ✓ KEEP
...

================================================================================
SUMMARY
================================================================================
At 75% threshold: 42 features kept, 6 dropped
At 90% threshold: 46 features kept, 2 dropped

Difference: +4 additional features retained at 90%
```

**Then decide:** Are those +4 features worth keeping if they're 75-90% imputed?

---

## 🎓 LEARNING FROM THE BASELINE

### What Original AI Clinician Did

```
Step 1: Forward fill (within each patient)
        ↓ Fills temporal gaps
Step 2: Linear interpolation (only for features <5% missing)
        ↓ Smooths small gaps
Step 3: KNN imputation (k=1, patient similarity)
        ↓ Uses similar patients for remaining gaps
Step 4: Any remaining → Drop the feature
```

**Key insight:** They use sophisticated imputation (KNN), not simple median!

### What You're Currently Doing

```
Step 1: Forward fill (within each patient)
        ↓ Fills temporal gaps
Step 2: Global median
        ↓ Simple but crude
Done!
```

**Problem:** Median doesn't capture patient similarity or temporal patterns.

---

## 🚀 RECOMMENDED PATHS

### Path 1: Quick Fix (5 minutes)

✅ **Best for:** Getting something working fast, reproducibility

1. Change `max_missing_ratio: 0.90` to `0.70` in config.yaml
2. Run preprocessing
3. Check how many features you get
4. If ≥40 features, you're good to go!

### Path 2: Add KNN Imputation (30 minutes)

✅ **Best for:** Better imputation quality, matching baseline more closely

1. Update `data_cleaning.py` to add KNN option (see detailed guide below)
2. Change config to: `strategy: "forward_fill_then_knn"`
3. Keep `max_missing_ratio: 0.70`
4. Run preprocessing

### Path 3: Hybrid Approach (2 hours)

✅ **Best for:** Keeping critical sparse features, best results

1. Implement adaptive threshold logic (see MISSING_DATA_DEEP_ANALYSIS.md)
2. Specify critical features
3. Use KNN imputation
4. Add missingness indicators
5. Run preprocessing

---

## 💻 CODE SNIPPETS

### Quick Fix: Update Config

```yaml
# configs/config.yaml
preprocessing:
  # ... other settings ...

  missing_data:
    strategy: "forward_fill_then_median"
    max_missing_ratio: 0.70  # ← CHANGE THIS LINE
```

### Add KNN Imputation

Add this to `src/preprocessing/data_cleaning.py` around line 380:

```python
# In the _impute_missing method, add this option:

elif self.strategy == 'forward_fill_then_knn':
    logger.info("  Using KNN imputation (matching AI Clinician)...")
    numeric_cols = features.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        # Use k=1 to match original AI Clinician
        # nan_euclidean handles missing values in distance computation
        imputer = KNNImputer(n_neighbors=1, metric='nan_euclidean')
        features[numeric_cols] = imputer.fit_transform(features[numeric_cols])
```

Then update config:

```yaml
missing_data:
  strategy: "forward_fill_then_knn"  # ← NEW
  max_missing_ratio: 0.70
```

---

## 📋 CHECKLIST

Before running preprocessing:

- [ ] I understand that lower threshold = stricter (e.g., 70% is stricter than 90%)
- [ ] I've decided on a threshold (recommended: 0.70)
- [ ] I've updated config.yaml
- [ ] I know that zero-filling is bad
- [ ] I've chosen an imputation strategy (start with median, upgrade to KNN later)

After first preprocessing run:

- [ ] Check how many features were retained (expect ~40-48)
- [ ] Check if critical features (lactate, vasopressor dose) are present
- [ ] Verify no missing values remain in final data
- [ ] Review the preprocessing log for warnings

---

## 🆘 TROUBLESHOOTING

### Problem: "Too few features retained"

```
Expected: 48 features
Got: 25 features
```

**Solution:**
1. Check actual missing percentages: `python analyze_missing.py`
2. Lower the threshold (try 0.80 instead of 0.70)
3. Or: Use hybrid approach to keep critical features

### Problem: "Still have missing values after imputation"

```
WARNING: 1,234 missing values remain
```

**Solution:**
1. Check which features: `data.isnull().sum()`
2. Verify forward-fill is working (needs stay_id and time_window)
3. Add fallback median imputation for any remaining gaps

### Problem: "KNN imputation is too slow"

```
Step 3: Imputing... (still running after 30 minutes)
```

**Solution:**
1. Use `n_neighbors=1` (faster than 5)
2. Use median instead for first pass, KNN only if needed
3. Process in chunks: `impute_in_chunks(data, chunk_size=10000)`

---

## 📚 WHERE TO LEARN MORE

- **Comprehensive analysis:** See `MISSING_DATA_DEEP_ANALYSIS.md`
- **Methodology comparison:** See `METHODOLOGY_COMPARISON.md`
- **Implementation details:** See `src/preprocessing/data_cleaning.py`
- **Original AI Clinician:** https://github.com/matthieukomorowski/AI_Clinician

---

## ✨ KEY TAKEAWAY

**The One Thing to Remember:**

```
Lower threshold number = Stricter filtering = Fewer features
Higher threshold number = More permissive = More features

70% threshold ← Start here (matches baseline)
90% threshold ← Too permissive for most cases
```

**And:** Never use zero-filling! Use forward-fill + median or KNN instead.

---

Need help implementing any of these? Let me know which path you want to take!

*Last updated: 2025-11-17*
