# Next Steps Guide - AI Clinician Implementation

**Date:** 2024-11-16
**Status:** Preprocessing Complete ✅ | Next: Model Training & Evaluation

---

## 📊 Current Status

### ✅ What's Complete:

1. **Preprocessing Pipeline** (fully functional)
   - 54,926 sepsis ICU stays identified
   - 1,613,734 observations with 4-hour windows
   - Train/val/test split completed
   - Data saved in `data/processed/`

2. **New RL Components** (just created):
   - Action extraction module
   - Reward computation module
   - MDP trajectory builder
   - Linear Q-learning algorithm
   - WDR-OPE evaluator
   - Data quality analyzer

### ⚠️ Critical Issue Identified:

**Only 13 features remain** after preprocessing (expected 48+)
- **Cause:** 27 features dropped due to >50% missing values
- **Impact:** Reduced model performance likely
- **Solutions:** See "Option 1" below

---

## 🚀 Recommended Next Steps

### **Option 1: Fix Data Quality (RECOMMENDED)**

The preprocessing dropped too many critical features. You should:

#### Step 1A: Analyze Current Data Quality

```bash
# Run the data quality analyzer
python analyze_data_quality.py
```

This will show you:
- Which features were dropped and why
- Missingness patterns
- Recommendations

#### Step 1B: Adjust Preprocessing Configuration

Edit `configs/config.yaml`:

```yaml
preprocessing:
  missing_data:
    strategy: "forward_fill_then_median"
    max_missing_ratio: 0.70  # ← Change from 0.5 to 0.7 or 0.8
```

**Rationale:** Clinical data is naturally sparse. Blood gas values, advanced labs are only measured when clinically indicated. A 70-80% threshold is more appropriate.

#### Step 1C: Re-run Preprocessing

```bash
# Re-run with adjusted config
python run_preprocessing.py --data-path /path/to/your/mimic-iv-3.1
```

Expected outcome:
- ~30-40 features retained (instead of 13)
- Better representation of patient state
- Higher model performance

---

### **Option 2: Proceed with Current Data (Quick Start)**

If you want to proceed quickly with the 13 features you have:

#### Step 2A: Extract Actions from Raw Data

Since actions aren't in the processed features yet, you need to extract them from the raw MIMIC-IV data:

```bash
python extract_actions.py \
  --cohort data/processed/cohort.csv \
  --inputevents-path /path/to/mimic-iv-3.1/icu/inputevents.csv \
  --output data/processed/
```

This will create:
- `data/processed/train_actions.csv`
- `data/processed/val_actions.csv`
- `data/processed/test_actions.csv`

#### Step 2B: Compute SOFA Scores and Rewards

With only 13 features, SOFA computation will be limited, but you can compute simplified rewards:

```bash
python compute_rewards.py \
  --features-dir data/processed/ \
  --cohort data/processed/cohort.csv \
  --output data/processed/
```

#### Step 2C: Train Q-Learning Model

```bash
python train_q_learning.py \
  --config configs/config.yaml \
  --data-dir data/processed/ \
  --output-dir results/
```

#### Step 2D: Evaluate with WDR-OPE

```bash
python evaluate_ope.py \
  --model results/q_learning_model.pkl \
  --test-data data/processed/test_*.csv \
  --output results/ope_evaluation.json
```

---

## 📋 Detailed Workflow

### Current Feature Availability

Based on your preprocessing output, you have:

**✓ Available (13 features):**
- Vitals: HR, RR, SpO2
- Fluid balance: input_4hourly, input_total, output_4hourly, output_total, cumulated_balance
- Demographics: gender, age, re_admission
- Possibly: max_dose_vaso (or one other feature)

**✗ Missing (35+ features):**
- Blood pressure (SysBP, MeanBP, DiaBP)
- Temperature, FiO2, GCS
- All chemistry labs (Glucose, Sodium, Potassium, etc.)
- Hematology (Platelets, WBC, Hb)
- Blood gas (Lactate, pH, pO2, pCO2)
- Derived scores (SOFA, SIRS)

### Impact on Model

**With 13 features:**
- ✓ Basic vital signs available
- ✓ Fluid balance tracking
- ✓ Demographics
- ✗ Cannot compute full SOFA score
- ✗ Limited physiological state representation
- ⚠️ Model performance will be suboptimal

**With 30-40 features (after adjusting threshold):**
- ✓ Complete vital signs
- ✓ Essential labs
- ✓ Blood gas values
- ✓ Can compute SOFA score
- ✓ Rich state representation
- ✓ Expected good performance

---

## 🔧 Scripts You Need to Create (If Proceeding with Option 2)

I've created the core modules, but you need integration scripts:

### 1. `extract_actions.py`

```python
#!/usr/bin/env python3
"""
Extract actions from MIMIC-IV inputevents.

Usage:
    python extract_actions.py --cohort data/processed/cohort.csv \\
                               --inputevents-path /path/to/inputevents.csv \\
                               --output data/processed/
"""

import argparse
import pandas as pd
from src.mdp import ActionExtractor
from src.utils.config_loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--inputevents-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    # Load config and cohort
    config = ConfigLoader(args.config).config
    cohort = pd.read_csv(args.cohort)

    # Extract actions
    extractor = ActionExtractor(config)
    actions = extractor.load_actions_from_events(
        args.inputevents_path,
        cohort,
        time_window_hours=4
    )

    # Load processed features to get train/val/test splits
    train_features = pd.read_csv(f"{args.output}/train_features.csv")
    val_features = pd.read_csv(f"{args.output}/val_features.csv")
    test_features = pd.read_csv(f"{args.output}/test_features.csv")

    # Split actions
    train_stays = train_features['stay_id'].unique()
    val_stays = val_features['stay_id'].unique()
    test_stays = test_features['stay_id'].unique()

    train_actions = actions[actions['stay_id'].isin(train_stays)]
    val_actions = actions[actions['stay_id'].isin(val_stays)]
    test_actions = actions[actions['stay_id'].isin(test_stays)]

    # Fit action bins on training data
    train_actions = extractor.fit_transform(train_actions, 'train')
    val_actions = extractor.transform(val_actions)
    test_actions = extractor.transform(test_actions)

    # Save
    train_actions.to_csv(f"{args.output}/train_actions.csv", index=False)
    val_actions.to_csv(f"{args.output}/val_actions.csv", index=False)
    test_actions.to_csv(f"{args.output}/test_actions.csv", index=False)
    extractor.save_bins(f"{args.output}/action_bins.pkl")

    print("✓ Actions extracted and saved")

if __name__ == "__main__":
    main()
```

### 2. `compute_rewards.py`

Similar structure for computing SOFA scores and rewards.

### 3. `train_q_learning.py`

Main training script (I'll create this below).

---

## 📁 Expected File Structure After Completion

```
RL/
├── data/
│   ├── processed/
│   │   ├── cohort.csv
│   │   ├── train_features.csv (13 features currently, 30-40 after reprocessing)
│   │   ├── train_features_normalized.csv
│   │   ├── train_actions.csv (need to create)
│   │   ├── train_rewards.csv (need to create)
│   │   ├── val_*.csv
│   │   ├── test_*.csv
│   │   ├── normalizer.pkl
│   │   └── action_bins.pkl (will be created)
│   └── intermediate/ (if saved)
├── results/
│   ├── q_learning_model.pkl
│   ├── training_history.csv
│   └── ope_evaluation.json
├── logs/
│   ├── preprocessing_*.log
│   ├── training_*.log
│   └── evaluation_*.log
└── configs/
    └── config.yaml
```

---

## 🎯 Success Criteria

### After Re-preprocessing (Option 1):
- [  ] 30-40 features retained
- [  ] SOFA score computable
- [  ] All vital signs available
- [  ] Key lab values present

### After Action Extraction:
- [  ] Actions extracted for all time windows
- [  ] 25 discrete actions (5 IV × 5 vaso)
- [  ] Action distribution reasonable

### After Training:
- [  ] Q-learning converges (loss decreases)
- [  ] Policy learned (not random)
- [  ] WDR estimate computed
- [  ] Confidence intervals reasonable

---

## 🚨 Common Issues & Solutions

### Issue 1: "FileNotFoundError: inputevents.csv"
**Solution:** Provide full path to MIMIC-IV data with `--inputevents-path`

### Issue 2: "Cannot compute SOFA - missing features"
**Solution:** Use simplified reward (just mortality) or re-run preprocessing with adjusted threshold

### Issue 3: "Out of memory during training"
**Solution:** Reduce batch size in config or use smaller learning rate

### Issue 4: "WDR estimate is NaN"
**Solution:** Check for zero behavior policy probabilities - increase softening epsilon

---

## 📚 References

1. **AI Clinician Paper:**
   Komorowski et al. (2018) "The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care"

2. **WDR-OPE:**
   Thomas & Brunskill (2016) "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning"

3. **MIMIC-IV Documentation:**
   https://mimic.mit.edu/docs/iv/

---

## 💡 Recommendations

### For Best Results:
1. ✅ **Re-run preprocessing with adjusted threshold** (Option 1)
2. ✅ Extract actions from inputevents
3. ✅ Compute SOFA-based rewards
4. ✅ Train Q-learning with hyperparameter search
5. ✅ Evaluate with WDR-OPE and bootstrap CI

### For Quick Prototype:
1. ✅ Proceed with 13 features (Option 2)
2. ✅ Use simplified rewards (mortality only)
3. ✅ Train basic Q-learning
4. ✅ Get preliminary results
5. ⚠️ Note: Performance will be limited

---

## 📞 Need Help?

Check these files:
- `analyze_data_quality.py` - Understand your data
- `PREPROCESSING_README.md` - Preprocessing documentation
- `IMPLEMENTATION_SUMMARY.md` - What's been implemented
- Code docstrings - All modules have detailed documentation

---

**Next Action:** Choose Option 1 (recommended) or Option 2, then follow the steps above.

Good luck! 🚀
