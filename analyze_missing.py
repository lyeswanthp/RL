#!/usr/bin/env python3
"""Analyze missing value percentages for different thresholds"""

import pandas as pd
import sys

def main():
    # Load the raw features (before cleaning)
    print('Loading features_raw.csv...')
    df = pd.read_csv('data/intermediate/features_raw.csv')

    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

    print('\n' + '='*80)
    print('MISSING VALUE ANALYSIS')
    print('='*80)
    print(f'Total observations: {len(df):,}')
    print(f'\n--- Features by Missing Percentage ---\n')

    # Show all features with their missing percentages
    for feature, pct in missing_pct.items():
        if feature not in ['stay_id', 'time_window']:
            status_75 = '❌ DROP' if pct > 75 else '✓ KEEP'
            status_90 = '❌ DROP' if pct > 90 else '✓ KEEP'

            if pct > 0:
                print(f'{feature:25s}: {pct:6.2f}%  |  @75%: {status_75:8s}  |  @90%: {status_90:8s}')

    # Summary
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)

    features_dropped_75 = (missing_pct > 75).sum()
    features_dropped_90 = (missing_pct > 90).sum()
    features_kept_75 = len(missing_pct) - features_dropped_75 - 2  # Exclude stay_id, time_window
    features_kept_90 = len(missing_pct) - features_dropped_90 - 2

    print(f'At 75% threshold: {features_kept_75} features kept, {features_dropped_75} dropped')
    print(f'At 90% threshold: {features_kept_90} features kept, {features_dropped_90} dropped')
    print(f'\nDifference: +{features_kept_90 - features_kept_75} additional features retained at 90%')

    # Show which critical features would be recovered
    print('\n' + '='*80)
    print('CRITICAL FEATURES: 75% → 90% COMPARISON')
    print('='*80)
    critical_features = ['max_dose_vaso', 'Arterial_lactate', 'WBC_count', 'Platelets_count',
                         'Temp_C', 'Arterial_pH', 'paCO2', 'paO2', 'Arterial_BE',
                         'INR', 'PT', 'Total_bili', 'SGOT', 'SGPT', 'SOFA', 'SIRS']

    recovered = []
    still_dropped = []

    for feat in critical_features:
        if feat in missing_pct:
            pct = missing_pct[feat]
            if 75 < pct <= 90:
                recovered.append(f'{feat:25s}: {pct:6.2f}% missing - ✓ RECOVERED')
            elif pct > 90:
                still_dropped.append(f'{feat:25s}: {pct:6.2f}% missing - ❌ STILL DROPPED')

    if recovered:
        print('\n✓ RECOVERED FEATURES:')
        for item in recovered:
            print(f'  {item}')

    if still_dropped:
        print('\n❌ STILL TOO SPARSE:')
        for item in still_dropped:
            print(f'  {item}')

    print('\n' + '='*80)

if __name__ == '__main__':
    main()
