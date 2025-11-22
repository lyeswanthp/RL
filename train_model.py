#!/usr/bin/env python3
"""
Main Training Pipeline for AI Clinician

This script orchestrates the complete training pipeline:
1. Load preprocessed features (with embedded actions)
2. Discretize actions into 25-bin action space
3. Compute rewards from SOFA changes and mortality
4. Build MDP trajectories
5. Train Linear Q-Learning model
6. Evaluate with WDR-OPE

Usage:
    python train_model.py --data-dir data/processed/ --output-dir results/
    python train_model.py --data-dir data/processed/ --output-dir results/ --epochs 200
"""

import argparse
import logging
import sys
import pickle
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import ConfigLoader
from src.mdp.reward_computation import RewardComputer
from src.mdp.trajectory_builder import TrajectoryBuilder
from src.rl.q_learning import LinearQLearning
from src.rl.policy import GreedyPolicy, BehaviorPolicy
from src.ope.wdr import WeightedDoublyRobust

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class ActionDiscretizer:
    """
    Discretizes continuous IV fluid and vasopressor doses into discrete action bins.

    Action Space: 25 discrete actions (5 IV bins x 5 vasopressor bins)
    """

    def __init__(self, n_iv_bins: int = 5, n_vaso_bins: int = 5):
        self.n_iv_bins = n_iv_bins
        self.n_vaso_bins = n_vaso_bins
        self.n_actions = n_iv_bins * n_vaso_bins

        self.iv_bins = None
        self.vaso_bins = None
        self.fitted = False

    def fit(self, iv_values: np.ndarray, vaso_values: np.ndarray) -> 'ActionDiscretizer':
        """
        Fit action bins using percentile-based discretization on training data.

        Args:
            iv_values: IV fluid amounts (input_4hourly)
            vaso_values: Vasopressor doses (max_dose_vaso)
        """
        logger.info("Fitting action discretizer...")

        # IV fluid bins: use percentiles on non-zero values
        iv_nonzero = iv_values[iv_values > 0]
        if len(iv_nonzero) > 0:
            percentiles = [0, 25, 50, 75, 100]
            iv_percentile_values = np.percentile(iv_nonzero, percentiles)
            # Bins: [0, p25, p50, p75, max, inf]
            self.iv_bins = np.array([0] + list(iv_percentile_values[1:]) + [np.inf])
        else:
            self.iv_bins = np.array([0, np.inf])

        logger.info(f"  IV fluid bins: {self.iv_bins[:-1]}")

        # Vasopressor bins: use percentiles on non-zero values
        vaso_nonzero = vaso_values[vaso_values > 0]
        if len(vaso_nonzero) > 0:
            vaso_percentile_values = np.percentile(vaso_nonzero, percentiles)
            self.vaso_bins = np.array([0] + list(vaso_percentile_values[1:]) + [np.inf])
        else:
            self.vaso_bins = np.array([0, np.inf])

        logger.info(f"  Vasopressor bins: {self.vaso_bins[:-1]}")

        self.fitted = True
        return self

    def transform(self, iv_values: np.ndarray, vaso_values: np.ndarray) -> np.ndarray:
        """
        Transform continuous actions to discrete action indices.

        Action index = iv_bin * n_vaso_bins + vaso_bin
        """
        if not self.fitted:
            raise ValueError("ActionDiscretizer must be fitted before transform")

        # Discretize IV fluids (bin 0 = no IV, bins 1-4 = quartiles)
        iv_bins = np.digitize(iv_values, self.iv_bins[1:-1], right=False)

        # Discretize vasopressors
        vaso_bins = np.digitize(vaso_values, self.vaso_bins[1:-1], right=False)

        # Combine into single action index
        actions = iv_bins * self.n_vaso_bins + vaso_bins

        # Clip to valid range
        actions = np.clip(actions, 0, self.n_actions - 1)

        return actions.astype(int)

    def fit_transform(self, iv_values: np.ndarray, vaso_values: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(iv_values, vaso_values).transform(iv_values, vaso_values)

    def save(self, path: str):
        """Save fitted bins to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'iv_bins': self.iv_bins,
                'vaso_bins': self.vaso_bins,
                'n_iv_bins': self.n_iv_bins,
                'n_vaso_bins': self.n_vaso_bins,
                'n_actions': self.n_actions
            }, f)
        logger.info(f"Action bins saved to {path}")

    def load(self, path: str):
        """Load fitted bins from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.iv_bins = data['iv_bins']
        self.vaso_bins = data['vaso_bins']
        self.n_iv_bins = data['n_iv_bins']
        self.n_vaso_bins = data['n_vaso_bins']
        self.n_actions = data['n_actions']
        self.fitted = True
        logger.info(f"Action bins loaded from {path}")


def load_data(data_dir: str):
    """
    Load preprocessed feature data.

    Args:
        data_dir: Directory containing train/val/test_features.csv

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Loading data from {data_dir}...")

    data_path = Path(data_dir)

    # Try normalized features first, fall back to regular
    train_file = data_path / "train_features_normalized.csv"
    if not train_file.exists():
        train_file = data_path / "train_features.csv"

    val_file = data_path / "val_features_normalized.csv"
    if not val_file.exists():
        val_file = data_path / "val_features.csv"

    test_file = data_path / "test_features_normalized.csv"
    if not test_file.exists():
        test_file = data_path / "test_features.csv"

    logger.info(f"  Loading {train_file}...")
    train_df = pd.read_csv(train_file)
    logger.info(f"    Train: {len(train_df):,} rows, {len(train_df.columns)} columns")

    logger.info(f"  Loading {val_file}...")
    val_df = pd.read_csv(val_file)
    logger.info(f"    Val: {len(val_df):,} rows, {len(val_df.columns)} columns")

    logger.info(f"  Loading {test_file}...")
    test_df = pd.read_csv(test_file)
    logger.info(f"    Test: {len(test_df):,} rows, {len(test_df.columns)} columns")

    return train_df, val_df, test_df


def prepare_trajectories(
    df: pd.DataFrame,
    action_discretizer: ActionDiscretizer,
    config: dict,
    fit_discretizer: bool = False
) -> pd.DataFrame:
    """
    Prepare trajectories with discretized actions and rewards.

    Args:
        df: Feature DataFrame with columns including:
            - stay_id, time_window (identifiers)
            - input_4hourly (IV fluid action)
            - max_dose_vaso (vasopressor action)
            - SOFA (for rewards)
            - hospital_expire_flag (mortality, optional)
        action_discretizer: ActionDiscretizer instance
        config: Configuration dictionary
        fit_discretizer: Whether to fit the discretizer (True for training data)

    Returns:
        DataFrame with added columns: action, reward, done
    """
    logger.info("Preparing trajectories...")

    df = df.copy()

    # Ensure required columns exist
    iv_col = 'input_4hourly'
    vaso_col = 'max_dose_vaso'

    if iv_col not in df.columns:
        logger.warning(f"  {iv_col} not found, using zeros")
        df[iv_col] = 0

    if vaso_col not in df.columns:
        logger.warning(f"  {vaso_col} not found, using zeros")
        df[vaso_col] = 0

    # Fill NaN in action columns with 0
    df[iv_col] = df[iv_col].fillna(0)
    df[vaso_col] = df[vaso_col].fillna(0)

    # Discretize actions
    iv_values = df[iv_col].values
    vaso_values = df[vaso_col].values

    if fit_discretizer:
        df['action'] = action_discretizer.fit_transform(iv_values, vaso_values)
    else:
        df['action'] = action_discretizer.transform(iv_values, vaso_values)

    # Log action distribution
    action_counts = df['action'].value_counts().sort_index()
    logger.info(f"  Action distribution (top 5):")
    for action_id, count in action_counts.head(5).items():
        pct = count / len(df) * 100
        logger.info(f"    Action {action_id}: {count:,} ({pct:.1f}%)")

    # Sort by stay and time
    df = df.sort_values(['stay_id', 'time_window']).reset_index(drop=True)

    # Compute rewards using SOFA changes
    reward_config = config.get('reward', {})
    terminal_survival = reward_config.get('terminal_survival', 15)
    terminal_death = reward_config.get('terminal_death', -15)
    sofa_decrease_reward = reward_config.get('intermediate_sofa_decrease', 0.1)
    sofa_increase_penalty = reward_config.get('intermediate_sofa_increase', -0.25)

    # SOFA change within each stay
    df['sofa_prev'] = df.groupby('stay_id')['SOFA'].shift(1)
    df['sofa_change'] = df['SOFA'] - df['sofa_prev']

    # Mark terminal states
    df['done'] = False
    last_time_windows = df.groupby('stay_id')['time_window'].transform('max')
    df.loc[df['time_window'] == last_time_windows, 'done'] = True

    # Initialize rewards
    df['reward'] = 0.0

    # Terminal rewards (if mortality info available)
    if 'hospital_expire_flag' in df.columns:
        terminal_mask = df['done']
        survived_mask = terminal_mask & (df['hospital_expire_flag'] == 0)
        died_mask = terminal_mask & (df['hospital_expire_flag'] == 1)

        df.loc[survived_mask, 'reward'] = terminal_survival
        df.loc[died_mask, 'reward'] = terminal_death

        logger.info(f"  Terminal rewards: {survived_mask.sum():,} survivors (+{terminal_survival}), "
                   f"{died_mask.sum():,} deaths ({terminal_death})")
    else:
        logger.warning("  No hospital_expire_flag found - using only intermediate rewards")

    # Intermediate rewards based on SOFA change
    intermediate_mask = ~df['done'] & df['sofa_prev'].notna()

    # SOFA decreased (improvement)
    improved_mask = intermediate_mask & (df['sofa_change'] < 0)
    df.loc[improved_mask, 'reward'] = (
        df.loc[improved_mask, 'sofa_change'].abs() * sofa_decrease_reward
    )

    # SOFA increased (worsening)
    worsened_mask = intermediate_mask & (df['sofa_change'] > 0)
    df.loc[worsened_mask, 'reward'] = (
        -df.loc[worsened_mask, 'sofa_change'] * abs(sofa_increase_penalty)
    )

    logger.info(f"  Intermediate rewards: {improved_mask.sum():,} improved, "
               f"{worsened_mask.sum():,} worsened")
    logger.info(f"  Reward range: [{df['reward'].min():.2f}, {df['reward'].max():.2f}]")

    return df


def get_state_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of state feature columns (exclude identifiers and derived columns).
    """
    exclude_cols = {
        'stay_id', 'time_window', 'action', 'reward', 'done',
        'hospital_expire_flag', 'sofa_prev', 'sofa_change',
        'iv_bin', 'vaso_bin', 'return', 'is_terminal'
    }

    state_cols = [col for col in df.columns
                  if col not in exclude_cols and not col.startswith('next_')]

    return state_cols


def create_training_arrays(
    trajectories: pd.DataFrame,
    state_cols: list
) -> tuple:
    """
    Create numpy arrays for training.

    Returns:
        Tuple of (states, actions, rewards, next_states, dones)
    """
    # Sort and filter non-terminal states for training
    trajectories = trajectories.sort_values(['stay_id', 'time_window']).reset_index(drop=True)

    # Create next-state columns
    for col in state_cols:
        trajectories[f'next_{col}'] = trajectories.groupby('stay_id')[col].shift(-1)

    # Filter to non-terminal transitions (need next state)
    train_data = trajectories[~trajectories['done']].copy()

    # Extract arrays
    states = train_data[state_cols].values.astype(np.float32)
    actions = train_data['action'].values.astype(np.int32)
    rewards = train_data['reward'].values.astype(np.float32)

    next_state_cols = [f'next_{col}' for col in state_cols]
    next_states = train_data[next_state_cols].values.astype(np.float32)

    # Done flags (all False since we filtered terminal states)
    dones = np.zeros(len(train_data), dtype=np.float32)

    # Handle NaN values
    states = np.nan_to_num(states, nan=0.0)
    next_states = np.nan_to_num(next_states, nan=0.0)
    rewards = np.nan_to_num(rewards, nan=0.0)

    logger.info(f"  States shape: {states.shape}")
    logger.info(f"  Actions shape: {actions.shape}")
    logger.info(f"  Rewards shape: {rewards.shape}")

    return states, actions, rewards, next_states, dones


def evaluate_policy(
    model: LinearQLearning,
    trajectories_df: pd.DataFrame,
    state_cols: list,
    n_actions: int,
    gamma: float = 0.99
) -> dict:
    """
    Evaluate learned policy using WDR-OPE.

    Args:
        model: Trained Q-learning model
        trajectories_df: Test trajectories
        state_cols: State feature columns
        n_actions: Number of actions
        gamma: Discount factor

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating policy with WDR-OPE...")

    # Prepare data
    trajectories_df = trajectories_df.sort_values(['stay_id', 'time_window']).reset_index(drop=True)

    states = trajectories_df[state_cols].values.astype(np.float32)
    states = np.nan_to_num(states, nan=0.0)
    actions = trajectories_df['action'].values.astype(np.int32)
    rewards = trajectories_df['reward'].values.astype(np.float32)

    # Target policy: greedy from learned Q-function
    target_policy = GreedyPolicy(model)
    greedy_actions = target_policy.select_actions_batch(states)

    # Target policy probabilities (deterministic: 1 for greedy action, 0 otherwise)
    target_probs = (greedy_actions == actions).astype(np.float32)
    # Soften slightly to avoid zero probabilities
    target_probs = target_probs * 0.99 + 0.01 / n_actions

    # Behavior policy from data
    behavior_policy = BehaviorPolicy(n_actions=n_actions, softening_epsilon=0.01)
    behavior_policy.fit(states, actions)
    behavior_probs = behavior_policy.get_action_probs_batch(states, actions)

    # Q-values for observed actions
    q_values = model.predict_q_values(states, actions)

    # WDR evaluation
    wdr = WeightedDoublyRobust(gamma=gamma)
    wdr_value, wdr_info = wdr.estimate_value(
        trajectories_df[['stay_id', 'time_window']],
        target_probs,
        behavior_probs,
        q_values,
        rewards
    )

    # Policy agreement with clinicians
    agreement = np.mean(greedy_actions == actions)

    # Q-value statistics
    all_q_values = model.predict_q_values(states)

    results = {
        'wdr_value': wdr_value,
        'wdr_std': wdr_info['wdr_std'],
        'wdr_ci_95': wdr_info['wdr_ci_95'],
        'n_trajectories': wdr_info['n_trajectories'],
        'policy_agreement': agreement,
        'mean_q_value': np.mean(q_values),
        'max_q_value': np.mean(np.max(all_q_values, axis=1)),
        'min_q_value': np.mean(np.min(all_q_values, axis=1)),
    }

    logger.info(f"  WDR Value: {wdr_value:.3f} +/- {wdr_info['wdr_ci_95']:.3f}")
    logger.info(f"  Policy Agreement with Clinicians: {agreement*100:.1f}%")
    logger.info(f"  Mean Q-value: {results['mean_q_value']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train AI Clinician RL model')
    parser.add_argument('--data-dir', required=True, help='Directory with processed features')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--l2-lambda', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('--early-stopping', type=int, default=20, help='Early stopping patience')
    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'metrics').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    logger.info("="*80)
    logger.info("AI CLINICIAN - MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"L2 lambda: {args.l2_lambda}")

    # Load configuration
    logger.info("\nLoading configuration...")
    config = ConfigLoader(args.config).config
    gamma = config.get('mdp', {}).get('gamma', 0.99)
    n_actions = config.get('action_space', {}).get('total_actions', 25)

    # Load data
    logger.info("\n" + "="*80)
    logger.info("LOADING DATA")
    logger.info("="*80)
    train_df, val_df, test_df = load_data(args.data_dir)

    # Initialize action discretizer
    action_discretizer = ActionDiscretizer(n_iv_bins=5, n_vaso_bins=5)

    # Prepare trajectories
    logger.info("\n" + "="*80)
    logger.info("PREPARING TRAJECTORIES")
    logger.info("="*80)

    logger.info("\nProcessing training data...")
    train_traj = prepare_trajectories(train_df, action_discretizer, config, fit_discretizer=True)

    logger.info("\nProcessing validation data...")
    val_traj = prepare_trajectories(val_df, action_discretizer, config, fit_discretizer=False)

    logger.info("\nProcessing test data...")
    test_traj = prepare_trajectories(test_df, action_discretizer, config, fit_discretizer=False)

    # Save action discretizer
    action_discretizer.save(str(output_dir / 'models' / 'action_discretizer.pkl'))

    # Get state feature columns
    state_cols = get_state_feature_columns(train_traj)
    logger.info(f"\nState features: {len(state_cols)} columns")

    # Create training arrays
    logger.info("\n" + "="*80)
    logger.info("CREATING TRAINING ARRAYS")
    logger.info("="*80)

    logger.info("\nTraining set:")
    train_states, train_actions, train_rewards, train_next_states, train_dones = \
        create_training_arrays(train_traj, state_cols)

    logger.info("\nValidation set:")
    val_states, val_actions, val_rewards, val_next_states, val_dones = \
        create_training_arrays(val_traj, state_cols)

    # Initialize and train model
    logger.info("\n" + "="*80)
    logger.info("TRAINING MODEL")
    logger.info("="*80)

    n_state_features = train_states.shape[1]
    logger.info(f"\nModel configuration:")
    logger.info(f"  State features: {n_state_features}")
    logger.info(f"  Actions: {n_actions}")
    logger.info(f"  Gamma: {gamma}")

    model = LinearQLearning(
        n_state_features=n_state_features,
        n_actions=n_actions,
        learning_rate=args.learning_rate,
        gamma=gamma,
        l2_lambda=args.l2_lambda,
        random_seed=42
    )

    # Train
    training_result = model.fit(
        train_states, train_actions, train_rewards, train_next_states, train_dones,
        val_states, val_actions, val_rewards, val_next_states, val_dones,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping,
        verbose=True
    )

    # Save model
    model_path = str(output_dir / 'models' / 'q_learning_model.pkl')
    model.save(model_path)

    # Evaluate on test set
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*80)

    test_results = evaluate_policy(model, test_traj, state_cols, n_actions, gamma)

    # Also evaluate on validation set for comparison
    logger.info("\nEvaluating on validation set...")
    val_results = evaluate_policy(model, val_traj, state_cols, n_actions, gamma)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'data_dir': args.data_dir,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'l2_lambda': args.l2_lambda,
            'gamma': gamma,
            'n_state_features': n_state_features,
            'n_actions': n_actions,
        },
        'training': {
            'best_val_loss': training_result.get('best_val_loss'),
            'n_epochs_trained': len(training_result.get('history', [])),
        },
        'validation': val_results,
        'test': test_results,
        'data_stats': {
            'train_samples': len(train_states),
            'val_samples': len(val_states),
            'test_trajectories': test_traj['stay_id'].nunique(),
        }
    }

    results_path = output_dir / 'metrics' / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nFinal Results:")
    logger.info(f"  Test WDR Value: {test_results['wdr_value']:.3f} +/- {test_results['wdr_ci_95']:.3f}")
    logger.info(f"  Policy Agreement: {test_results['policy_agreement']*100:.1f}%")
    logger.info(f"  Model saved to: {model_path}")
    logger.info(f"  Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
