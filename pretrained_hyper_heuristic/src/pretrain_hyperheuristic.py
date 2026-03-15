"""
pretrain_hyperheuristic.py
==========================
Pre-training pipeline for AdvancedChoiceFunctionHH.
 
Architecture
------------
PRE-TRAINING
  1. Offline data collection  (run HH on training instances, log (state, action, reward))
  2. Keras surrogate          (supervised model: state → expected reward per heuristic)
  3. Q-Table warm-start       (offline RL: build Q from logged transitions)
  4. Inter-domain transfer    (normalised state so Q transfers across problem families)
  5. ε-decay schedule         (rich exploration early → exploitation at end)
  6. Reward shaping           (dense feedback beyond raw Δfitness)
  7. Tail system              (sliding-window performance memory per heuristic)
  8. Perturbation mechanism   (forced escape from local optima)
  9. Reset function           (hard reset of degenerate Q rows)
 
ONLINE DEPLOYMENT
  10. Continued Q updates      (small learning rate, small re-introduced ε)
  11. Sliding window tracker   (dynamic tail system adjustment)
  12. Transfer injection       (load pre-trained Q at construction time)

  (Note: Code format is different but same architecture)
"""

from __future__ import annotations

import math
import random
import pickle  # Serialises Python objects into binary files to be saves or reloaded
import time
import numpy as np

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import Model, Sequential, layers, regularizers
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dense, Dropout, Rescaling, BatchNormalization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---- Optional if Keras is installed (graceful degradation if not installed)
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
    print("[pretrain] TensorFlow/Keras not found - surrogate model disabled.")

# ----- Local Import -----
from python_hyper_heuristic.src.hyperheuristic import AdvancedChoiceFunctionHH, HHConfig, Phase, AcceptanceKind
from python_hyper_heuristic.domains.Python.AbstractProblem.ProblemDomain import ProblemDomain

# ============================================================================
# §1  STATE REPRESENTATION
#     Normalised feature vector fed to both Q-Table (discretised) and Keras.
# ============================================================================

@dataclass
class HHState:
    """
    Fixed-size state vector for heuristic selection.
    All values are normalised to [0, 1] for inter-domain transfer.
    """
    # --- per-heuristic features (h_count values each) ---
    reward_ema:     np.ndarray = field(default_factory=lambda: np.zeros(1))
    worsen_ema:     np.ndarray = field(default_factory=lambda: np.zeros(1))
    tail_score:     np.ndarray = field(default_factory=lambda: np.zeros(1))  # §7
    recency_norm:   np.ndarray = field(default_factory=lambda: np.zeros(1))

    # --- global scalar features ---
    phase_flag:     float = 0.0   # 0 = intensify, 1 = diversify
    stall_norm:     float = 0.0   # iterations_since_improve / stall_restart_threshold
    epsilon:        float = 0.1   # current exploration rate

    def as_vector(self) -> np.ndarray:
        """Flatten to 1-D array; used by both Q-Table bucketing and Keras."""
        return np.concatenate([
            self.reward_ema,
            self.worsen_ema,
            self.tail_score,
            self.recency_norm,
            [self.phase_flag, self.stall_norm, self.epsilon],
        ])
    
def build_state(hh: "PreTrainedHH", h_count: int) -> HHState:
    """Extract a normalised HHState from a live PreTrainedHH instance."""
    stats = hh.h_stats

    # Raw per-heuristic arrays
    rew  = np.array([s.reward_ema for s in stats], dtype=np.float32)
    wor  = np.array([s.worsen_ema for s in stats], dtype=np.float32)
    tail = np.array([hh.tail_system.score(i) for i in range(h_count)], dtype=np.float32)

    # Recency: iteartions since last used, normalised by stall threshold
    horizon = max(1, hh.cfg.stall_iterations_to_restart)
    rec = np.array([
        min(1.0, (hh.iteration - s.last_used_iter) / horizon)
        for s in stats
    ], dtype=np.float32)

    # Normalise ema arrays to [-1,1] range for stable training
    def _norm(v: np.ndarray) -> np.ndarray:
        span = v.max() - v.min()
        if span < 1e-9:
            return np.zeros_like(v)
        return (v - v.min()) / span * 2.0 - 1.0
    
    phase_flag = 1.0 if hh.phase == Phase.DIVERSIFY else 0.0
    stall_norm = min(1.0, (hh.iteration - hh.last_improve_iter) / horizon)

    return HHState(
        reward_ema=_norm(rew),
        worsen_ema=_norm(wor),
        tail_score=_norm(tail),
        recency_norm=rec,
        phase_flag=phase_flag,
        stall_norm=stall_norm,
        epsilon=hh.epsilon,
    )

# ============================================================================
# §7  TAIL SYSTEM  –  sliding-window performance memory per heuristic
# ============================================================================

class TailSystem:
    """
    Tracks a rolling window of the last `window` outcomes for each heuristic.
    Score = fraction of improvements in the window  (∈ [0,1]).
    Window length can be adjusted dynamically (§11).
    """

    def __init__(self, h_count: int, window: int = 20):
        self.h_count = h_count
        self.window = window
        self._windows: List[deque] = [deque(maxlen=window) for _ in range (h_count)]

    def record(self, h: int, improved: bool) -> None:
        self._windows[h].append(1.0 if improved else 0.0)

    def score(self, h: int) -> float:
        if h < 0 or h >= len(self._windows):
            return 0.5  # uninformative prior for out-of-range heuristic
        w = self._windows[h]
        return float(np.mean(w)) if w else 0.5  # uninformative prior = 0.5
    
    def set_window(self, new_window: int) -> None:
        """§11 - dynamically resize all windows."""
        new_window = max(5, int(new_window))
        if new_window == self.window:
            return
        self.window = new_window
        new_wins = []
        for old in self._windows:
            nd: deque = deque(maxlen=new_window)
            nd.extend(list(old)[-new_window:])
            new_wins.append(nd)
        self._windows = new_wins

    def reset_heuristic(self, h: int) -> None:
        self._windows[h].clear()
         
# ============================================================================
# §6  REWARD SHAPING
# ============================================================================

def shaped_reward(
        before: float,
        after: float,
        best_ever: float,
        accepted: bool,
        improved_global: bool,
        iter_since_improve: int,
        max_stall: int,
) -> float:
    """
    Dense reward signal with four components:
 
    R = R_delta  +  R_acceptance  +  R_global  +  R_urgency
    """
    # R1: normalised improvement delta
    span = abs(before) + 1e-9
    delta = (before - after) / span          # positive = improvement
    R_delta = math.copysign(math.log1p(abs(delta)), delta) * 2.0
 
    # R2: mild bonus just for acceptance (enables LAHC/SA escapes)
    R_acceptance = 0.1 if accepted else -0.05
 
    # R3: bonus for improving the global best
    R_global = 1.0 if improved_global else 0.0
 
    # R4: urgency shaping – reward more if close to stall/restart
    stall_ratio = min(1.0, iter_since_improve / max(1, max_stall))
    R_urgency = 0.5 * stall_ratio if delta > 0 else 0.0
 
    return R_delta + R_acceptance + R_global + R_urgency

# ============================================================================
# §3  Q-TABLE  (discrete state → action → value)
# ============================================================================

class QTable:
    """
    Simple tabular Q-learner.
 
    State is discretised via `_discretise()`.
    Works with any number of heuristics (actions).
 
    Warm-start from offline data (§3) and transferred across domains (§4)
    because the state features are normalised.
    """

    def __init__(
        self,
        h_count:    int,
        state_bins: int = 5,
        lr:         float = 0.1,
        gamma:      float = 0.95,
    ):
        self.h_count    = h_count
        self.state_bins = state_bins
        self.lr         = lr
        self.gamma      = gamma
        # table: dict  state_key → np.ndarray(h_count)
        self.table: Dict[tuple, np.ndarray] = {}

        # ------------------------------------------------------------------
    def _discretise(self, state_vec: np.ndarray) -> tuple:
        """
        Bucket each feature into `state_bins` equal bins in [-1, 1] / [0, 1].
        Returns a hashable key.
        """
        clipped = np.clip(state_vec, -1.0, 1.0)
        buckets  = ((clipped + 1.0) / 2.0 * (self.state_bins - 1)).astype(int)
        return tuple(buckets.tolist())
 
    def _get_row(self, key: tuple) -> np.ndarray:
        if key not in self.table:
            self.table[key] = np.zeros(self.h_count, dtype=np.float32)
        return self.table[key]
 
    # ------------------------------------------------------------------
    def q_values(self, state_vec: np.ndarray) -> np.ndarray:
        return self._get_row(self._discretise(state_vec)).copy()
 
    def best_action(self, state_vec: np.ndarray) -> int:
        return int(np.argmax(self.q_values(state_vec)))
 
    def update(
        self,
        state_vec:      np.ndarray,
        action:         int,
        reward:         float,
        next_state_vec: np.ndarray,
        done:           bool = False,
    ) -> None:
        key      = self._discretise(state_vec)
        next_key = self._discretise(next_state_vec)
        row      = self._get_row(key)
        next_row = self._get_row(next_key)
 
        td_target = reward + (0.0 if done else self.gamma * float(np.max(next_row)))
        row[action] += self.lr * (td_target - row[action])
 
    # §9  Reset function ─────────────────────────────────────────────────────
    def reset_action(self, action: int) -> None:
        """
        Zero out Q-values for one heuristic across all states.
        Called when a heuristic enters a degenerate regime.
        """
        for row in self.table.values():
            row[action] = 0.0
 
    def reset_state(self, state_vec: np.ndarray) -> None:
        """Zero out a single state row (avoids stuck exploitation)."""
        key = self._discretise(state_vec)
        if key in self.table:
            self.table[key][:] = 0.0
 
    # §4  Inter-domain transfer ──────────────────────────────────────────────
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "h_count":    self.h_count,
                "state_bins": self.state_bins,
                "lr":         self.lr,
                "gamma":      self.gamma,
                "table":      self.table,
            }, f)
 
    @classmethod
    def load(cls, path: str, h_count_override: Optional[int] = None) -> "QTable":
        with open(path, "rb") as f:
            d = pickle.load(f)
        h_count = h_count_override or d["h_count"]
        qt = cls(h_count=h_count, state_bins=d["state_bins"],
                 lr=d["lr"], gamma=d["gamma"])
        if h_count == d["h_count"]:
            qt.table = d["table"]
        else:
            # §4: resize rows when transferring to domain with different heuristic count
            for k, v in d["table"].items():
                new_row = np.zeros(h_count, dtype=np.float32)
                copy_len = min(h_count, len(v))
                new_row[:copy_len] = v[:copy_len]
                qt.table[k] = new_row
        return qt
    
# ============================================================================
# §2  KERAS SURROGATE MODEL
# ============================================================================

def build_surrogate_model(state_dim: int, h_count: int) -> "keras.Model":
    """
    Supervised model:  state_vec  →  Q-value vector (one per heuristic).
    Used to warm-start the Q-Table or directly guide selection.
    """
    if not HAS_KERAS:
        raise RuntimeError("TensorFlow/Keras required for surrogate model.")
 
    inp = keras.Input(shape=(state_dim,), name="state")
    x   = keras.layers.Dense(128, activation="relu")(inp)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Dense(64,  activation="relu")(x)
    x   = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(h_count, activation="linear", name="q_values")(x)
    model = keras.Model(inp, out, name="HH_Surrogate")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model
 
def train_surrogate(
    model:        "keras.Model",
    X:            np.ndarray,   # (N, state_dim)
    Y:            np.ndarray,   # (N, h_count)  – target Q vectors
    val_split:    float = 0.15,
    test_split:   float = 0.15,
    epochs:       int   = 30,
    batch_size:   int   = 64,
    random_state: int   = 42,
) -> Tuple["keras.callbacks.History", float]:
    """
    §1+§2: Three-way split (train / val / test) with shuffling for generalisation.

    Split strategy:
        - Shuffle first to remove time-ordering bias from sequential transitions
        - test_split  : held-out set, never seen during training (default 15%)
        - val_split   : used by EarlyStopping during training       (default 15%)
        - remainder   : training data                               (default 70%)

    Returns
    -------
    history   : Keras History object
    test_loss : final MSE on the held-out test set
    """
    if not HAS_KERAS:
        raise RuntimeError("TensorFlow/Keras required.")

    # --- Step 1: shuffle + carve out test set ---
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y,
        test_size=test_split,
        shuffle=True,           # removes time-ordering bias
        random_state=random_state,
    )

    # --- Step 2: carve validation from the remaining trainval block ---
    # Adjust val fraction relative to the trainval subset size
    val_fraction_of_trainval = val_split / (1.0 - test_split)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval,
        test_size=val_fraction_of_trainval,
        shuffle=True,
        random_state=random_state,
    )

    print(
        f"[train_surrogate] Split sizes — "
        f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )

    # --- Step 3: train with explicit validation data ---
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),   # explicit val set, not validation_split
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    # --- Step 4: evaluate on held-out test set ---
    test_loss = model.evaluate(X_test, Y_test, verbose=0)
    print(f"[train_surrogate] Test MSE: {test_loss:.6f}")

    return history, test_loss
 
def surrogate_to_qtable(
    model:      "keras.Model",
    qt:         QTable,
    sample_X:   np.ndarray,  # (N, state_dim) – representative state samples
) -> None:
    """
    Inject surrogate predictions into the Q-Table as a warm-start.
    For each sample state, set the Q-row to the model's prediction.
    """
    if not HAS_KERAS:
        return
    preds = model.predict(sample_X, verbose=0)          # (N, h_count)
    for vec, q_row in zip(sample_X, preds):
        key = qt._discretise(vec)
        qt.table[key] = q_row.astype(np.float32)

# ============================================================================
# §5  ε-DECAY SCHEDULE
# ============================================================================

class EpsilonSchedule:
    """
    Pre-training: fast decay from ε_start → ε_min over `decay_steps`.
    Online:       small re-introduced ε (§11) for adaptive exploration.
    """

    def __init__(
        self,
        eps_start:   float = 1.0,
        eps_min:     float = 0.02,
        decay_steps: int   = 5000,
        eps_online:  float = 0.05, # re-introduced ε for online phase
    ):
        self.eps_start   = eps_start
        self.eps_min     = eps_min
        self.decay_steps = decay_steps
        self.eps_online  = eps_online
        self._step       = 0
        self._online     = False

    def step(self) -> float:
        if self._online:
            return self.eps_online
        self._step += 1
        # Exponential decay
        frac = self._step / max(1, self.decay_steps)
        eps  = self.eps_min + (self.eps_start - self.eps_min) * math.exp(-5.0 * frac)
        return max(self.eps_min, eps)
    
    @property
    def current(self) -> float:
        return self.step()
    
    def switch_to_online(self) -> None:
        """§10: called at deployment time."""
        self._online = True

# ============================================================================
# §8  PERTURBATION MECHANISM
# ============================================================================

class PerturbationMechanism:
    """
    Detects plateau and triggers a forced perturbation.
 
    Strategy:
      - If the best fitness hasn't improved for `trigger_iters` iterations,
        force `n_perturb` random (high-diversity) heuristics.
      - Preference given to heuristics with low recent usage (recency).
    """
        
    def __init__(self, trigger_iters: int = 300, n_perturb: int = 3):
        self.trigger_iters = trigger_iters
        self.n_perturb     = n_perturb
        self._active       = False
        self._remaining    = 0

    def check(self, iter_since_improve: int) -> bool:
        """Returns True when pertubation mode should activate."""
        if iter_since_improve >= self.trigger_iters and not self._active:
            self._active    = True
            self._remaining = self.n_perturb
        return self._active
            
    def consume(self) -> bool:
        """Call each iteration during perturabtion; returns True while active."""
        if not self._active:
            return False
        self._remaining -= 1
        if self._remaining <= 0:
            self._active    = False
            self._remaining = 0
        return True
    
    def select_perturb_heuristic(self, h_count: int, h_stats, rng: random.Random, iteration: int) -> int:
        """Pick the least-recently-used heuristic as perturbation move."""
        staleness = [iteration - s.last_used_iter for s in h_stats]
        # weighted random among top-50% most scale
        top_n     = max(1, h_count // 2)
        top_idxs  = sorted(range(h_count), key=lambda i: -staleness[i])[:top_n]
        return rng.choice(top_idxs)
    
# ============================================================================
# OFFLINE DATA COLLECTION  (§1)
# ============================================================================

@dataclass
class Transition:
    """Single (s, a, r, s') tuple for offline replay."""
    state_vec:      np.ndarray
    action:         int
    reward:         float
    next_state_vec: np.ndarray
    done:           bool = False

class OfflineCollector:
    """
    Runs a vanilla HH on a set of training problem instances and records all
    (state, action, reward, next_state) transitions for offline RL.
    """

    def __init__(self, hh_factory, time_limit_ms: int = 5000):
        """
        hh_factory: callable() → AdvancedChoiceFunctionHH instance
        """
        self.hh_factory    = hh_factory
        self.time_limit_ms = time_limit_ms
        self.transitions:  List[Transition] = []

    def collect(self, problems: List[ProblemDomain]) -> List[Transition]:
        """Run the HH on each problem and record transitions."""
        all_transitions: List[Transition] = []
        for problem in problems:
            hh = self.hh_factory()
            hh.setTimeLimit(self.time_limit_ms)
            hh.loadProblemDomain(problem)
            # Monkey-patch _solve to intercept transitions # Might remove comment
            all_transitions.extend(self._run_and_collect(hh, problem))
        self.transitions = all_transitions
        return all_transitions

    def _run_and_collect(self, hh: "PreTrainedHH", problem: ProblemDomain) -> List[Transition]:
        """
        Thin wrapper around _solve that records transition.
        PreTrainedHH exposes hooks for this.
        """
        transitions: List[Transition] = []
        hh._transition_hook = lambda t: transitions.append(t)
        hh.run()
        hh._transition_hook = None
        return transitions

    def to_numpy(self, h_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_states, actions, rewards, X_next_states)"""
        if not self.transitions:
            raise ValueError("No transitions collected yet.")
        X     = np.stack([t.state_vec      for t in self.transitions])
        A     = np.array([t.action         for t in self.transitions], dtype=int)
        R     = np.array([t.reward         for t in self.transitions], dtype=np.float32)
        X_nxt = np.stack([t.next_state_vec for t in self.transitions])
        return X, A, R, X_nxt

    def build_q_targets(self, qt: QTable, gamma: float = 0.95) -> np.ndarray:
        """
        Build target Q-vector for each transition (for Keras training)
        Y[i, a] = R + gamma * max Q(s)
        """
        X, A, R, X_nxt = self.to_numpy(qt.h_count)
        Y = np.array([qt.q_values(s) for s in X], dtype=np.float32)
        for i, (a, r, s_nxt, done) in enumerate(zip(A, R, X_nxt, [t.done for t in self.transitions])):
            td = r if done else r + gamma * float(np.max(qt.q_values(s_nxt)))
            Y[i, a] = td
        return Y
    
# ============================================================================
# PRE-TRAINING ORCHESTRATOR  (ties §1-§9 together)
# ============================================================================

class PreTrainer:
    """
    Orchestrates the full pre-training pipeline:

    1. Collect offline data from training problem instances
    2. Warm-start Q-Table from offline transactions (offline RL)
    3. Optionally train Keras surrogate and inject into Q-Table
    4. Return a ready-to-deploy PreTrainedHH
    """

    def __init__(
        self,
        h_count:          int,
        hh_factory,          # callable() → PreTrainedHH
        state_dim:        int,
        qtable_bins:      int    = 5,
        qtable_lr:        float = 0.15,
        qtable_gamma:     float = 0.95,
        eps_start:        float = 1.0,
        eps_min:          float = 0.02,
        decay_steps:      int   = 5000,
        tail_window:      int   = 20,
        surrogate_epochs: int   = 30,
        use_surrogate:    bool  = True,
    ):
        self.h_count            = h_count
        self.hh_factory         = hh_factory
        self.state_dim          = state_dim
        self.use_surrogate      = use_surrogate and HAS_KERAS

        self.qt = QTable(h_count, state_bins=qtable_bins, lr=qtable_lr, gamma=qtable_gamma)
        self.eps_schedule = EpsilonSchedule(eps_start=eps_start, eps_min=eps_min, decay_steps=decay_steps)
        self.tail = TailSystem(h_count, window=tail_window)

        self.surrogate_model = (build_surrogate_model(state_dim, h_count) if self.use_surrogate else None)
        self.surrogate_epochs = surrogate_epochs

    # -----------------------------------------------------------------------------------------------------
    def run_offline(
        self,
        training_problems: List[ProblemDomain],
        time_limit_ms:     int = 5000,
    ) -> None:
        """§1: Collect data + §3: offline Q updates + §2: Keras fit."""
        print("[PreTrainer] Collecting offline transitions...")
        collector = OfflineCollector(self.hh_factory, time_limit_ms=time_limit_ms)
        transitions = collector.collect(training_problems)
        print(f"[PreTrainer] Collected {len(transitions)} transitions.")

        # §3: offline Q-Table warm-start (TD updates)
        print("[PreTrainer] Warm-starting Q-Table from offline data...")
        for t in transitions:
            self.qt.update(
                state_vec=t.state_vec,
                action=t.action,
                reward=t.reward,
                next_state_vec=t.next_state_vec,
                done=t.done,
            )

        # §2+§1: Keras surrogate train/test split
        if self.use_surrogate and self.surrogate_model is not None:
            print("[PreTrainer] Training Keras surrogate model...")
            X, A, R, X_nxt = collector.to_numpy(self.h_count)
            Y = collector.build_q_targets(self.qt, gamma=self.qt.gamma)

            history, test_loss = train_surrogate(self.surrogate_model, X, Y, epochs=self.surrogate_epochs) # Need to tweak history, give a purpose
            print(f"[PreTrainer] Surrogate test MSE: {test_loss:.6f}")
            
            # Inject surrogate predictions back into Q-Table (warm-start overlap)
            surrogate_to_qtable(self.surrogate_model, self.qt, X)
            print("[PreTrainer] Surrogate injected into Q-Table.")

        print("[PreTrainer] Pre-training complete.")

    def save_qtable(self, path: str) -> None:
        """Saves Qtable"""
        self.qt.save(path)
        print(f"[PreTrainer] Q-Table saved to {path}.")

    def make_deployment_hh(self, config: Optional[HHConfig] = None) -> "PreTrainedHH":
        """Return a PreTrainedHH loaded with the pre-trained Q-Table."""
        hh = PreTrainedHH(config=config, qtable=self.qt)
        hh.tail_system = self.tail
        hh.eps_schedule = self.eps_schedule
        hh.eps_schedule.switch_to_online()       # §10: small re-introduced ε
        return hh

# ============================================================================
# PreTrainedHH  –  extends AdvancedChoiceFunctionHH with Q-Table selection,
#                  tail system, perturbation, and online updates.
# ============================================================================

class PreTrainedHH(AdvancedChoiceFunctionHH):
    """
    Drop-in replacement for AdvancedChoiceFunctionHH with:
    - Q-Table guided heuristic selection (with ε-greedy)
    - Tail system integration
    - Perturbation mechanism
    - Online Q updates (§10)
    - Sliding-window tracking (§11)
    - Transition hook for offline data collection
    """

    def __init__(
            self,
            seed:        Optional[int]      = None,
            config:      Optional[HHConfig] = None,
            qtable:      Optional[QTable]   = None,
            eps_start:   float = 0.3,      # lower default for online
            eps_min:     float = 0.02,
            decay_steps: int   = 2000,
    ):
        super().__init__(seed=seed, config=config)
        self.qtable: Optional[QTable] = qtable

        self.eps_schedule = EpsilonSchedule(
            eps_start=eps_start,
            eps_min=eps_min,
            decay_steps=decay_steps,
            eps_online=0.05,
        )
        self.epsilon: float = eps_start

        self.tail_system: Optional[TailSystem] = None   # set after _init_stats
        self.perturb = PerturbationMechanism(
            trigger_iters=self.cfg.stall_iterations_to_diversify // 2,
            n_perturb=4,
        )

        # §11: sliding window performance tracker (global, for dynamic tail)
        self.sliding_window: deque = deque(maxlen=50)
        self.sliding_variance: float = 0.0

        # Hook for offline data collection (set by OfflineCollector)
        self._transition_hook = None

    # ------------------------------------------------------------------
    # Override _init_stats to also initialise tail system
    # ------------------------------------------------------------------
    def _init_stats(self, h_count: int) -> None:
        super()._init_stats(h_count)
        if self.tail_system is None:
            self.tail_system = TailSystem(h_count, window=20)
        elif self.tail_system.h_count != h_count:
            # Tail system was pre-built with a different h_count — resize it
            old_windows = self.tail_system._windows
            self.tail_system = TailSystem(h_count, window=self.tail_system.window)
            # Preserve any existing windows
            for i in range(min(h_count, len(old_windows))):
                self.tail_system._windows[i] = old_windows[i]

    # ------------------------------------------------------------------
    # §10+§11: Override _select_heuristic
    # ------------------------------------------------------------------
    def _select_heuristic(self, h_count: int, prev_h) -> int:
        # Update ε
        self.epsilon = self.eps_schedule.step()

        # §8: Perturbation override
        stall = self.iteration - self.last_improve_iter
        if self.perturb.check(stall) and self.perturb.consume():
            return self.perturb.select_perturb_heuristic(
                h_count, self.h_stats, self.rng, self.iteration
            )

        # ε-greedy: random exploration
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(h_count)

        # Q-Table guided selection (if available)
        if self.qtable is not None:
            state_vec = build_state(self, h_count).as_vector()
            tabu_set  = set(self.tabu)
            q_vals    = self.qtable.q_values(state_vec).copy()

            # Blend Q-values with CF scores for robustness
            for h in range(h_count):
                cf_score = self._choice_function_score(h, prev_h)
                q_vals[h] = 0.6 * q_vals[h] + 0.4 * cf_score
                if h in tabu_set:
                    q_vals[h] = -float("inf")

            return int(np.argmax(q_vals))
 
        # Fallback: parent CF selection
        return super()._select_heuristic(h_count, prev_h)

    # ------------------------------------------------------------------
    # Override _credit_assignment to include Q-Table update + tail
    # ------------------------------------------------------------------
    def _credit_assignment(self, h, prev_h, before, after, accepted, improved) -> None:
        # Parent CF update
        super()._credit_assignment(h, prev_h, before, after, accepted, improved)

        # §7: Tail system update
        if self.tail_system is not None:
            self.tail_system.record(h, improved)

        # §11: Sliding window
        self.sliding_window.append(1.0 if improved else 0.0)
        if len(self.sliding_window) >= 10:
            arr = np.array(self.sliding_window)
            self.sliding_variance = float(np.var(arr))
            # Dynamic tail window: widen when variance is high (unstable search)
            new_win = 20 + int(self.sliding_variance * 60)
            if self.tail_system is not None:
                self.tail_system.set_window(new_win)

        # §10: Online Q-Table update
        if self.qtable is not None:
            h_count   = len(self.h_stats)
            state_vec = build_state(self, h_count).as_vector()
            reward    = shaped_reward(
                before=before, after=after,
                best_ever=self.best_fitness,
                accepted=accepted,
                improved_global=(after < self.best_fitness),
                iter_since_improve=self.iteration - self.last_improve_iter,
                max_stall=self.cfg.stall_iterations_to_restart,
            )
            # We can't get next_state until next iteration; approximate with current
            next_state_vec = state_vec # approximation; replace with deferred update if needed
            self.qtable.update(state_vec, h, reward, next_state_vec)

            # Fire transition hook (for offline collection)
            if self._transition_hook is not None:
                self._transition_hook(Transition(
                    state_vec=state_vec,
                    action=h,
                    reward=reward,
                    next_state_vec=next_state_vec
                ))

        # §9: Reset degenerate heuristic rows
        if self.qtable is not None:
            hs = self.h_stats[h]
            if hs.uses > 30 and hs.accepts / max(1, hs.uses) < 0.03:
                # This heuristic is being selected but also never accepted → reset
                self.qtable.reset_action(h)
                if self.tail_system is not None:
                    self.tail_system.reset_heuristic(h)

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return "Pre-Trained Q-Table + CF Hyper-Heuristic"

# ============================================================================
# CONVENIENCE: full pipeline in one call
# ============================================================================

def pretrain_and_deploy(
    training_problems:  List[ProblemDomain],
    h_count:            int,
    state_dim:          int,
    time_limit_ms:      int           = 30000,
    n_pretrain_runs:    int           = 5,        
    qtable_save_path:   Optional[str] = None,
    use_surrogate:      bool          = True,
    config:             Optional[HHConfig] = None,
) -> PreTrainedHH:
    """
    One-liner convenience wrapper:

        hh = pretrain_and_deploy(training_problems, h_count=8, state_dim=35,
                                 n_pretrain_runs=5)
        hh.setTimeLimit(30000)
        hh.loadProblemDomain(test_problem)
        hh.run()
        print(hh.getBestSolutionValue())
    """
    def _factory():
        return PreTrainedHH(config=config)

    trainer = PreTrainer(
        h_count=h_count,
        hh_factory=_factory,
        state_dim=state_dim,
        use_surrogate=use_surrogate and HAS_KERAS,
    )

    # Run offline collection n_pretrain_runs times
    # Q-Table accumulates experience across all runs
    for run_idx in range(n_pretrain_runs):
        print(f"[pretrain_and_deploy] Pre-training run {run_idx + 1}/{n_pretrain_runs} ...")
        trainer.run_offline(training_problems, time_limit_ms=time_limit_ms)

    if qtable_save_path:
        trainer.save_qtable(qtable_save_path)

    return trainer.make_deployment_hh(config=config)
