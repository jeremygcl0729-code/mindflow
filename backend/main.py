"""
MindFlow API v5 — Non-Homogeneous Markov Chain Cognitive Model

Mathematical features (v5):
  - Sigmoidal transition modifiers (logistic, not linear)
  - State-dependent circadian sensitivity
  - Flow-entry warmup (attention ramp-up)
  - Flow deepening with sudden collapse (flow inertia + tipping point)
  - Cognitive momentum (fatigue acceleration when fatigue is rising)
  - Biexponential recovery (fast 2-min sympathetic + slow 120-min parasympathetic)
  - Intervention sensitivity (break effectiveness drops as fatigue rises)
  - Cognitive capacity ceiling (diminishing returns beyond total load threshold)
  - Temporal cognitive drain (flow erosion, fatigue gravity, recovery resistance)
  - Asymmetric transitions with micro-recovery and micro-regression
  - Custom initial state + attention residue support
  - Optimal break duration with biexponential inversion

Matches src/utils/markovEngine.js v5 exactly.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import math
import numpy as np
from typing import Optional, List

app = FastAPI(title="MindFlow API", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Constants
# =============================================================================

# -- Base transition matrix (v5 asymmetric + micro-recovery + micro-regression) --
P_BASE = np.array([
    [0.80, 0.15, 0.05, 0.00],  # From Flow
    [0.20, 0.60, 0.20, 0.00],  # From Distracted
    [0.04, 0.15, 0.78, 0.03],  # From Fatigued  (v5: micro-recovery)
    [0.65, 0.10, 0.02, 0.23],  # From Recovery (v5: micro-regression)
])

INITIAL_STATE = np.array([1.0, 0.0, 0.0, 0.0])

# -- Warmup --
WARMUP_TICKS = 3
WARMUP_TAU = 1.5
WARMUP_MIN = 0.70

# -- Biexponential recovery --
RECOVERY_TAU_FAST = 2.0      # minutes — acute sympathetic recovery
RECOVERY_TAU_SLOW = 120.0    # minutes — deep parasympathetic recovery
RECOVERY_WEIGHT_FAST = 0.40  # 40% fast component
RECOVERY_WEIGHT_SLOW = 0.60  # 60% slow component

# -- Flow inertia --
FLOW_INERTIA_BUILD = 0.06     # per-tick flow anchor strengthening
FLOW_INERTIA_MAX = 1.60       # max 60% stronger than baseline
FLOW_COLLAPSE_THRESHOLD = 12  # ~2 hours — tipping point (ticks)
FLOW_COLLAPSE_STEEPNESS = 0.3 # how sharp the collapse is

# -- Cognitive momentum --
MOMENTUM_AMPLIFY = 0.15       # how much acceleration amplifies transitions

# -- Intervention sensitivity --
INTERVENTION_MIDPOINT = 0.40  # fatigue level where break is 50% effective
INTERVENTION_STEEPNESS = 10.0 # how sharply effectiveness drops

# -- Cognitive capacity --
CAPACITY_BASE = 180.0         # base cognitive capacity (load units)

# -- Temporal cognitive drain (v4) --
DRAIN_MIDPOINT = 0.50         # session progress where drain becomes noticeable (50%)
DRAIN_STEEPNESS = 5.0         # how sharply drain activates
FLOW_EROSION_MAX = 0.35       # max flow retention loss at session end
FATIGUE_GRAVITY_MAX = 0.40    # max fatigue transition amplification
RECOVERY_RESISTANCE_MAX = 0.35 # max spontaneous recovery reduction

# Backward-compatible alias
RECOVERY_TAU_MINUTES = RECOVERY_TAU_SLOW


# =============================================================================
# Validation
# =============================================================================

def validate_params(alpha: float, beta: float, gamma: float, steps: int) -> Optional[str]:
    if not (math.isfinite(alpha) and math.isfinite(beta) and math.isfinite(gamma)):
        return "parameters must be finite numbers"
    if alpha < 0.3 or alpha > 3.0:
        return f"alpha must be in [0.3, 3.0], got {alpha}"
    if beta < 1 or beta > 5:
        return f"beta must be in [1, 5], got {beta}"
    if gamma < 0.5 or gamma > 2.0:
        return f"gamma must be in [0.5, 2.0], got {gamma}"
    if steps < 0 or steps > 144:
        return f"steps must be in [0, 144], got {steps}"
    return None


# =============================================================================
# Math Helpers
# =============================================================================

def sigmoid(x: float, center: float, steepness: float) -> float:
    """
    Logistic sigmoid: sigma(x) = 1 / (1 + e^(-k * (x - x0)))

    Guarded against overflow: exp(>709) = Infinity in IEEE 754.
    """
    z = -steepness * (x - center)
    if z > 700:
        return 1.0   # effectively 1/(1+0) = 1
    if z < -700:
        return 0.0   # effectively 1/(1+inf) = 0
    return 1.0 / (1.0 + math.exp(z))


def clamp(x: float) -> float:
    """Clamp to [0, 1] with NaN/Inf guard."""
    if not math.isfinite(x):
        return 0.0
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    if x < 1e-10:
        return 0.0
    return x


def biexponential_decay(t: float) -> float:
    """
    Biexponential decay: R(t) = w_fast * e^(-t/tau_fast) + w_slow * e^(-t/tau_slow)

    Returns fraction of fatigue remaining after t minutes of recovery.
    """
    return (RECOVERY_WEIGHT_FAST * math.exp(-t / RECOVERY_TAU_FAST) +
            RECOVERY_WEIGHT_SLOW * math.exp(-t / RECOVERY_TAU_SLOW))


def invert_biexponential_decay(ratio: float) -> float:
    """
    Invert the biexponential decay: find t such that decay(t) = ratio.

    Uses binary search (20 iterations) — more accurate than slow-only
    approximation, especially for short breaks where fast component matters.
    """
    lo, hi = 1.0, 120.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if biexponential_decay(mid) > ratio:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def validate_initial_state(state: Optional[List[float]]) -> np.ndarray:
    """Validate and normalize an initial state vector."""
    if state is None or not isinstance(state, list) or len(state) != 4:
        return INITIAL_STATE.copy()
    s = sum(state)
    if s <= 0 or not math.isfinite(s):
        return INITIAL_STATE.copy()
    arr = np.array([clamp(x / s) for x in state])
    return arr


# =============================================================================
# Attention Residue
# =============================================================================

RESIDUE_MAP = {
    'academic': {'sports': 0.22, 'arts': 0.14, 'other': 0.12, 'academic': 0.05},
    'sports':   {'academic': 0.18, 'arts': 0.12, 'other': 0.10, 'sports': 0.05},
    'arts':     {'academic': 0.14, 'sports': 0.10, 'other': 0.11, 'arts': 0.05},
    'other':    {'academic': 0.12, 'sports': 0.10, 'arts': 0.11, 'other': 0.05},
}


def compute_attention_residue(prev_type: Optional[str], new_type: Optional[str]) -> float:
    """
    Attention residue: cognitive carryover when switching between task types.

    Args:
        prev_type: Previous task type ('academic', 'sports', 'arts', 'other')
        new_type: New task type

    Returns:
        Residue factor [0, 1] — higher = more disruption
    """
    if not prev_type or not new_type:
        return 0.0
    if prev_type == new_type:
        return 0.05  # same domain = minimal switching cost
    return RESIDUE_MAP.get(prev_type, {}).get(new_type, 0.10)


def apply_attention_residue(state: np.ndarray, prev_type: Optional[str]) -> np.ndarray:
    """
    Apply attention residue to an initial state vector.
    Shifts flow -> distracted to model context-switching cost.
    """
    residue = compute_attention_residue(prev_type, 'other') if prev_type else 0.12
    flow_loss = state[0] * residue
    return np.array([
        clamp(state[0] - flow_loss),
        clamp(state[1] + flow_loss * 0.7),
        clamp(state[2] + flow_loss * 0.15),
        clamp(state[3] + flow_loss * 0.15),
    ])


# =============================================================================
# Cognitive Capacity
# =============================================================================

def compute_cognitive_capacity(alpha: float = 1.0) -> float:
    """
    Compute the cognitive capacity ceiling for a given alpha.
    Higher alpha -> higher capacity before diminishing returns kick in.
    """
    return CAPACITY_BASE * (0.5 + alpha * 0.5)


# =============================================================================
# Dynamic Transition Matrix Construction (v5)
# =============================================================================

def build_dynamic_matrix(
    alpha: float,
    beta: float,
    gamma: float,
    tick: int = 0,
    current_fatigue: float = 0.0,
    prev_fatigue: Optional[float] = None,
    flow_streak: int = 0,
    cumulative_load: float = 0.0,
    total_steps: int = 18,
) -> np.ndarray:
    """
    Build the 4x4 dynamic transition matrix (v5).

    Features:
      - Flow inertia: flow retention strengthens with consecutive flow ticks,
        then suddenly collapses after ~2 hours (tipping point).
      - Cognitive momentum: when fatigue is accelerating, off-diagonal
        transitions are amplified (slippery slope).
      - Capacity ceiling: when cumulative load exceeds cognitive capacity,
        all fatigue transitions are amplified.
      - Temporal cognitive drain: sigmoid curve over session progress that
        erodes flow retention, amplifies fatigue gravity, and weakens
        spontaneous recovery.
    """
    # Clamp inputs
    a = max(0.3, min(3.0, alpha)) if math.isfinite(alpha) else 1.0
    b = max(1, min(5, beta)) if math.isfinite(beta) else 3
    g = max(0.5, min(2.0, gamma)) if math.isfinite(gamma) else 1.0

    # -- Sigmoidal modifiers (v2) --
    alpha_flow_mod = sigmoid(a, 0.75, 4.0)        # [0.27, 0.95] at alpha=[0.5,1.5]
    alpha_recovery_mod = sigmoid(a, 0.80, 3.0)    # gentler recovery slope
    beta_fatigue_mod = sigmoid(b, 3.0, 1.2)       # [0.08, 0.92] at beta=[1,5]
    beta_distract_mod = sigmoid(b, 3.5, 1.5)      # harder to trigger distraction

    # -- State-dependent gamma (v2) --
    gamma_state_boost = 1.0 + current_fatigue * 0.6
    effective_gamma = 1.0 + (g - 1.0) * gamma_state_boost
    gamma_mod = clamp(effective_gamma)

    # -- Warmup factor (v2) --
    if tick < WARMUP_TICKS:
        warmup_factor = WARMUP_MIN + (1.0 - WARMUP_MIN) * (1.0 - math.exp(-tick / WARMUP_TAU))
    else:
        warmup_factor = 1.0

    # -- Flow inertia (v3) --
    flow_inertia = 1.0 + min(FLOW_INERTIA_MAX - 1.0, FLOW_INERTIA_BUILD * flow_streak)
    # Collapse risk: sigmoid that activates after ~2 hours
    if flow_streak > FLOW_COLLAPSE_THRESHOLD:
        collapse_risk = sigmoid(flow_streak - FLOW_COLLAPSE_THRESHOLD, 3, FLOW_COLLAPSE_STEEPNESS)
    else:
        collapse_risk = 0.0
    flow_anchor_mod = warmup_factor * (flow_inertia * (1.0 - collapse_risk * 0.7))

    # -- Cognitive momentum (v3) --
    if prev_fatigue is not None:
        fatigue_delta = max(0.0, current_fatigue - prev_fatigue)
    else:
        fatigue_delta = 0.0
    momentum_amplify = min(2.0, 1.0 + fatigue_delta * MOMENTUM_AMPLIFY * (1.0 / 0.05))

    # -- Capacity ceiling (v3) --
    capacity = compute_cognitive_capacity(a)
    if cumulative_load > capacity:
        capacity_factor = 1.0 + (cumulative_load - capacity) / capacity
    else:
        capacity_factor = 1.0

    # -- Temporal cognitive drain (v4) --
    session_progress = tick / total_steps if total_steps > 0 else 0.0
    drain = sigmoid(session_progress, DRAIN_MIDPOINT, DRAIN_STEEPNESS)

    flow_erosion_factor = 1.0 - drain * FLOW_EROSION_MAX
    fatigue_gravity_factor = 1.0 + drain * FATIGUE_GRAVITY_MAX
    recovery_resistance_factor = 1.0 - drain * RECOVERY_RESISTANCE_MAX

    # -- Build matrix --
    P = P_BASE.copy()

    # Row 0 — Flow
    P[0, 0] *= alpha_flow_mod * flow_anchor_mod * flow_erosion_factor
    P[0, 1] *= (1.0 + beta_distract_mod) * momentum_amplify * (1.0 + drain * 0.25)
    P[0, 2] *= (1.0 + beta_fatigue_mod) * gamma_mod * momentum_amplify * capacity_factor * fatigue_gravity_factor

    # Row 1 — Distracted
    P[1, 0] *= alpha_flow_mod * flow_erosion_factor
    P[1, 1] *= (2.0 - alpha_flow_mod) * (1.0 + drain * 0.15)
    P[1, 2] *= gamma_mod * momentum_amplify * capacity_factor * fatigue_gravity_factor

    # Row 2 — Fatigued
    P[2, 0] *= (0.3 + alpha_flow_mod * 0.2) * flow_erosion_factor
    P[2, 1] *= gamma_mod
    P[2, 2] *= gamma_mod * capacity_factor * fatigue_gravity_factor
    P[2, 3] *= alpha_recovery_mod * recovery_resistance_factor

    # Row 3 — Recovery
    P[3, 0] *= alpha_recovery_mod * recovery_resistance_factor
    P[3, 1] *= (1.5 - alpha_flow_mod * 0.5)
    P[3, 2] *= gamma_mod * fatigue_gravity_factor   # v5: micro-regression into fatigue
    P[3, 3] *= alpha_flow_mod

    # Normalise rows
    for i in range(4):
        row_sum = float(P[i].sum())
        if row_sum <= 0 or not math.isfinite(row_sum):
            P[i] = np.array([0.25, 0.25, 0.25, 0.25])
        elif abs(row_sum - 1.0) > 1e-12:
            P[i] /= row_sum

    return P


# =============================================================================
# Simulation
# =============================================================================

def make_tick(tick: int, v: np.ndarray) -> dict:
    """Build a timeline point dict from state vector."""
    total_mins = tick * 10
    h = total_mins // 60
    m = total_mins % 60
    return {
        "tick": tick,
        "timeLabel": f"{h}h{m:02d}",
        "flow": round(clamp(v[0]), 6),
        "distracted": round(clamp(v[1]), 6),
        "fatigue": round(clamp(v[2]), 6),
        "recovery": round(clamp(v[3]), 6),
    }


def simulate_trajectory(
    alpha: float,
    beta: float,
    gamma: float,
    steps: int,
    v0: np.ndarray,
    cumulative_load: float = 0.0,
    prev_type: Optional[str] = None,
) -> List[dict]:
    """Evolve state vector for `steps` ticks with full v5 dynamic matrix."""
    timeline = []
    v = v0.copy().astype(float)
    prev_fatigue_val = None
    flow_streak = 0
    total_steps = steps

    for t in range(steps + 1):
        timeline.append(make_tick(t, v))

        current_fatigue = float(v[2])

        # Track flow streak (for flow inertia)
        if v[0] > 0.3:
            flow_streak += 1
        else:
            flow_streak = max(0, flow_streak - 2)

        if t >= steps:
            break

        # Build full v5 dynamic matrix
        load = cumulative_load + t * (beta / 5.0) * gamma
        P = build_dynamic_matrix(
            alpha, beta, gamma, t, current_fatigue,
            prev_fatigue_val, flow_streak, load, total_steps
        )

        # v(t+1) = v(t) @ P
        next_v = v @ P
        s = float(next_v.sum())
        if s > 0 and math.isfinite(s):
            v = next_v / s
        else:
            v = np.array([0.25, 0.25, 0.25, 0.25])

        prev_fatigue_val = current_fatigue

    return timeline


def simulate_trajectory_from(
    alpha: float,
    beta: float,
    gamma: float,
    v0: np.ndarray,
    steps: int,
    start_tick: int,
    cumulative_load: float = 0.0,
    prev_type: Optional[str] = None,
) -> List[dict]:
    """
    Continue simulation from a post-break state.

    v5 feature: post-break drain reset — a break partially restores cognitive
    resources. The temporal drain is reduced proportional to fatigue recovery.
    """
    timeline = []
    v = v0.copy().astype(float)
    prev_fatigue_val = None
    flow_streak = 0
    total_steps = start_tick + steps

    # Post-break drain reset
    post_break_fatigue = float(v0[2])
    drain_retention = min(1.0, post_break_fatigue / 0.50)

    for t in range(steps + 1):
        timeline.append(make_tick(start_tick + t, v))

        current_fatigue = float(v[2])

        if v[0] > 0.3:
            flow_streak += 1
        else:
            flow_streak = max(0, flow_streak - 2)

        if t >= steps:
            break

        # Drain-effective tick: reduced by drain_retention after a break
        drain_tick = start_tick * drain_retention + t
        load = cumulative_load + (start_tick + t) * (beta / 5.0) * gamma
        P = build_dynamic_matrix(
            alpha, beta, gamma, drain_tick, current_fatigue,
            prev_fatigue_val, flow_streak, load, total_steps
        )

        next_v = v @ P
        s = float(next_v.sum())
        if s > 0 and math.isfinite(s):
            v = next_v / s
        else:
            v = np.array([0.25, 0.25, 0.25, 0.25])

        prev_fatigue_val = current_fatigue

    return timeline


def find_burnout_tick(timeline: List[dict], threshold: float = 0.50) -> int:
    """Return tick where P(Fatigue) first exceeds threshold, or -1."""
    for i, point in enumerate(timeline):
        if point["fatigue"] > threshold:
            return i
    return -1


# =============================================================================
# Recovery & Break Optimization (v5 — Biexponential)
# =============================================================================

def compute_recovery_state(
    current_state: np.ndarray, break_minutes: float = 15.0
) -> np.ndarray:
    """
    Compute post-break cognitive state using biexponential recovery.

    Recovery has two phases:
      Fast (tau=2min): acute sympathetic recovery — quick but shallow
      Slow (tau=120min): deep parasympathetic recovery — gradual but thorough

    Intervention sensitivity: a break at 30% fatigue recovers ~90% of possible
    flow. The same break at 70% fatigue only recovers ~40%.
    """
    flow, distracted, fatigue, recovery = [float(x) for x in current_state]

    # Biexponential decay of fatigue
    decay_fast = math.exp(-break_minutes / RECOVERY_TAU_FAST)
    decay_slow = math.exp(-break_minutes / RECOVERY_TAU_SLOW)
    total_decay = RECOVERY_WEIGHT_FAST * decay_fast + RECOVERY_WEIGHT_SLOW * decay_slow

    new_fatigue = fatigue * total_decay
    fatigue_reduced = fatigue - new_fatigue

    # Intervention sensitivity
    sensitivity = 1.0 - sigmoid(fatigue, INTERVENTION_MIDPOINT, INTERVENTION_STEEPNESS)

    # Conversion efficiency
    base_efficiency = 0.7 if flow > 0.3 else 0.4
    conversion_efficiency = base_efficiency * max(0.2, sensitivity)

    to_flow = fatigue_reduced * conversion_efficiency
    to_recovery = fatigue_reduced * (1.0 - conversion_efficiency)

    new_flow = flow + to_flow
    new_distracted = distracted * total_decay
    distracted_reduced = distracted - new_distracted
    new_flow += distracted_reduced * 0.5
    new_recovery_val = recovery + to_recovery + distracted_reduced * 0.5

    result = np.array([
        clamp(new_flow),
        clamp(new_distracted),
        clamp(new_fatigue),
        clamp(new_recovery_val),
    ])

    s = float(result.sum())
    if s > 0 and math.isfinite(s):
        result /= s

    return result


def compute_optimal_break_duration(
    timeline: List[dict],
    burnout_tick: int,
    target_fatigue: float = 0.30,
) -> int:
    """
    Compute optimal break duration using full biexponential recovery model.

    Inverts the actual recovery curve R(t) = w_fast * e^(-t/tau_fast) +
    w_slow * e^(-t/tau_slow) rather than using only the slow component.
    This gives more accurate results for short breaks (5-15 min) where
    the fast sympathetic component contributes significantly.

    Then scaled by intervention sensitivity:
      effectiveness = 1 - sigma(fatigue - 0.40, 10)
      t_adjusted = t_raw / effectiveness

    Returns:
        Break duration in minutes (rounded to nearest 5)
    """
    if not timeline or len(timeline) == 0:
        return 15
    if burnout_tick <= 0 or burnout_tick >= len(timeline):
        return 15

    state = timeline[burnout_tick]
    current_fatigue = state["fatigue"]
    current_flow = state["flow"]

    if current_fatigue <= target_fatigue:
        return 5

    ratio = target_fatigue / current_fatigue
    if ratio <= 0 or ratio >= 1:
        return 15

    # Invert the full biexponential decay
    raw_minutes = invert_biexponential_decay(ratio)

    # Intervention sensitivity: breaks are less effective at higher fatigue
    effectiveness = 1.0 - sigmoid(current_fatigue, INTERVENTION_MIDPOINT, INTERVENTION_STEEPNESS)
    effective_factor = max(0.25, effectiveness)

    # Recovery capacity depends on flow
    recovery_capacity = current_flow * 0.8 + 0.2

    adjusted_minutes = raw_minutes / (effective_factor * recovery_capacity)

    return max(5, min(60, round(adjusted_minutes / 5.0) * 5))


# =============================================================================
# Break Optimization (full simulate + break + continue)
# =============================================================================

def optimize_with_break(
    alpha: float,
    beta: float,
    gamma: float,
    steps: int = 18,
    burnout_tick: Optional[int] = None,
    break_minutes: float = 15.0,
    cumulative_load: float = 0.0,
    prev_type: Optional[str] = None,
) -> dict:
    """
    Simulate with a break inserted before burnout.

    Returns:
        {"original": [...], "optimized": [...]}
    """
    v0 = INITIAL_STATE.copy()
    original = simulate_trajectory(alpha, beta, gamma, steps, v0, cumulative_load, prev_type)

    if not burnout_tick or burnout_tick <= 0:
        return {"original": original, "optimized": original}

    break_insert_tick = max(0, burnout_tick - 1)

    # Pre-break simulation
    pre_break = simulate_trajectory(alpha, beta, gamma, break_insert_tick, v0.copy(), cumulative_load, prev_type)

    # Get state at break point
    pre_break_state = np.array([
        pre_break[-1]["flow"],
        pre_break[-1]["distracted"],
        pre_break[-1]["fatigue"],
        pre_break[-1]["recovery"],
    ])

    # Compute recovery
    post_break_v = compute_recovery_state(pre_break_state, break_minutes)

    remaining_steps = steps - break_insert_tick - 1
    if remaining_steps <= 0:
        recovery_tick = make_tick(break_insert_tick + 1, post_break_v)
        return {"original": original, "optimized": pre_break + [recovery_tick]}

    # Post-break simulation (with drain reset)
    post_break = simulate_trajectory_from(
        alpha, beta, gamma, post_break_v, remaining_steps,
        break_insert_tick + 1, cumulative_load, prev_type
    )

    return {"original": original, "optimized": pre_break + post_break}


# =============================================================================
# API Routes
# =============================================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.5.0"}


@app.post("/api/simulate")
def simulate_endpoint(
    alpha: float = 1.0,
    beta: float = 3.0,
    gamma: float = 1.0,
    steps: int = 18,
    initial_state: Optional[str] = None,  # JSON array: "[0.8, 0.1, 0.05, 0.05]"
    cumulative_load: float = 0.0,
    prev_task_type: Optional[str] = None,
    burnout_threshold: float = 0.50,
    break_minutes: float = 15.0,
    target_fatigue: float = 0.30,
):
    """
    Run a full v5 non-homogeneous Markov-chain simulation.

    Query params:
      - alpha: Cognitive baseline (0.3–3.0, default 1.0)
      - beta: Task difficulty (1–5, default 3)
      - gamma: Circadian coefficient (0.5–2.0, default 1.0)
      - steps: 10-minute ticks (0–144, default 18 = 3 hours)
      - initial_state: JSON array [flow, distracted, fatigue, recovery]
      - cumulative_load: Total cognitive load so far today
      - prev_task_type: Previous task type for attention residue
      - burnout_threshold: Fatigue threshold for break detection (default 0.50)
      - break_minutes: Break duration for optimization
      - target_fatigue: Target fatigue for optimal break computation

    Returns:
      - timeline: probability vectors at each 10-min tick
      - burnout_tick: tick where fatigue exceeds threshold (or -1)
      - optimal_break_minutes: computed optimal break duration
      - optimized_timeline: timeline with break inserted at burnout
      - matrix: final transition matrix used
      - params: input parameters
    """
    import json

    err = validate_params(alpha, beta, gamma, steps)
    if err:
        raise HTTPException(status_code=422, detail=err)

    # Parse initial state
    v0 = INITIAL_STATE.copy()
    if initial_state:
        try:
            parsed = json.loads(initial_state)
            v0 = validate_initial_state(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    # Apply attention residue if switching task types
    if prev_task_type and initial_state is None:
        v0 = apply_attention_residue(v0, prev_task_type)

    # Determine if this is a resume from degraded state
    actual_prev_type = prev_task_type if initial_state is None else None

    # Run simulation
    timeline = simulate_trajectory(alpha, beta, gamma, steps, v0, cumulative_load, actual_prev_type)

    # Find burnout
    burnout_tick = find_burnout_tick(timeline, burnout_threshold)

    # Compute optimal break
    optimal_break_minutes = None
    if burnout_tick > 0:
        optimal_break_minutes = compute_optimal_break_duration(
            timeline, burnout_tick, target_fatigue
        )

    # Optimize with break
    optimized = optimize_with_break(
        alpha, beta, gamma, steps, burnout_tick, break_minutes,
        cumulative_load, actual_prev_type
    )

    # Build final matrix at last tick for inspection
    last_tick = steps
    final_v = np.array([
        timeline[-1]["flow"], timeline[-1]["distracted"],
        timeline[-1]["fatigue"], timeline[-1]["recovery"]
    ])
    load = cumulative_load + steps * (beta / 5.0) * gamma
    final_matrix = build_dynamic_matrix(
        alpha, beta, gamma, last_tick, float(final_v[2]),
        None, 0, load, steps
    )

    return {
        "timeline": timeline,
        "burnout_tick": burnout_tick,
        "optimal_break_minutes": optimal_break_minutes,
        "optimized_timeline": optimized["optimized"],
        "matrix": final_matrix.tolist(),
        "params": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "steps": steps,
            "cumulative_load": cumulative_load,
            "prev_task_type": prev_task_type,
        },
    }


@app.post("/api/recovery")
def recovery_endpoint(
    flow: float = 0.3,
    distracted: float = 0.2,
    fatigue: float = 0.4,
    recovery: float = 0.1,
    break_minutes: float = 15.0,
):
    """
    Compute post-break cognitive state using biexponential recovery.

    Query params: current state vector + break duration in minutes.
    Returns: new state vector after recovery.
    """
    current = np.array([flow, distracted, fatigue, recovery])
    # Normalize input
    s = float(current.sum())
    if s > 0 and math.isfinite(s):
        current /= s

    new_state = compute_recovery_state(current, break_minutes)

    return {
        "before": {
            "flow": round(float(current[0]), 4),
            "distracted": round(float(current[1]), 4),
            "fatigue": round(float(current[2]), 4),
            "recovery": round(float(current[3]), 4),
        },
        "after": {
            "flow": round(float(new_state[0]), 4),
            "distracted": round(float(new_state[1]), 4),
            "fatigue": round(float(new_state[2]), 4),
            "recovery": round(float(new_state[3]), 4),
        },
        "break_minutes": break_minutes,
    }


@app.post("/api/attention-residue")
def attention_residue_endpoint(
    prev_type: str = Query(...),
    new_type: str = Query(...),
):
    """
    Compute attention residue between two task types.
    """
    residue = compute_attention_residue(prev_type, new_type)
    return {
        "prev_type": prev_type,
        "new_type": new_type,
        "residue": residue,
    }


@app.post("/api/cognitive-capacity")
def cognitive_capacity_endpoint(alpha: float = 1.0):
    """
    Compute cognitive capacity for a given alpha.
    """
    capacity = compute_cognitive_capacity(alpha)
    return {
        "alpha": alpha,
        "capacity": capacity,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
