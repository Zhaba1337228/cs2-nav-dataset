# Labeling Guide

This guide explains how raw keyboard/mouse input is converted into action labels for imitation learning.

## Overview

The labeling stage (`label-samples`) takes aligned per-frame samples and assigns categorical action labels based on configurable thresholds applied to key states and mouse movement.

## Action Labels

### 1. action_move — Movement Direction

Derived from WASD key states:

| Keys Pressed | Label |
|--------------|-------|
| (none) | `stop` |
| W | `forward` |
| W + A | `forward_left` |
| W + D | `forward_right` |
| A | `left` |
| D | `right` |
| S | `back` |
| S + A | `back_left` |
| S + D | `back_right` |

### 2. action_turn — Mouse Turn Direction & Magnitude

Derived from aggregated mouse delta (dx, dy) over the frame interval:

| Condition | Label |
|-----------|-------|
| abs(dx) + abs(dy) < idle_mouse_epsilon | `no_turn` |
| dx < -mouse_large_threshold | `turn_left_large` |
| dx < -mouse_medium_threshold | `turn_left_medium` |
| dx < -mouse_small_threshold | `turn_left_small` |
| dx > mouse_large_threshold | `turn_right_large` |
| dx > mouse_medium_threshold | `turn_right_medium` |
| dx > mouse_small_threshold | `turn_right_small` |
| dy < -mouse_medium_threshold | `look_up` |
| dy > mouse_medium_threshold | `look_down` |

Horizontal movement takes priority over vertical when abs(dx) >= abs(dy).

### 3. action_jump — Jump

Binary (0/1) from `key_space` state.

### 4. action_crouch — Crouch

Binary (0/1) from `key_ctrl` state.

### 5. action_fire — Fire

Binary (0/1) from `mouse_left` button state.

### 6. action_macro — Composite Action

Built by combining move + turn + jump/crouch/fire:

| Pattern | Example Label |
|---------|---------------|
| No input | `idle` |
| W only | `move_forward` |
| W + D | `move_forward_right` |
| W + turn_right_small | `move_forward_turn_right_small` |
| W + jump | `move_forward_jump` |
| W + crouch | `move_forward_crouch` |
| W + fire | `move_forward_fire` |
| S + any turn | `back_off` |
| Moving keys + no mouse movement (sustained) | `unstuck_candidate` |

## Configurable Thresholds

All thresholds are in `config.py` under `LabelingConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mouse_small_threshold` | 5.0 | Minimum mouse delta for "small" turn |
| `mouse_medium_threshold` | 30.0 | Minimum mouse delta for "medium" turn |
| `mouse_large_threshold` | 80.0 | Minimum mouse delta for "large" turn |
| `min_action_duration_ms` | 50.0 | Minimum duration for intentional action |
| `unstuck_detection_window_ms` | 2000.0 | Window for stuck detection |
| `idle_mouse_epsilon` | 3.0 | Below this = mouse is idle |

## Adjusting Thresholds

### If turns are classified too aggressively (too many "large" turns):
- Increase `mouse_large_threshold` and `mouse_medium_threshold`

### If turns are classified too conservatively (too many "no_turn"):
- Decrease `mouse_small_threshold`

### If the model confuses idle with small turns:
- Increase `idle_mouse_epsilon`

### If unstuck detection fires too often:
- Increase `idle_mouse_epsilon` or decrease `unstuck_detection_window_ms`

## Label Distribution

After labeling, check the distribution:

```bash
python main.py label-samples
```

Look for:
- **Balanced move labels** — forward should be most common, back less so
- **Reasonable turn distribution** — no_turn should dominate, small turns common
- **Low but non-zero jump/crouch/fire** — these should be sparse

If a label has zero samples, you may need to record more diverse data.
