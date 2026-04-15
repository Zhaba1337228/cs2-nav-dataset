# Data Collection Guide

This guide explains how to record high-quality gameplay sessions for the CS2 navigation imitation learning dataset.

## Goal

Record diverse, clean navigation data that teaches a model how to move through the game world based on visual input alone.

## Setup

1. Launch CS2 in **windowed** or **borderless windowed** mode
2. Set resolution to at least 1280x720 (the recorder will resize to 640x480)
3. Disable any overlay software that might interfere with screen capture
4. Close unnecessary applications to maximize performance

## Recording a Session

```bash
python main.py record --map de_dust2 --scenario navigation
```

### Hotkeys During Recording

| Key | Action |
|-----|--------|
| F6 | Stop recording |
| F7 | Pause / Resume |
| F8 | Cycle scenario type |
| F9 | Start new session |

## Scenario Types

Use these scenario types to categorize your recordings:

| Scenario | Description | Example |
|----------|-------------|---------|
| `navigation` | General movement through the map | Walking around de_dust2 |
| `navigation_corridor` | Moving through narrow corridors/hallways | Tunnels on de_inferno |
| `navigation_doorway` | Passing through doorways and tight openings | Doors on de_mirage |
| `obstacle_avoidance` | Navigating around obstacles | Boxes, barrels, cars |
| `unstuck` | Recovery from being stuck | Wiggling out of corners |
| `open_area` | Movement in open spaces | Mid on de_dust2 |

## Session Duration Guidelines

| Scenario Type | Recommended Duration |
|---------------|---------------------|
| navigation | 2-5 minutes |
| navigation_corridor | 30-90 seconds |
| navigation_doorway | 30-60 seconds |
| obstacle_avoidance | 30-60 seconds |
| unstuck | 10-30 seconds |
| open_area | 1-3 minutes |

## What to Record

### Good Data

- **Smooth, intentional movement** — walk as if you're navigating purposefully
- **Diverse paths** — take different routes through the same area
- **Varied speeds** — mix walking, running (shift), and crouching
- **Recovery actions** — when you get stuck, record how you free yourself
- **Different maps** — record on multiple maps for generalization
- **Different times of day** — if maps have lighting variations

### Bad Data (Avoid)

- **AFK / standing still** for long periods (some idle is fine)
- **Spinning in circles** without purpose
- **Menu screens / loading screens** — always start recording in-game
- **Chat typing** — avoid opening chat during recording
- **Buy menu** — avoid recording while in buy menu
- **Spectating** — only record first-person gameplay
- **Extreme sensitivity** — if your mouse sensitivity is so high that small movements cause huge turns, the model will struggle

## Recording Strategy

### Phase 1: Basic Navigation (50% of data)
- Walk through corridors and hallways
- Pass through doorways
- Navigate open areas
- Use W + A/D for turning while moving

### Phase 2: Complex Navigation (30% of data)
- Navigate around obstacles
- Combine movement with crouching (Ctrl)
- Use shift-walk for precision
- Practice recovery from stuck positions

### Phase 3: Edge Cases (20% of data)
- Intentionally get stuck and recover
- Navigate tight spaces
- Quick direction changes
- Jump over small obstacles (Space)

## Tips

1. **Be consistent** — use similar movement patterns each session
2. **Don't rush** — smooth, deliberate movements produce better labels
3. **Vary your paths** — don't always take the same route
4. **Record errors** — getting stuck and recovering is valuable data
5. **Check your sessions** — use `inspect-session` to verify quality
6. **Aim for 30+ minutes** of total recording across all sessions
7. **Multiple maps** — at least 2-3 different maps recommended

## After Recording

1. Inspect each session: `python main.py inspect-session --session session_0001`
2. Check for dropped frames in the alignment report
3. Verify event counts are reasonable (should be hundreds or thousands)
4. If a session looks bad, delete its directory and re-record
