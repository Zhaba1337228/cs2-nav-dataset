# QA Checklist

Use this checklist to verify the quality of recorded sessions and the final dataset.

## Session Quality (After Recording)

Run: `python main.py inspect-session --session session_XXXX`

- [ ] **Frame count > 0** — Session has captured frames
- [ ] **Event count > 0** — Session has recorded input events
- [ ] **Frame count matches expected** — At ~15 FPS, a 1-minute session should have ~900 frames
- [ ] **No dropped frames** — Check alignment report for dropped_frames = 0 (or very low)
- [ ] **Session metadata is complete** — map_name and scenario_type are set correctly
- [ ] **events.csv is not empty** — File exists and has content
- [ ] **session.json is valid** — Can be parsed as JSON

## Sample Quality (After build-samples)

Run: `python main.py build-samples`

- [ ] **Samples were created** — Output shows non-zero sample count
- [ ] **Sample count ≈ frame count** — Should be close to total frames across sessions
- [ ] **samples.parquet exists** — File was created in dataset/processed/
- [ ] **samples.csv exists** — CSV version was also created
- [ ] **Alignment report looks reasonable** — Check dataset/debug/alignment_report.json

## Label Quality (After label-samples)

Run: `python main.py label-samples`

- [ ] **All label columns are populated** — No NaN or empty values
- [ ] **action_move distribution is reasonable**:
  - `forward` should be the most common
  - `stop` should appear (when standing still)
  - `back` should be less common than `forward`
- [ ] **action_turn distribution is reasonable**:
  - `no_turn` should dominate (when walking straight)
  - Small/medium turns should be present
- [ ] **action_jump/crouch/fire are sparse but non-zero**
- [ ] **action_macro has variety** — Not just one or two labels
- [ ] **Label stats saved** — Check dataset/debug/label_stats.json

## Dataset Quality (After export-dataset)

Run: `python main.py export-dataset`

- [ ] **Train/val split exists** — Both train.parquet and val.parquet created
- [ ] **Val split is ~15%** — Check splits.json for val_samples / total_samples ratio
- [ ] **Manifests created** — train_manifest.jsonl and val_manifest.jsonl exist
- [ ] **Manifests are valid JSONL** — Each line is valid JSON
- [ ] **Dataset schema saved** — dataset_schema.json exists
- [ ] **No data leakage** — Sessions are split, not individual frames (same session = same split)

## Training Readiness

Run: `python training/train_stub.py`

- [ ] **Dataset loads without errors** — No file-not-found or parse errors
- [ ] **Batch shapes are correct** — Images: (B, history_len, C, H, W)
- [ ] **Target labels are populated** — All action keys present
- [ ] **No NaN in targets** — All labels are valid integers/floats
- [ ] **Multiple batches work** — DataLoader iterates without crashing

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| No frames recorded | CS2 not running or wrong window title | Check window title in config |
| No events recorded | Input hooks failed | Run as administrator on Windows |
| High dropped frame rate | System too slow | Lower FPS in config |
| All labels are "stop" | No keys pressed during recording | Make sure you're actively playing |
| All turns are "no_turn" | Mouse sensitivity too low | Decrease mouse thresholds |
| Dataset won't load | Missing files | Re-run the full pipeline |

## Minimum Dataset Size

For initial training experiments:
- **Minimum**: 10 sessions, ~30 minutes total, ~15,000 samples
- **Recommended**: 30+ sessions, ~2 hours total, ~100,000+ samples
- **Good**: 100+ sessions, ~10 hours total, ~500,000+ samples
