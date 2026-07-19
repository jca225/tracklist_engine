# Phase 2 — Live-mediated audio round-trip (Mac)

Offline denotation (`python -m labeling.audio_roundtrip`) is the **hard gate**.
This doc is the gold-standard follow-up: both sides rendered by Ableton Live.

## Procedure

1. Open the **hand** labeling `.als` in Live. Export Audio (arrangement, freeze
   as you normally would for a reference bounce) → `hand_export.wav`.
2. Build a re-seeded session from the committed GT (or open a session that only
   contains the exported GT clips). Export Audio the same way → `gt_export.wav`.
3. Compare:

```bash
venvs/audio/bin/python - <<'PY'
from pathlib import Path
import numpy as np
import soundfile as sf
from labeling.audio_roundtrip import assert_audio_equivalent
from labeling.als.render_offline import DEFAULT_SR
import librosa

def load(p):
    y, sr = sf.read(p, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != DEFAULT_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=DEFAULT_SR)
    return y

a = load("hand_export.wav")
b = load("gt_export.wav")
r = assert_audio_equivalent(a, b)
print(("PASS: " if r.ok else "FAIL: ") + r.detail)
raise SystemExit(0 if r.ok else 1)
PY
```

## Notes

- Match export settings (sample rate, normalize, include return tracks) on both
  bounces or the assert will fail for DSP reasons unrelated to GT shaping.
- Automation / AppleScript for headless Export Audio is future work; until then
  this is operator-run for BB11/BB12 certification, not CI.
