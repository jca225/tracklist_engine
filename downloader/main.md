"""
TODO:
1. Download song
2. Get acapella, instrumental
3. Get key, bpm
4. Get measure data
5. Persist everything properly


Structure for hard drive:

artists/
└── Daft Punk/
    └── Discovery/
        └── 2001-01-12 - One More Time/
            ├── original/
            ├── stems/
            ├── derived/
            └── meta/

            
original/
    ├── source.wav
    ├── source.flac
    └── checksum.sha256


stems/
└── demucs/
    └── htdemucs_v4.1/
        ├── params.json
        ├── vocals.wav
        ├── instrumental.wav
        ├── drums.wav
        ├── bass.wav
        └── other.wav

"""
"""

Stem separation:

pip install demucs

demucs \
  -n htdemucs \
  --device mps \
  --two-stems vocals \
  --segment 7 \
  --overlap 0.25 \
  your_song.wav

change ouput with -o ./stems

from demucs.apply import apply_model
from demucs.pretrained import get_model
import torch

model = get_model("htdemucs")
model.to("mps")

# then apply_model(...) on waveforms


"""


"""
cue

git clone https://github.com/<cue-detr-repo>.git
cd cue-detr
pip install -r requirements.txt


Idea

Step 1 — Run Cue-DETR on the DJ set
dj_cues = cue_detr.predict(dj_set_audio)
Output:
[
  { "time": 312.4, "type": "drop", "confidence": 0.93 },
  { "time": 354.8, "type": "phrase_end", "confidence": 0.88 }
]
Step 2 — Segment the DJ set
segments = split_audio_by_cues(
    dj_audio,
    cue_types=["phrase_start", "drop"]
)
"""


"""
In the CFG how will we measure things in the global set like global
measure? Global bpm? Do we even need to measure these?
Note serato does not determine global measures that well...
"""


"""
Embeddings:
2. Should you embed a cappellas and instrumentals?
Yes. Absolutely. Separately.
This is not optional if you care about DJ sets.


"""