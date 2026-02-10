from __future__ import annotations

from pathlib import Path

SUPPORTED_PLATFORMS = ("spotify", "youtube", "soundcloud")
USER_AGENT = "Mozilla/5.0"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}

KEY_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KRUMHANSL_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_MINOR = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
CAMELOT_MAJOR = {
    "C": "8B",
    "C#": "3B",
    "D": "10B",
    "D#": "5B",
    "E": "12B",
    "F": "7B",
    "F#": "2B",
    "G": "9B",
    "G#": "4B",
    "A": "11B",
    "A#": "6B",
    "B": "1B",
}
CAMELOT_MINOR = {
    "C": "5A",
    "C#": "12A",
    "D": "7A",
    "D#": "2A",
    "E": "9A",
    "F": "4A",
    "F#": "11A",
    "G": "6A",
    "G#": "1A",
    "A": "8A",
    "A#": "3A",
    "B": "10A",
}

# Set your mounted hard-drive root here (example: Path("/Volumes/MySSD")).
HARD_DRIVE_MOUNT = Path("/Volumes/tracklist_drive")
HARD_DRIVE_OUTPUT_ROOT = HARD_DRIVE_MOUNT / "audio_corpus"

# Global audio spec for the corpus.
TARGET_AUDIO_FORMAT = "wav"
TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 2
TARGET_BIT_DEPTH = "32-bit float"
TARGET_PCM_CODEC = "pcm_f32le"

# Corpus folder convention.
OBJECTS_DIRNAME = "objects"
TRACKS_DIRNAME = "tracks"
SOURCE_DIRNAME = "source"
SOURCE_AUDIO_FILENAME = "audio.wav"
SOURCE_AUDIO_META_FILENAME = "audio.json"
STEMS_DIRNAME = "stems"
