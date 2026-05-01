"""HTML row tokenizer for 1001tracklists DJ-set rows.

Pipeline stage between scraping (web_crawler/, which writes raw HTML to
dj_set_rows.raw_html) and any downstream that needs structured track data
(track_metadata builder, audio downloader, generator).

Dispatch entry point is `tokenize_row(raw_html)` / `classify_row(raw_html)`
in `tokenizer.py`. The submodules parse the row families:
- track_tokenizer:      `div.tlpItem` — main track rows (TrackRow)
- suggestion_tokenizer: `div.sugTog`  — user-suggested track IDs (SuggestionRow)
- text_tokenizer:       `div.bItmH`   — headers / notices / warnings / recycle links
- id_tokenizer:         alternate lens on `div.tlpItem` exposing linked-tracklist
                        hints for unidentified tracks (IDTrack)

Package name is `tokenizer` (singular) — distinct from HuggingFace
`tokenizers` (plural), so no import-time collision.
"""
from .tokenizer import tokenize_row, classify_row

__all__ = ["tokenize_row", "classify_row"]
