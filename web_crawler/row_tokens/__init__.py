"""HTML row tokenizers for 1001tracklists DJ set rows.

Named `row_tokens` (not `tokenizers`) to avoid colliding with the HuggingFace
`tokenizers` package on any shared Python environment.

Dispatch entry point is `tokenize_row(raw_html)` / `classify_row(raw_html)`
in `tokenizer.py`. The submodules parse the three row families:
- track_tokenizer: `div.tlpItem` — main track rows
- suggestion_tokenizer: `div.sugTog` — user-suggested track IDs
- text_tokenizer: `div.bItmH` — headers / notices / warnings / recycle links
- id_tokenizer: parses the `ID` track half-rows (unidentified tracks)
"""
from .tokenizer import tokenize_row, classify_row

__all__ = ["tokenize_row", "classify_row"]
