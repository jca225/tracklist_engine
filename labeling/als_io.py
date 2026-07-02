"""DEPRECATED shim — the `.als` codec now lives in `labeling.als`.

Import from `labeling.als` (or its submodules `models` / `read` / `identity` /
`tags` / `write`) instead. This shim keeps legacy `labeling.als_io` imports
working during the transition and will be dropped once no consumer references
it (see docs/als_codec_subpackage_plan.md).
"""

from __future__ import annotations

from labeling.als import *  # noqa: F401,F403
from labeling.als.identity import (  # noqa: F401
    _SLOT_FROM_PATH,
    _filename_stem_marker,
    _normalize_path,
    _stem_folder_name,
)
from labeling.als.semantics import (  # noqa: F401
    _find_mix_splice_beat,
    _split_monotonic_arr_interval,
)
from labeling.als.tags import _BRACKET_TAG, _USER_TAG_PATTERN  # noqa: F401
from labeling.als.write import _ENV_INIT_TIME  # noqa: F401
