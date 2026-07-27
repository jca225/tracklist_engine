.PHONY: check check-rust check-python migrate-dry-run vertical-lf cotrain-round features-demo \
	canary-smoke collide land-budget land-verify ready help

VENV := python/.venv
PIP := $(VENV)/bin/pip
ART_ROOT := staging/artifacts
PROV_DB := staging/provenance.db

help:
	@echo "Quality"
	@echo "  make check            — rustc/clippy/test + ruff/pyright/pytest"
	@echo "  make migrate-dry-run  — BB11/BB12 transfer dry-run (no byte copy)"
	@echo "  make vertical-lf      — LFs + independent canary → Rust pws-fit"
	@echo "  make cotrain-round    — independent canary → promote/reject + pseudo-labels"
	@echo "  make canary-smoke     — pytest + cotrain-round promote check"
	@echo "  make features-demo    — put_audio → stub mert → FeatureBlob (+ LF ref)"
	@echo ""
	@echo "Branch hygiene (requires git + braid from ~/workspace/braid)"
	@echo "  make ready            — is THIS branch landable? (exit 0/1)"
	@echo "  make collide          — collisions, hotspots, landing order"
	@echo "  make land-budget      — branches past the ugly-landing commit budget"
	@echo "  make land-verify      — after merge/rebase: parse check + cargo check"

check: check-rust check-python

check-rust:
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings
	cargo test --workspace

$(VENV)/bin/python:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e "python[dev]"

check-python: $(VENV)/bin/python
	cd python && ../$(VENV)/bin/ruff check .
	cd python && ../$(VENV)/bin/ruff format --check .
	cd python && ../$(VENV)/bin/pyright
	cd python && ../$(VENV)/bin/pytest -q

migrate-dry-run:
	cargo run -q -p dj_migrate -- inventory --fixtures-only --fixtures fixtures/gold --out staging/inventory.json
	cargo run -q -p dj_migrate -- map --inventory staging/inventory.json --out staging/map_plan.json
	cargo run -q -p dj_migrate -- stage --map staging/map_plan.json --staging-dir staging --dry-run
	cargo run -q -p dj_migrate -- verify-manifest --manifest staging/staging_manifest.json
	@echo "dry-run complete; apply is gated behind --i-know-this-copies-bytes"

vertical-lf: $(VENV)/bin/python
	$(VENV)/bin/python -c "from pathlib import Path; from sensors.lfs import write_vertical_slice; print(write_vertical_slice(Path('fixtures/gold'), Path('staging')))"
	cargo run -q -p dj_migrate -- pws-fit --bundle staging/vertical_lf_bundle.json --out staging/vertical_lf_report.json

# Round 0: independent canary (title heuristic vs canary_stem_gold) → promote/reject.
cotrain-round: $(VENV)/bin/python
	$(VENV)/bin/python -c "from pathlib import Path; from sensors.lfs import write_vertical_slice; print(write_vertical_slice(Path('fixtures/gold'), Path('staging')))"
	cargo run -q -p dj_migrate -- run-round \
		--bundle staging/vertical_lf_bundle.json \
		--out staging/cotrain_round_report.json \
		--out-pseudo staging/cotrain_pseudo_labels.json

canary-smoke: $(VENV)/bin/python
	cd python && ../$(VENV)/bin/pytest -q tests/test_claimed_stem_lf.py
	$(MAKE) cotrain-round
	@python3 -c "import json; r=json.load(open('staging/cotrain_round_report.json')); assert r['round']['status']=='promoted' and r['gate']['passed']; print('canary-smoke ok:', r['gate']['detail'])"

# put_audio → analyze.mert (stub) → FeatureBlob registration; no live DB.
# Uses a tiny synthetic wav under staging/; embeddings are deterministic stubs.
features-demo: $(VENV)/bin/python
	@mkdir -p staging/mert
	@printf 'RIFF....WAVEfmt features-demo-audio' > staging/demo.wav
	cargo run -q -p dj_migrate -- audio-put \
		--file staging/demo.wav \
		--artifact-root $(ART_ROOT) \
		--provenance-db $(PROV_DB) \
		--out staging/audio_put.json
	@AUDIO_ID=$$(python3 -c "import json; print(json.load(open('staging/audio_put.json'))['artifact_id'])"); \
	$(VENV)/bin/python -m sensors.mert \
		--audio staging/demo.wav \
		--out-dir staging/mert \
		--audio-artifact-id $$AUDIO_ID; \
	MERT_JSON=$$(ls staging/mert/*_mert.json | head -1); \
	cargo run -q -p dj_migrate -- feature-register \
		--parent-audio-id $$AUDIO_ID \
		--mert-json $$MERT_JSON \
		--artifact-root $(ART_ROOT) \
		--provenance-db $(PROV_DB) \
		--out staging/feature_register.json
	$(VENV)/bin/python -c "from pathlib import Path; from sensors.lfs import write_feature_ref_demo; print(write_feature_ref_demo(Path('staging/feature_register.json'), Path('staging/feature_ref_emissions.json')))"
	@echo "features-demo ok — see staging/audio_put.json staging/feature_register.json"

# ---------- branch hygiene (braid) ------------------------------------------
_BRAID_HINT = braid not installed — uv tool install --editable ~/workspace/braid
_GIT_HINT = not a git repo yet — git init (then braid can forecast landings)

ready:
	@command -v braid >/dev/null || { echo "$(_BRAID_HINT)"; exit 0; }; \
	git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "$(_GIT_HINT)"; exit 0; }; \
	braid ready

collide:
	@command -v braid >/dev/null || { echo "$(_BRAID_HINT)"; exit 0; }; \
	git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "$(_GIT_HINT)"; exit 0; }; \
	braid status

land-budget:
	@command -v braid >/dev/null || { echo "$(_BRAID_HINT)"; exit 0; }; \
	git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "$(_GIT_HINT)"; exit 0; }; \
	braid budget

land-verify:
	@command -v braid >/dev/null || { echo "$(_BRAID_HINT)"; exit 0; }; \
	git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "$(_GIT_HINT)"; exit 0; }; \
	braid verify --since 'HEAD@{1}' || true; \
	cargo check --workspace
