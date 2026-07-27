# sensors (Python Propose layer)

Project root: `core/python/`. Import package: `sensors`.

```text
python/
  pyproject.toml      # install: pip install -e "python[dev]" from core/
  tests/
  sensors/         # importable package (lfs, plugs, mert, …)
  .venv/
```

Typed JSON/IPC boundary for MIR, download, Ableton, and labeling functions.
The Rust kernel owns identity, provenance, and the PWS matrix; this package
validates payloads against `core/schemas/` so agents get precise errors.

- LFs: `sensors.lfs` (claimed stem, title heuristic, feature ref)
- Experimental Propose plugs: `sensors.plugs` (open-vocab OD, agentic search)

See also: [`docs/storage.md`](../docs/storage.md), [`docs/vision_od.md`](../docs/vision_od.md).
