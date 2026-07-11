# cast — output/embodiment lane (Stratum 2)

**Status:** design approved 2026-07-11, pending spec review
**Working title:** `cast` (rename before scaffold; not load-bearing)
**Home:** new standalone repo `~/Desktop/cast`, in the spirit of `~/Desktop/mashup_compiler`
(clean, BYO-audio, no corpus, no scraper, no downloader).

---

## North star this serves

The summit is a **fully autonomous party DJ**: reads the room, takes natural-language
requests, plays a continuous, beatmatched, taste-driven set through real speakers,
pulling from Spotify/SoundCloud/corpus, out to Sonos and beyond. Built as a "layers of
rock" stack — each stratum requires the last and recombines everything beneath it:

| Stratum | Layer | Status |
|---|---|---|
| S0 | Understanding — `tracklist_engine` (identity, analysis, taste + transition knowledge) | exists |
| S1 | Mixing primitive — `mashup_compiler` (one beatmatched/key-shifted/LUFS transition; the atom of a mix) | exists |
| **S2** | **Embodiment/output — `cast` (this spec)** | **now** |
| S3 | The set — continuous N-track autonomous mix from corpus | later |
| S4 | Open library — Spotify/SoundCloud as discovery + request resolution + (legal) audio; universal Source | later |
| S5 | The autonomous DJ — agent fusing all strata: reads room, takes requests, decides what-next-&-why, mixes live, plays out | summit |

**A (this repo) is the embodiment layer.** Every higher stratum is inert without it —
S3's set has nowhere to play, S4's requests have nowhere to land, the S5 agent has no
voice and no ears. Its MCP/NL surface is where "take a request" is *born*: today
"play this in the living room, quieter," tomorrow "play something like this but darker."

## Scope of this sub-project

**In:** play and control audio on real targets (Mac + Sonos zones), zone-aware,
driven by Claude in natural language via an **MCP server**. Plays any file path or URL.

**Out (each a later stratum / sub-project):** rendering (S1/S3), multi-zone grouping,
live/continuous streaming, voice STT (NL is text-through-Claude for now), source/
discovery (S4), lights/home. Keeping S2 a pure, universal Sink is what makes it
shippable in a weekend.

## Architecture

```
Claude (desktop / Code / phone)
   │  MCP (stdio)
   ▼
cast MCP server ──► control core (library)
                     ├─ Sink (protocol)         ← the universal abstraction
                     │    ├─ LocalSink   (sounddevice stream thread)
                     │    └─ SonosSink   (SoCo + local HTTP file server)
                     ├─ ZoneRegistry     (discover targets, resolve NL name → Sink)
                     └─ PlaybackState    (what's playing where)
```

Style follows the tracklist_engine house guide (Rust-flavoured functional Python):
explicit + typed, frozen dataclasses for records, errors-as-values in core
(`Result`), fail-fast/clean-message at the MCP edge.

## Components

### Sink (protocol) — the piece that outlives A

```python
class Sink(Protocol):
    def play(self, source: AudioSource) -> Result[None]
    def pause(self) -> Result[None]
    def resume(self) -> Result[None]
    def stop(self) -> Result[None]
    def set_volume(self, level: int) -> Result[None]   # 0–100
    def now_playing(self) -> NowPlaying | None
```

- `AudioSource` — frozen dataclass over a file path **or** URL.
- `NowPlaying` — frozen dataclass: source, position/duration if known, state.
- One interface, N targets. When S3 (set renderer), S4 (sources), or AirPlay/
  Chromecast arrive, they are new Sinks / new sources, not a rewrite.

### LocalSink (Mac)

Background thread streaming frames (`soundfile` decode → `sounddevice` output),
honoring atomic pause/stop/volume flags. Chosen over `afplay` **specifically because
`afplay` cannot pause or set volume** — and NL control is the entire point. Handles
mono→stereo upmix and samplerate from the file.

### SonosSink (the interesting one)

Sonos plays URLs, not file paths. So:

1. A small **threaded HTTP server** serves the target file, bound to the Mac's
   **LAN IP** (resolved via a UDP-socket trick; note: **Tailscale/MagicDNS will not
   reach Sonos** — must be the same-LAN address the Sonos can fetch).
2. `device.play_uri("http://<lan-ip>:<port>/<file>")`.
3. Control via **SoCo**: `pause()` / `play()` / `stop()`, `.volume = n`,
   `get_current_track_info()` for `now_playing`.

SoCo chosen over the Sonos cloud Control API: local, no OAuth, full queue control,
sub-second latency. Sonos buffers ~2–4 s on a fresh URL — fine for playing a file,
irrelevant to this scope (no live interactivity here).

### ZoneRegistry

Enumerates targets: `local` (always present) + Sonos zones from `soco.discover()`.
Resolves an NL name → Sink via case-insensitive / fuzzy match on zone name
("living room" → the `Living Room` device). Caches discovered devices; re-discovers
on miss.

### MCP server

stdio transport (works in Claude desktop app + Claude Code). Tools:

- `list_zones()` → `[{name, type, state}]`
- `play(source, zone)` — source = path or URL; zone name
- `pause(zone)` · `resume(zone)` · `stop(zone)`
- `set_volume(zone, level)` — 0–100
- `now_playing(zone)`

Holds the ZoneRegistry + live Sink instances + PlaybackState. Core returns `Result`;
the MCP layer converts `Err` into a clean tool-error message Claude can relay
(Sonos offline, zone not found, file missing, LAN IP unreachable).

## Data flow (happy path)

1. Claude: "play `out/demo.wav` in the living room, kinda quiet."
2. MCP `play(source="out/demo.wav", zone="living room")` → ZoneRegistry resolves
   `living room` → SonosSink(`Living Room`).
3. SonosSink starts the HTTP server for the file, computes `http://<lan-ip>:<port>/demo.wav`,
   calls `device.play_uri(...)`.
4. Claude: `set_volume("living room", 30)` → `device.volume = 30`.
5. `now_playing("living room")` → SoCo track info → `NowPlaying`.

## Error handling

- Core functions return `Result[T]`; no exceptions across module boundaries.
- Edge (MCP tool) maps `Err(msg)` → structured tool error. Cases: zone not found
  (list available), Sonos unreachable, file/URL not found, LAN IP undetectable,
  unsupported audio format.

## Testing

- **FakeSink** (Protocol → trivial) drives MCP-tool + ZoneRegistry name-resolution
  unit tests.
- **LocalSink**: pause/stop/volume state machine tested against a short synthetic
  buffer with `sounddevice` mocked (no real device in CI).
- **SonosSink**: mocked SoCo device — assert a well-formed LAN URL is passed to
  `play_uri`, and that the HTTP server actually serves the requested file bytes.
- **Manual demo-day checks** (not automatable, mirroring the compiler's "open in
  Ableton" checks): real playback on a real Sonos zone; volume/pause/stop by NL.

## Proposed repo layout

```
cast/
  cast/
    __init__.py
    sinks/
      base.py        # Sink protocol, AudioSource, NowPlaying
      local.py       # LocalSink (sounddevice thread)
      sonos.py       # SonosSink (SoCo + http)
    http_server.py   # local file server for Sonos
    zones.py         # ZoneRegistry (discover + resolve)
    netutil.py       # LAN IP resolution
    result.py        # Result type (house idiom)
  server/
    mcp.py           # MCP server exposing the tools
  tests/
  README.md
  requirements.txt   # soco, sounddevice, soundfile, mcp (+ python 3.14 to match compiler)
```

## Open questions (resolve during planning, not blocking)

- Final repo/tool name.
- `now_playing` position accuracy for LocalSink (track our own frame counter).
- Whether the HTTP file server is per-play ephemeral or a long-lived singleton.
