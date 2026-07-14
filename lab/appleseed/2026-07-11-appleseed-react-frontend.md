# Appleseed React/TypeScript Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the hand-rolled Jinja + vanilla-JS UI with a typed React/Vite/TypeScript SPA talking to the existing Python FastAPI over JSON — killing the CSS/JS bug class (the name-gate and picker overlays that froze the app) and unlocking a maintainable base for "make it sick." The Python engine + compiler + API are untouched.

**Architecture:** `web/` (new, in mashup_compiler) = Vite+React+TS SPA. FastAPI keeps all `/api/*` JSON endpoints (adds 2: `/api/feed`, `/api/library`) and serves the built SPA bundle with an index.html fallback; the Jinja templates are retired at the end. Dev: Vite dev server proxies `/api` + `/media` to :8500. Prod/tailnet: `vite build` → `web/dist` served by FastAPI. **Every interactive view ships a Playwright test that actually clicks** — this is the point of the switch.

**Tech Stack:** Vite, React 18, TypeScript (strict), Vitest + @testing-library/react (unit), Playwright (interaction), the existing FastAPI backend. No CSS framework — port the Serato-mono tokens as a global stylesheet + CSS Modules.

## Global Constraints

- Repo: `/Users/johnnycabrahams/Desktop/mashup_compiler` ($MC). New frontend under `$MC/web/`.
- **Python engine/compiler/ are untouched.** Server (`server/app.py`) only GAINS 2 JSON endpoints + static-serving; existing `/api/*` unchanged. `engine/` FROZEN.
- Python offline suite stays green (`venv/bin/python -m pytest tests/ -m "not integration"` → 53 passed, 1 deselected) after any server change.
- TS is **strict** (`"strict": true`); no `any` without a written reason. Every interactive view has a Playwright test that clicks the primary flow and asserts a real state change (NOT just "renders").
- Design is the shipped **Serato mono**: ink `#0a0a0b`, panels `#131316`/`#1a1a1f`, line `#26262c`, paper `#ededf0`, dim `#86868f`, accent `#33e1cf`; Space Grotesk (UI) + Space Mono (data); radius 7px; no per-key hue. Port these as CSS variables — do not invent a new palette.
- The API contracts are FIXED by the existing backend — the SPA consumes them as-is:
  - `GET /api/feed` (NEW) → `{mashes: [{id, vocal_title, vocal_artist, instr_title, instr_artist, drop_bar, status, error, created_by, verb, parent_mash_id}]}`
  - `GET /api/library` (NEW) → `{songs: [{id, title, artist, status, bpm, key_tonic, key_mode, error}], requests: [{id, query, status, error}]}`
  - `GET /api/status` → `{songs, mashes, queue}` (exists)
  - `GET /api/compatible?vocal_id=N` → `{candidates: [{id, title, artist, bpm, key_tonic, key_mode, stretch, transpose}]}` (exists)
  - `POST /api/mash` (form vocal_id, instr_id, drop_bar) → `{id}` (exists)
  - `POST /api/refine` (form mash_id, verb) → `{id}` (exists)
  - `POST /api/request` (form query) → `{id}` (exists); `GET /api/requests` (exists)
  - `POST /api/name` (form name) → 303 (exists); media `GET /media/mash/{id}.{wav,als}` (exists)
- A global hook ruff-formats .py; it does not touch .ts/.tsx/.css.

---

### Task 1: Scaffold web/ (Vite+React+TS+Vitest+Playwright) + FastAPI static serving

**Files:** create `$MC/web/{package.json, tsconfig.json, vite.config.ts, playwright.config.ts, index.html, src/main.tsx, src/App.tsx, src/api.ts, src/styles/tokens.css}`; modify `$MC/server/app.py` (serve the SPA), `$MC/.gitignore` (web/node_modules, web/dist).

**Interfaces:**
- Produces: `web/src/api.ts` — a typed fetch layer: `getFeed()`, `getLibrary()`, `getCompatible(vocalId)`, `postMash(vocalId, instrId, dropBar)`, `postRefine(mashId, verb)`, `postRequest(query)`, `postName(name)` — each returning typed promises. Base URL empty (same origin); dev proxy handles it.
- Produces: FastAPI serves `web/dist` at `/` with SPA fallback, `/api/*` + `/media/*` unchanged.

- [ ] **Step 1: Scaffold** — `cd $MC && npm create vite@latest web -- --template react-ts` then in `web/`: `npm i` and `npm i -D vitest @testing-library/react @testing-library/user-event jsdom @playwright/test`. Set `tsconfig.json` `"strict": true`. Add `.gitignore` entries `web/node_modules/`, `web/dist/`, `web/test-results/`.

- [ ] **Step 2: vite.config.ts** — React plugin; dev server `proxy: { "/api": "http://localhost:8500", "/media": "http://localhost:8500" }`; `build.outDir: "dist"`; vitest `environment: "jsdom"`.

- [ ] **Step 3: playwright.config.ts** — `webServer` starts Vite preview (or dev) + assumes the Python API on :8500; `baseURL` the Vite port; one project (chromium). testDir `web/tests`.

- [ ] **Step 4: `web/src/api.ts`** — typed API client. Define the response types (Mash, Song, Request, Candidate) from the FIXED contracts above; implement each function with `fetch`, throwing on non-ok. Form-POSTs use `new URLSearchParams`.

- [ ] **Step 5: `tokens.css`** — the Serato-mono CSS variables (exact hex above) + font imports (Space Grotesk + Space Mono) + a minimal reset. Global.

- [ ] **Step 6: Minimal App.tsx** — renders "Appleseed" wordmark + the bottom tab nav (Feed/＋/Library) as routes (use a tiny hash-router or react-router). Empty view stubs for now.

- [ ] **Step 7: FastAPI static serving** — in `server/app.py`: if `web/dist` exists, `app.mount("/", StaticFiles(directory=web_dist, html=True))` LAST (after all /api and /media routes), and add an index.html fallback for SPA client routes. Guard: only mount if the dir exists (so the Python tests + dev without a build still work). Do NOT remove the Jinja routes yet (retired in Task 6).

- [ ] **Step 8: Verify** — `cd web && npm run build` succeeds; `npx tsc --noEmit` clean; Python suite still 53 passed; `npx vitest run` (no tests yet → passes/empty). Commit: `git add -A && git commit -m "feat(web): scaffold React/Vite/TS SPA + typed api client + FastAPI static serving"`

---

### Task 2: Backend JSON completeness (`/api/feed`, `/api/library`)

**Files:** modify `$MC/server/app.py`; test `$MC/tests/test_json_views.py`

**Interfaces:** the two NEW endpoints per the contracts above. `/api/feed` mirrors the current feed() JOIN as JSON; `/api/library` mirrors library() (songs + non-done requests) as JSON.

- [ ] **Step 1: Failing tests** (`tests/test_json_views.py`) — TestClient fixture (tmp DB, monkeypatch _scan/enqueue like test_refine.py); seed 2 ready songs + 1 done mash + 1 searching request; assert `GET /api/feed` returns the mash with vocal_title/instr_title, and `GET /api/library` returns 2 songs + 1 request.

- [ ] **Step 2: Run RED.**

- [ ] **Step 3: Implement** the two endpoints in app.py reusing the existing SQL (the feed() JOIN, the library() song+request queries) but returning `dict` lists instead of TemplateResponse.

- [ ] **Step 4: GREEN** (2 tests) + full Python suite (55 passed, 1 deselected).

- [ ] **Step 5: Commit** — `git commit -am "feat(server): /api/feed + /api/library JSON endpoints for the SPA"`

---

### Task 3: Design system components (mono, typed)

**Files:** create `$MC/web/src/components/{Nav.tsx, Chip.tsx, Card.tsx, Pill.tsx}` + `web/src/styles/*.module.css`; test `web/src/components/Chip.test.tsx`

**Interfaces:** `Nav` (bottom tabs, active state, ＋ center); `Chip({label, tone?})`; `Card` (mono panel wrapper); `Pill` (primary button). All Serato-mono.

- [ ] **Step 1: Failing Vitest** — render `<Chip label="128 bpm" />`, assert text present + the mono class applied.
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** the four components with CSS Modules using the tokens (no per-key hue). Nav uses the accent for the active tab + the center ＋.
- [ ] **Step 4: GREEN** (`npx vitest run`).
- [ ] **Step 5: Commit** — `git commit -am "feat(web): mono design-system components (Nav/Chip/Card/Pill)"`

---

### Task 4: Feed view + interaction test

**Files:** create `$MC/web/src/views/Feed.tsx`, `web/src/components/MashCard.tsx`, `web/tests/feed.spec.ts`

**Interfaces:** `Feed` fetches `getFeed()`, renders `MashCard`s (pair title, creator, `<audio>` player, .als download link, 4 verb buttons, lineage line when `verb` set, failed/working states); polls while any mash is non-done.

- [ ] **Step 1: Failing Playwright** (`web/tests/feed.spec.ts`) — with the Python API running (seeded demo mashes exist), load `/`, assert a mash card renders with an audio element and 4 verb buttons; click "different hook" and assert a POST to /api/refine fires (intercept the request) — a REAL interaction assertion, not just presence.
- [ ] **Step 2: RED** (view not built).
- [ ] **Step 3: Implement** Feed + MashCard (mono). Verb click → `postRefine` → optimistic "queued" state or reload the feed.
- [ ] **Step 4: GREEN** (`npx playwright test feed`) with the API up; also a Vitest unit test for MashCard rendering states (done/failed/working).
- [ ] **Step 5: Commit** — `git commit -am "feat(web): Feed view + MashCard with verbs (playwright interaction test)"`

---

### Task 5: New Mash view + the picker (the freeze-bug regression test)

**Files:** create `$MC/web/src/views/NewMash.tsx`, `web/src/components/PickerSheet.tsx`, `web/tests/newmash.spec.ts`

**Interfaces:** two slots (vocal/instr), a bottom-sheet `PickerSheet` (opens on slot click, lists songs / gate-compatible candidates, closes on pick/cancel/backdrop), the seam line showing live gate math when both chosen, a drop-bar stepper, the Mash button (enabled only on a valid pair). Picker visibility is REACT STATE (`open: boolean`) — no CSS `hidden`-attribute footgun.

- [ ] **Step 1: Failing Playwright** (`web/tests/newmash.spec.ts`) — THE regression test for the freeze: load `/new`; assert the picker sheet is NOT covering the page initially (the page's Mash button / a slot is clickable — `await expect(slot).toBeVisible()` AND clickable: click it and the sheet appears); pick a vocal, assert the sheet CLOSES (backdrop gone, page interactive again — click something behind where it was); pick a compatible instrumental; assert the seam shows the stretch/transpose math; click Mash and assert POST /api/mash fires. This test would FAIL on the old overlay bug.
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** NewMash + PickerSheet. Sheet mount/visibility driven by `useState` (render nothing when closed, or a portal that's absent from the DOM when closed — NOT a CSS-hidden always-mounted overlay). Fetch candidates via `getCompatible` on vocal pick. Seam math from the chosen candidate's stretch/transpose.
- [ ] **Step 4: GREEN** (`npx playwright test newmash`).
- [ ] **Step 5: Commit** — `git commit -am "feat(web): New Mash view + picker sheet (state-driven, freeze regression test)"`

---

### Task 6: Library view + add-song + retire Jinja

**Files:** create `$MC/web/src/views/Library.tsx`, `web/tests/library.spec.ts`; modify `$MC/server/app.py` (retire the 3 Jinja page routes now the SPA owns `/`, `/new`, `/library`); remove `server/templates/*` + the Jinja dependency IF nothing else uses them.

**Interfaces:** Library lists songs (status chips, BPM/key) + the "add any song" input (posts `getRequest`) + pending-request list; polls `/api/library` while anything is searching/analyzing. The optional name field lives in the header (App shell), non-blocking.

- [ ] **Step 1: Failing Playwright** (`web/tests/library.spec.ts`) — load `/library`; type a query in the add-song box, submit, assert POST /api/request fires and a pending "searching" row appears; assert NOTHING is greyed out / the input is focusable (name-gate/freeze regression).
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** Library + the header name field (optional, non-blocking) in the App shell.
- [ ] **Step 4:** Retire Jinja — in `server/app.py`, once `web/dist` exists and the SPA serves all three pages, delete the `feed()/new_mash()/library()` HTML routes and the `templates` mount; confirm `/`, `/new`, `/library` are served by the SPA fallback. Keep all `/api/*` + `/media/*`. Remove `templates/` dir + `jinja2` from requirements if unused. Python suite must stay green (the Jinja-page tests, if any, are removed with the routes; the /api tests remain).
- [ ] **Step 5: GREEN** (playwright library) + full Python suite green.
- [ ] **Step 6: Commit** — `git commit -am "feat(web): Library view + add-song; retire Jinja templates (SPA owns the pages)"`

---

### Task 7: Build, wire, and go live

**Files:** modify `$MC/README.md`, `$MC/BACKLOG.md`; add `$MC/web/README.md` (dev/build commands)

- [ ] **Step 1: Production build** — `cd $MC/web && npm run build` → `web/dist`. Confirm `server/app.py` serves it (dir-exists guard from Task 1).
- [ ] **Step 2: Full green** — Python suite (all /api tests) + `npx vitest run` + `npx playwright test` (all three interaction specs) against the running API. **All three freeze-regression tests must pass** — this is the acceptance bar the old app failed.
- [ ] **Step 3: Restart live** — controller restarts uvicorn on :8500; the SPA is now the app. Verify `/`, `/new`, `/library` load the React bundle and the picker opens/closes (the actual thing that was frozen).
- [ ] **Step 4: Docs** — README: dev (`npm run dev` + uvicorn) vs prod (`npm run build` + uvicorn serves dist); note the Python engine is unchanged. BACKLOG: mark "TS frontend" done; add "make it sick" items now unblocked (real waveform rendering, generate animation, scrub/seek). Commit.

---

## Deferred (NOT in this plan)
- Server-to-Node migration (stays FastAPI — the coin-flip we declined).
- Real waveform rendering / animations (the payoff this unblocks; separate "make it sick" plan).
- Any change to compiler/ or engine/.

## Self-review notes
- The whole point — interaction tests — is enforced: Tasks 4/5/6 each ship a Playwright spec that CLICKS and asserts state change; Task 5's is the explicit freeze regression. The old bug (200 OK but unclickable) cannot survive these.
- Boundary held: Python engine/compiler untouched; server only gains 2 JSON routes + static serving, then sheds Jinja. TS is strict.
- API contracts are consumed as the backend already defines them (2 additions for feed/library that the Jinja pages currently render server-side).
- Timing risk (stated to John): this is a multi-day rebuild producing the same features with a maintainable, tested surface. The current app keeps working until Task 6 retires Jinja, so nothing breaks mid-flight.
