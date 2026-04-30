const loadTracksBtn = document.getElementById("loadTracksBtn");
const trackListEl = document.getElementById("trackList");
const trackSearchInput = document.getElementById("trackSearchInput");
const trackBrowserScrollEl = document.getElementById("trackBrowserScroll");
const arrangerEl = document.getElementById("arranger");
const lanesContainerEl = document.getElementById("lanesContainer");
const timeRulerEl = document.getElementById("timeRuler");
const apiBaseInput = document.getElementById("apiBase");
const themeSelect = document.getElementById("themeSelect");
const projectSelect = document.getElementById("projectSelect");
const masterBpmInput = document.getElementById("masterBpmInput");
const masterCamelotSelect = document.getElementById("masterCamelotSelect");
const saveProjectBtn = document.getElementById("saveProjectBtn");
const autoSyncCheckbox = document.getElementById("autoSyncCheckbox");
const zoomOutBtn = document.getElementById("zoomOutBtn");
const zoomInBtn = document.getElementById("zoomInBtn");
const zoomSlider = document.getElementById("zoomSlider");
const zoomValue = document.getElementById("zoomValue");
const splitClipBtn = document.getElementById("splitClipBtn");
const trimLeftBtn = document.getElementById("trimLeftBtn");
const trimRightBtn = document.getElementById("trimRightBtn");
const moveClipUpBtn = document.getElementById("moveClipUpBtn");
const moveClipDownBtn = document.getElementById("moveClipDownBtn");
const addLaneBtn = document.getElementById("addLaneBtn");
const deleteClipBtn = document.getElementById("deleteClipBtn");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const stopBtn = document.getElementById("stopBtn");
const playheadEl = document.getElementById("playhead");
const loopRegionEl = document.getElementById("loopRegion");
const loopEnabledCheckbox = document.getElementById("loopEnabledCheckbox");
const loopStartInput = document.getElementById("loopStartInput");
const loopEndInput = document.getElementById("loopEndInput");
const setLoopFromPlayheadBtn = document.getElementById("setLoopFromPlayheadBtn");
const setLoopEndFromPlayheadBtn = document.getElementById("setLoopEndFromPlayheadBtn");
const clipVolumeSlider = document.getElementById("clipVolumeSlider");
const clipVolumeValue = document.getElementById("clipVolumeValue");

const BASE_PX_PER_SEC = 18;
const RULER_HEIGHT = 24;
let dragState = null;
let selectedClipId = null;
let tracksById = new Map();
let playheadSeconds = 0;
let transportTimer = null;
let audioCtx = null;
let projectClips = [];
const decodedBufferCache = new Map();
const stemManifestCache = new Map();
let activeSources = [];
const clipVolumeById = new Map();
const trackVariantSelections = new Map();
let transportIsRunning = false;
let laneCount = 3;
let allTracks = [];
let zoomLevel = 1;
let isPointerOverArranger = false;

function pxPerSec() {
  return BASE_PX_PER_SEC * zoomLevel;
}

function secToPx(sec) {
  return sec * pxPerSec();
}

function pxToSec(px) {
  return px / pxPerSec();
}

function quantStepSec() {
  const pps = pxPerSec();
  if (pps < 16) return 4;
  if (pps < 24) return 2;
  if (pps < 40) return 1;
  if (pps < 72) return 0.5;
  return 0.25;
}

function quantizeSeconds(sec) {
  const step = quantStepSec();
  return Math.max(0, Math.round((Math.round(sec / step) * step) * 1000) / 1000);
}

function apiBase() {
  return apiBaseInput.value.trim();
}

function keyLabel(keyPc, keyMode) {
  if (keyPc === null || keyPc === undefined) return "Unknown";
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const mode = keyMode === "minor" ? "m" : keyMode === "major" ? "M" : "";
  return `${names[keyPc]}${mode}`;
}

function clearClips() {
  arrangerEl.querySelectorAll(".clip").forEach((el) => el.remove());
}

function renderLanes() {
  lanesContainerEl.innerHTML = "";
  for (let laneIdx = 1; laneIdx <= laneCount; laneIdx += 1) {
    const lane = document.createElement("div");
    lane.className = "lane";
    lane.dataset.lane = String(laneIdx);
    lanesContainerEl.appendChild(lane);
  }
}

function timelineContentWidthPx() {
  const maxEndSec = projectClips.reduce((acc, clip) => {
    const end = Number(clip.timeline_start_s) + clipDurationTimelineSeconds(clip);
    return Math.max(acc, end);
  }, 16);
  const minTimelineSec = 300; // keep a broad pan-able window even with few/no clips
  const lookaheadSec = playheadSeconds + 120;
  const horizonSec = Math.max(maxEndSec, minTimelineSec, lookaheadSec);
  return Math.max(arrangerEl.clientWidth, Math.ceil(secToPx(horizonSec) + 320));
}

function applyTimelineWidth() {
  const w = timelineContentWidthPx();
  lanesContainerEl.style.width = `${w}px`;
  timeRulerEl.style.width = `${w}px`;
}

function renderTimeMarkers() {
  const ctx = timeRulerEl.getContext("2d");
  if (!ctx) return;
  const cssWidth = timelineContentWidthPx();
  const cssHeight = RULER_HEIGHT;
  const dpr = window.devicePixelRatio || 1;
  timeRulerEl.width = Math.max(1, Math.floor(cssWidth * dpr));
  timeRulerEl.height = Math.max(1, Math.floor(cssHeight * dpr));
  timeRulerEl.style.width = `${cssWidth}px`;
  timeRulerEl.style.height = `${cssHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const styles = getComputedStyle(document.body);
  const panel2 = styles.getPropertyValue("--panel-2").trim() || "#24272e";
  const laneBorder = styles.getPropertyValue("--lane-border").trim() || "#313540";
  const muted = styles.getPropertyValue("--muted").trim() || "#a6acb8";
  ctx.fillStyle = panel2;
  ctx.fillRect(0, 0, cssWidth, cssHeight);
  ctx.strokeStyle = laneBorder;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, cssHeight - 0.5);
  ctx.lineTo(cssWidth, cssHeight - 0.5);
  ctx.stroke();

  const step = quantStepSec();
  const totalSec = Math.ceil(pxToSec(cssWidth));
  const majorStep = step * 4;
  ctx.font = "10px Inter, system-ui, sans-serif";
  ctx.textBaseline = "top";
  for (let sec = 0; sec <= totalSec + 0.0001; sec += step) {
    const major = Math.abs((sec / majorStep) - Math.round(sec / majorStep)) < 0.001;
    const x = Math.floor(secToPx(sec)) + 0.5;
    ctx.strokeStyle = major ? muted : laneBorder;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cssHeight);
    ctx.stroke();
    if (major) {
      ctx.fillStyle = muted;
      ctx.fillText(`${Math.round(sec * 100) / 100}s`, x + 3, 2);
    }
  }
}

function selectedProjectId() {
  const id = Number(projectSelect.value);
  if (!id) return null;
  return id;
}

function clipDurationTimelineSeconds(clip) {
  const srcDur = Number(clip.src_end_s) - Number(clip.src_start_s);
  return srcDur / Number(clip.tempo_ratio || 1);
}

async function loadProjects() {
  const resp = await fetch(`${apiBase()}/projects`);
  if (!resp.ok) {
    alert(`Failed to load projects: ${resp.status}`);
    return;
  }
  const payload = await resp.json();
  const items = payload.items || [];
  projectSelect.innerHTML = "";
  items.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = String(p.project_id);
    opt.textContent = `#${p.project_id} ${p.name}`;
    projectSelect.appendChild(opt);
  });
  if (items.length > 0) {
    projectSelect.value = String(items[0].project_id);
    applyProjectFields(items[0]);
    await loadProjectClips();
  }
}

function applyProjectFields(project) {
  masterBpmInput.value = String(project.master_bpm ?? 128);
  masterCamelotSelect.value = project.master_camelot || "";
}

async function saveProjectSettings() {
  const projectId = selectedProjectId();
  if (!projectId) return;
  const body = {
    master_bpm: Number(masterBpmInput.value || 128),
    master_camelot: masterCamelotSelect.value || null,
  };
  const resp = await fetch(`${apiBase()}/projects/${projectId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    alert(`Failed to save project: ${resp.status}`);
  }
}

async function loadTracks() {
  const resp = await fetch(`${apiBase()}/tracks?limit=150`);
  if (!resp.ok) {
    alert(`Failed to load tracks: ${resp.status}`);
    return;
  }
  const payload = await resp.json();
  allTracks = payload.items || [];
  renderTrackList(allTracks);
}

function renderTrackList(items) {
  tracksById = new Map();
  trackListEl.innerHTML = "";
  items.forEach((track) => {
    tracksById.set(track.track_id, track);
    const li = document.createElement("li");
    const variants = track.available_variants && track.available_variants.length > 0
      ? track.available_variants
      : ["original"];
    const selectedVariant = trackVariantSelections.get(track.track_id) || (variants.includes("original") ? "original" : variants[0]);
    const stems = Array.isArray(track.available_stems) ? track.available_stems : [];
    const derivedVariants = (track.available_variants || []).filter((v) => v === "acappella" || v === "instrumental");
    li.innerHTML = `
      <strong>${track.song_name || track.track_id}</strong><br/>
      <span>${track.artist_name || "Unknown artist"}</span><br/>
      <span>BPM: ${track.bpm ?? "?"} | Key: ${keyLabel(track.key_pc, track.key_mode)} | ${track.platform || "?"}/${track.variant_tag || "?"}</span><br/>
      <span class="muted">ID: ${track.track_id}</span><br/>
      <span class="muted">${track.release_label || ""} ${track.isrc ? `| ISRC ${track.isrc}` : ""}</span><br/>
      <span class="muted">meta: ${track.metadata_source || "unresolved"}</span><br/>
      <span class="muted">stems: ${stems.length ? stems.join(", ") : "none"}</span><br/>
      <span class="muted">derived: ${derivedVariants.length ? derivedVariants.join(", ") : "none"}</span>
    `;
    const variantSelect = document.createElement("select");
    variants.forEach((variant) => {
      const opt = document.createElement("option");
      opt.value = variant;
      opt.textContent = variant;
      variantSelect.appendChild(opt);
    });
    variantSelect.value = selectedVariant;
    variantSelect.addEventListener("change", () => {
      trackVariantSelections.set(track.track_id, variantSelect.value);
    });
    const addBtn = document.createElement("button");
    addBtn.textContent = "Add";
    addBtn.addEventListener("click", () => createClip(track.track_id, 1, variantSelect.value));
    li.appendChild(document.createElement("br"));
    li.appendChild(variantSelect);
    li.appendChild(addBtn);
    trackListEl.appendChild(li);
  });
}

function applyTrackSearchFilter() {
  const q = (trackSearchInput.value || "").trim().toLowerCase();
  if (!q) {
    renderTrackList(allTracks);
    return;
  }
  const filtered = allTracks.filter((track) => {
    const hay = [
      track.song_name || "",
      track.artist_name || "",
      track.track_id || "",
      track.scraped_track_title || "",
    ]
      .join(" ")
      .toLowerCase();
    return hay.includes(q);
  });
  renderTrackList(filtered);
}

async function createClip(trackId, laneIdx, variantTag) {
  const projectId = selectedProjectId();
  if (!projectId) return;
  const body = {
    track_id: trackId,
    variant_tag: variantTag || "original",
    lane_idx: Math.max(1, Math.min(256, Number(laneIdx) || 1)),
    timeline_start_s: 0,
    src_start_s: 0,
    src_end_s: 32,
    auto_sync: autoSyncCheckbox.checked,
  };
  const resp = await fetch(`${apiBase()}/projects/${projectId}/clips`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    alert(`Failed to create clip: ${resp.status}`);
    return;
  }
  await loadProjectClips();
}

async function loadProjectClips() {
  const projectId = selectedProjectId();
  if (!projectId) return;
  const resp = await fetch(`${apiBase()}/projects/${projectId}/clips`);
  if (!resp.ok) {
    alert(`Failed to load clips: ${resp.status}`);
    return;
  }
  const payload = await resp.json();
  projectClips = payload.items || [];
  const maxLane = projectClips.reduce((acc, clip) => Math.max(acc, Number(clip.lane_idx) || 1), 1);
  laneCount = Math.max(laneCount, maxLane);
  renderLanes();
  applyTimelineWidth();
  renderTimeMarkers();
  clearClips();
  projectClips.forEach((clip) => renderClip(clip));
  if (selectedClipId !== null) {
    const exists = projectClips.some((c) => Number(c.clip_id) === Number(selectedClipId));
    if (!exists) selectedClipId = null;
  }
  syncSelectedClass();
}

function renderClip(clip) {
  const lane = arrangerEl.querySelector(`.lane[data-lane="${clip.lane_idx}"]`);
  if (!lane) return;
  const track = tracksById.get(clip.track_id);
  const el = document.createElement("div");
  el.className = "clip";
  el.dataset.clipId = String(clip.clip_id);
  el.dataset.lane = String(clip.lane_idx);
  el.style.left = `${secToPx(Number(clip.timeline_start_s))}px`;
  el.style.width = `${secToPx(clipDurationTimelineSeconds(clip))}px`;
  const waveformCanvas = document.createElement("canvas");
  waveformCanvas.className = "waveform";
  el.appendChild(waveformCanvas);
  const leftHandle = document.createElement("div");
  leftHandle.className = "trim-handle left";
  leftHandle.addEventListener("pointerdown", (event) => onTrimHandlePointerDown(event, clip, "left"));
  el.appendChild(leftHandle);
  const rightHandle = document.createElement("div");
  rightHandle.className = "trim-handle right";
  rightHandle.addEventListener("pointerdown", (event) => onTrimHandlePointerDown(event, clip, "right"));
  el.appendChild(rightHandle);
  const clipName = track?.song_name || clip.track_id;
  const label = document.createElement("div");
  label.className = "clip-label";
  label.textContent = `${clipName} [${clip.variant_tag || "original"}] (${track?.bpm ?? "?"} BPM, shift ${clip.pitch_shift_semi})`;
  el.appendChild(label);
  el.addEventListener("click", () => {
    selectedClipId = Number(clip.clip_id);
    syncSelectedClass();
    syncVolumeUiForSelection();
  });
  el.addEventListener("pointerdown", onClipPointerDown);
  lane.appendChild(el);
  renderClipWaveform(el, clip).catch((err) => {
    console.warn("waveform render failed", clip.clip_id, err);
  });
}

function syncSelectedClass() {
  arrangerEl.querySelectorAll(".clip").forEach((el) => {
    const isSelected = Number(el.dataset.clipId) === Number(selectedClipId);
    el.classList.toggle("selected", isSelected);
  });
}

function onClipPointerDown(event) {
  const clip = event.currentTarget;
  const clipId = Number(clip.dataset.clipId);
  selectedClipId = clipId;
  syncSelectedClass();
  syncVolumeUiForSelection();
  dragState = {
    mode: "move",
    clip,
    clipId,
    pointerStartX: event.clientX,
    pointerStartY: event.clientY,
    clipStartLeft: parseInt(clip.style.left, 10) || 0,
    laneStartIdx: Number(clip.dataset.lane || 1),
  };
  clip.setPointerCapture(event.pointerId);
  clip.style.cursor = "grabbing";
}

function onPointerMove(event) {
  if (!dragState) return;
  if (dragState.mode === "trim-left" || dragState.mode === "trim-right") {
    onTrimPointerMove(event);
    return;
  }
  const dx = event.clientX - dragState.pointerStartX;
  const rawLeft = Math.max(0, dragState.clipStartLeft + dx);
  const newLeft = secToPx(quantizeSeconds(pxToSec(rawLeft)));
  dragState.clip.style.left = `${newLeft}px`;
  const target = document.elementFromPoint(event.clientX, event.clientY);
  const laneEl = target && target.closest ? target.closest(".lane") : null;
  if (laneEl && laneEl.dataset.lane) {
    const laneIdx = Number(laneEl.dataset.lane);
    if (laneIdx >= 1) {
      dragState.clip.dataset.lane = String(laneIdx);
      laneEl.appendChild(dragState.clip);
    }
  }
}

async function onPointerUp() {
  if (!dragState) return;
  if (dragState.mode === "trim-left" || dragState.mode === "trim-right") {
    await onTrimPointerUp();
    return;
  }
  dragState.clip.style.cursor = "grab";
  const newStartS = quantizeSeconds(pxToSec(parseInt(dragState.clip.style.left, 10) || 0));
  const clipId = dragState.clipId;
  const newLane = Number(dragState.clip.dataset.lane || dragState.laneStartIdx || 1);
  dragState = null;
  await patchClip(clipId, {
    timeline_start_s: Math.round(newStartS * 1000) / 1000,
    lane_idx: Math.max(1, Math.min(256, newLane)),
  });
}

function onTrimHandlePointerDown(event, clip, side) {
  event.stopPropagation();
  const clipEl = event.currentTarget.closest(".clip");
  if (!clipEl) return;
  const clipId = Number(clipEl.dataset.clipId);
  selectedClipId = clipId;
  syncSelectedClass();
  const timelineStart = Number(clip.timeline_start_s);
  const timelineDur = clipDurationTimelineSeconds(clip);
  dragState = {
    mode: side === "left" ? "trim-left" : "trim-right",
    clip: clipEl,
    clipId,
    pointerStartX: event.clientX,
    initialTimelineStart: timelineStart,
    initialTimelineEnd: timelineStart + timelineDur,
    initialSrcStart: Number(clip.src_start_s),
    initialSrcEnd: Number(clip.src_end_s),
    tempoRatio: Number(clip.tempo_ratio || 1),
  };
  clipEl.setPointerCapture(event.pointerId);
}

function onTrimPointerMove(event) {
  if (!dragState) return;
  const dx = event.clientX - dragState.pointerStartX;
  const deltaSec = quantizeSeconds(pxToSec(dx));
  const ratio = dragState.tempoRatio || 1;
  const minSrcDur = 0.1;
  const minTimelineDur = minSrcDur / ratio;
  let nextStart = dragState.initialTimelineStart;
  let nextEnd = dragState.initialTimelineEnd;
  if (dragState.mode === "trim-left") {
    const maxExpandLeft = dragState.initialSrcStart / ratio;
    const maxShrinkLeft = (dragState.initialSrcEnd - dragState.initialSrcStart) / ratio - minTimelineDur;
    const bounded = Math.max(-maxExpandLeft, Math.min(maxShrinkLeft, deltaSec));
    nextStart = dragState.initialTimelineStart + bounded;
  } else {
    const maxShrinkRight = (dragState.initialSrcEnd - dragState.initialSrcStart) / ratio - minTimelineDur;
    const bounded = Math.max(-maxShrinkRight, deltaSec);
    nextEnd = dragState.initialTimelineEnd + bounded;
  }
  dragState.previewTimelineStart = nextStart;
  dragState.previewTimelineEnd = Math.max(nextStart + minTimelineDur, nextEnd);
  dragState.clip.style.left = `${secToPx(dragState.previewTimelineStart)}px`;
  dragState.clip.style.width = `${secToPx(dragState.previewTimelineEnd - dragState.previewTimelineStart)}px`;
}

async function onTrimPointerUp() {
  if (!dragState) return;
  const ratio = dragState.tempoRatio || 1;
  const finalStart = dragState.previewTimelineStart ?? dragState.initialTimelineStart;
  const finalEnd = dragState.previewTimelineEnd ?? dragState.initialTimelineEnd;
  const body = {};
  if (dragState.mode === "trim-left") {
    const timelineShift = finalStart - dragState.initialTimelineStart;
    body.timeline_start_s = quantizeSeconds(finalStart);
    body.src_start_s = Math.max(0, Math.round((dragState.initialSrcStart + timelineShift * ratio) * 1000) / 1000);
    body.src_end_s = dragState.initialSrcEnd;
  } else {
    const timelineExtend = finalEnd - dragState.initialTimelineEnd;
    body.src_start_s = dragState.initialSrcStart;
    body.src_end_s = Math.max(
      dragState.initialSrcStart + 0.1,
      Math.round((dragState.initialSrcEnd + timelineExtend * ratio) * 1000) / 1000,
    );
  }
  const clipId = dragState.clipId;
  dragState = null;
  await patchClip(clipId, body);
}

async function patchClip(clipId, body) {
  const resp = await fetch(`${apiBase()}/clips/${clipId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    alert(`Failed to update clip: ${resp.status}`);
    return;
  }
  await loadProjectClips();
}

async function splitSelectedClip() {
  if (!selectedClipId) return;
  const clipEl = arrangerEl.querySelector(`.clip[data-clip-id="${selectedClipId}"]`);
  if (!clipEl) return;
  const clip = await getSelectedClip();
  if (!clip) return;
  const splitSrcMid = (Number(clip.src_start_s) + Number(clip.src_end_s)) / 2;
  const splitSrc = await snapToNearestPoint(clip.track_id, splitSrcMid, 2.0);
  const resp = await fetch(`${apiBase()}/clips/${selectedClipId}/split?split_at_src_s=${encodeURIComponent(splitSrc)}`, {
    method: "POST",
  });
  if (!resp.ok) {
    alert(`Failed to split clip: ${resp.status}`);
    return;
  }
  await loadProjectClips();
}

async function trimSelectedLeft() {
  await trimSelected(1, 0);
}

async function trimSelectedRight() {
  await trimSelected(0, -1);
}

async function trimSelected(deltaStartS, deltaEndS) {
  const clip = await getSelectedClip();
  if (!clip) return;
  let srcStart = Math.max(0, Number(clip.src_start_s) + deltaStartS);
  let srcEnd = Math.max(srcStart + 0.1, Number(clip.src_end_s) + deltaEndS);
  if (deltaStartS !== 0) {
    srcStart = await snapToDirectionalPoint(clip.track_id, srcStart, "next", Number(clip.src_start_s));
  }
  if (deltaEndS !== 0) {
    srcEnd = await snapToDirectionalPoint(clip.track_id, srcEnd, "prev", Number(clip.src_end_s));
  }
  srcEnd = Math.max(srcStart + 0.1, srcEnd);
  await patchClip(selectedClipId, { src_start_s: srcStart, src_end_s: srcEnd });
}

async function deleteSelectedClip() {
  if (!selectedClipId) return;
  const resp = await fetch(`${apiBase()}/clips/${selectedClipId}`, { method: "DELETE" });
  if (!resp.ok) {
    alert(`Failed to delete clip: ${resp.status}`);
    return;
  }
  selectedClipId = null;
  syncVolumeUiForSelection();
  await loadProjectClips();
}

function onGlobalKeyDown(event) {
  const target = event.target;
  const tag = target && target.tagName ? target.tagName.toLowerCase() : "";
  const isTypingField = tag === "input" || tag === "textarea" || tag === "select";
  if (isTypingField) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (transportIsRunning) {
      pauseTransport();
    } else {
      startTransport();
    }
    return;
  }
  if ((event.key === "Delete" || event.key === "Backspace") && selectedClipId) {
    event.preventDefault();
    deleteSelectedClip().catch((err) => {
      console.error(err);
    });
  }
}

async function moveSelectedClipLane(delta) {
  const clip = await getSelectedClip();
  if (!clip) return;
  const nextLane = Math.max(1, Number(clip.lane_idx) + delta);
  laneCount = Math.max(laneCount, nextLane);
  renderLanes();
  await patchClip(selectedClipId, { lane_idx: nextLane });
}

async function getSelectedClip() {
  if (!selectedClipId) return null;
  const projectId = selectedProjectId();
  const clipsResp = await fetch(`${apiBase()}/projects/${projectId}/clips`);
  if (!clipsResp.ok) return null;
  const clipsPayload = await clipsResp.json();
  return (clipsPayload.items || []).find((c) => Number(c.clip_id) === Number(selectedClipId)) || null;
}

async function fetchTrackSnapPoints(trackId) {
  const resp = await fetch(`${apiBase()}/tracks/${encodeURIComponent(trackId)}/analysis`);
  if (!resp.ok) return [];
  const payload = await resp.json();
  const points = [...(payload.measure_times || []), ...(payload.cue_points || [])]
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v) && v >= 0);
  return Array.from(new Set(points)).sort((a, b) => a - b);
}

async function snapToNearestPoint(trackId, valueS, maxDistanceS) {
  const points = await fetchTrackSnapPoints(trackId);
  if (points.length === 0) return valueS;
  let best = valueS;
  let bestDist = Number.POSITIVE_INFINITY;
  points.forEach((p) => {
    const d = Math.abs(p - valueS);
    if (d < bestDist) {
      bestDist = d;
      best = p;
    }
  });
  return bestDist <= maxDistanceS ? best : valueS;
}

async function snapToDirectionalPoint(trackId, fallbackValueS, direction, anchorValueS) {
  const points = await fetchTrackSnapPoints(trackId);
  if (points.length === 0) return fallbackValueS;
  if (direction === "next") {
    const next = points.find((p) => p > anchorValueS + 0.001);
    return next !== undefined ? next : fallbackValueS;
  }
  if (direction === "prev") {
    for (let i = points.length - 1; i >= 0; i -= 1) {
      if (points[i] < anchorValueS - 0.001) return points[i];
    }
  }
  return fallbackValueS;
}

function renderPlayhead() {
  playheadEl.style.left = `${secToPx(playheadSeconds)}px`;
  renderLoopRegion();
}

function currentLoopBounds() {
  const loopStart = Math.max(0, Number(loopStartInput.value || 0));
  const loopEndRaw = Math.max(0, Number(loopEndInput.value || 0));
  const loopEnd = Math.max(loopStart + 0.01, loopEndRaw);
  return { loopStart, loopEnd };
}

function renderLoopRegion() {
  const { loopStart, loopEnd } = currentLoopBounds();
  const left = secToPx(loopStart);
  const width = Math.max(0, secToPx(loopEnd - loopStart));
  loopRegionEl.style.left = `${left}px`;
  loopRegionEl.style.width = `${width}px`;
  loopRegionEl.style.display = loopEnabledCheckbox.checked ? "block" : "none";
}

function startTransport() {
  if (transportIsRunning) return;
  transportIsRunning = true;
  stopAllAudioSources();
  playAudioFromPlayhead().catch((err) => {
    console.error(err);
    alert(`Audio playback failed: ${err.message || err}`);
  });
  let last = performance.now();
  transportTimer = window.setInterval(() => {
    const now = performance.now();
    const dt = (now - last) / 1000;
    last = now;
    playheadSeconds += dt;
    if (loopEnabledCheckbox.checked) {
      const { loopStart, loopEnd } = currentLoopBounds();
      if (playheadSeconds >= loopEnd) {
        playheadSeconds = loopStart;
        stopAllAudioSources();
        playAudioFromPlayhead().catch((err) => {
          console.error(err);
        });
      }
    }
    renderPlayhead();
  }, 16);
}

function stopTransport() {
  if (transportTimer) {
    window.clearInterval(transportTimer);
    transportTimer = null;
  }
  transportIsRunning = false;
  stopAllAudioSources();
}

function resetTransport() {
  stopTransport();
  playheadSeconds = 0;
  renderPlayhead();
}

function pauseTransport() {
  stopTransport();
}

function stopAllAudioSources() {
  activeSources.forEach((src) => {
    try {
      src.stop();
    } catch (_) {
      // ignore stop races
    }
  });
  activeSources = [];
}

async function ensureAudioContext() {
  if (!audioCtx) {
    audioCtx = new window.AudioContext();
  }
  if (audioCtx.state === "suspended") {
    await audioCtx.resume();
  }
  return audioCtx;
}

async function fetchDecodedTrackBuffer(trackId, variantTag) {
  const cacheKey = `${trackId}::${variantTag || "original"}`;
  if (decodedBufferCache.has(cacheKey)) {
    return decodedBufferCache.get(cacheKey);
  }
  if ((variantTag || "").toLowerCase() === "instrumental" || (variantTag || "").toLowerCase() === "acappella") {
    const derived = await buildDerivedVariantBuffer(trackId, variantTag || "original");
    if (derived) {
      decodedBufferCache.set(cacheKey, derived);
      return derived;
    }
  }
  const ctx = await ensureAudioContext();
  const resp = await fetch(
    `${apiBase()}/tracks/${encodeURIComponent(trackId)}/audio?variant_tag=${encodeURIComponent(variantTag || "original")}`,
  );
  if (!resp.ok) {
    throw new Error(`Failed audio fetch for ${trackId}: HTTP ${resp.status}`);
  }
  const arr = await resp.arrayBuffer();
  const buf = await ctx.decodeAudioData(arr.slice(0));
  decodedBufferCache.set(cacheKey, buf);
  return buf;
}

async function fetchStemManifest(trackId) {
  if (stemManifestCache.has(trackId)) {
    return stemManifestCache.get(trackId);
  }
  const resp = await fetch(`${apiBase()}/tracks/${encodeURIComponent(trackId)}/stems`);
  if (!resp.ok) {
    return [];
  }
  const payload = await resp.json();
  const items = payload.items || [];
  stemManifestCache.set(trackId, items);
  return items;
}

async function fetchDecodedStemBuffer(trackId, stemName) {
  const cacheKey = `${trackId}::stem::${stemName}`;
  if (decodedBufferCache.has(cacheKey)) {
    return decodedBufferCache.get(cacheKey);
  }
  const items = await fetchStemManifest(trackId);
  const stem = items.find((item) => item.stem_name === stemName && item.exists);
  if (!stem) return null;
  const ctx = await ensureAudioContext();
  const url = `${apiBase()}${stem.url}`;
  const resp = await fetch(url);
  if (!resp.ok) return null;
  const arr = await resp.arrayBuffer();
  const buf = await ctx.decodeAudioData(arr.slice(0));
  decodedBufferCache.set(cacheKey, buf);
  return buf;
}

async function buildDerivedVariantBuffer(trackId, variantTag) {
  const ctx = await ensureAudioContext();
  const variant = (variantTag || "").toLowerCase();
  if (variant === "acappella") {
    return fetchDecodedStemBuffer(trackId, "vocals");
  }
  if (variant !== "instrumental") {
    return null;
  }
  const drumBuf = await fetchDecodedStemBuffer(trackId, "drums");
  const bassBuf = await fetchDecodedStemBuffer(trackId, "bass");
  const otherBuf = await fetchDecodedStemBuffer(trackId, "other");
  const parts = [drumBuf, bassBuf, otherBuf].filter(Boolean);
  if (parts.length === 0) return null;

  const sampleRate = parts[0].sampleRate;
  const channels = Math.max(...parts.map((b) => b.numberOfChannels));
  const length = Math.max(...parts.map((b) => b.length));
  const out = ctx.createBuffer(channels, length, sampleRate);

  for (let ch = 0; ch < channels; ch += 1) {
    const outData = out.getChannelData(ch);
    for (const part of parts) {
      const src = part.getChannelData(Math.min(ch, part.numberOfChannels - 1));
      const gain = 1 / parts.length;
      const n = Math.min(src.length, outData.length);
      for (let i = 0; i < n; i += 1) {
        outData[i] += src[i] * gain;
      }
    }
  }
  return out;
}

function clipVolumeGainForId(clipId) {
  const v = clipVolumeById.get(Number(clipId));
  if (v === undefined) return 0.72;
  return Math.max(0, Math.min(1.2, Number(v)));
}

function syncVolumeUiForSelection() {
  if (!selectedClipId) {
    clipVolumeSlider.value = "0.72";
    clipVolumeValue.textContent = "72%";
    return;
  }
  const v = clipVolumeGainForId(selectedClipId);
  clipVolumeSlider.value = String(v);
  clipVolumeValue.textContent = `${Math.round(v * 100)}%`;
}

async function renderClipWaveform(clipEl, clip) {
  const canvas = clipEl.querySelector(".waveform");
  if (!canvas) return;
  const buffer = await fetchDecodedTrackBuffer(clip.track_id, clip.variant_tag);
  const w = Math.max(80, Math.floor(secToPx(clipDurationTimelineSeconds(clip))));
  const h = 44;
  canvas.width = w;
  canvas.height = h;
  const ctx2d = canvas.getContext("2d");
  if (!ctx2d) return;
  ctx2d.clearRect(0, 0, w, h);
  ctx2d.fillStyle = "rgba(15, 30, 60, 0.4)";
  ctx2d.fillRect(0, 0, w, h);
  ctx2d.strokeStyle = "rgba(180, 220, 255, 0.85)";
  ctx2d.lineWidth = 1;
  const data = buffer.getChannelData(0);
  const srcStart = Math.max(0, Math.floor(Number(clip.src_start_s) * buffer.sampleRate));
  const srcEnd = Math.min(data.length, Math.floor(Number(clip.src_end_s) * buffer.sampleRate));
  const samples = Math.max(1, srcEnd - srcStart);
  const step = Math.max(1, Math.floor(samples / w));
  ctx2d.beginPath();
  for (let x = 0; x < w; x += 1) {
    const start = srcStart + x * step;
    const end = Math.min(srcEnd, start + step);
    let peak = 0;
    for (let i = start; i < end; i += 1) {
      const a = Math.abs(data[i] || 0);
      if (a > peak) peak = a;
    }
    const y = (1 - peak) * (h / 2);
    ctx2d.moveTo(x + 0.5, y);
    ctx2d.lineTo(x + 0.5, h - y);
  }
  ctx2d.stroke();
}

async function playAudioFromPlayhead() {
  const ctx = await ensureAudioContext();
  const baseWhen = ctx.currentTime + 0.05;
  const nowPlayhead = playheadSeconds;

  const clipsToStart = projectClips
    .map((clip) => ({ clip, startTimeline: Number(clip.timeline_start_s) }))
    .filter(({ clip, startTimeline }) => {
      const endTimeline = startTimeline + clipDurationTimelineSeconds(clip);
      return endTimeline > nowPlayhead;
    });

  for (const { clip, startTimeline } of clipsToStart) {
    const buffer = await fetchDecodedTrackBuffer(clip.track_id, clip.variant_tag);
    const ratio = Number(clip.tempo_ratio || 1);
    const clipSrcStart = Number(clip.src_start_s);
    const clipSrcEnd = Number(clip.src_end_s);
    const clipTimelineDuration = (clipSrcEnd - clipSrcStart) / ratio;
    const playOffsetTimeline = Math.max(0, nowPlayhead - startTimeline);
    const sourceOffsetInTrack = clipSrcStart + playOffsetTimeline * ratio;
    const remainTimeline = Math.max(0, clipTimelineDuration - playOffsetTimeline);
    const sourceDuration = remainTimeline * ratio;
    if (sourceDuration <= 0.01) continue;
    if (sourceOffsetInTrack >= buffer.duration) continue;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.playbackRate.value = ratio;
    const gainNode = ctx.createGain();
    gainNode.gain.value = clipVolumeGainForId(clip.clip_id);
    source.connect(gainNode);
    gainNode.connect(ctx.destination);

    const startWhen = baseWhen + Math.max(0, startTimeline - nowPlayhead);
    const safeDuration = Math.min(sourceDuration, Math.max(0, buffer.duration - sourceOffsetInTrack));
    try {
      source.start(startWhen, sourceOffsetInTrack, safeDuration);
      activeSources.push(source);
      source.onended = () => {
        activeSources = activeSources.filter((s) => s !== source);
      };
    } catch (err) {
      console.warn("clip start failed", clip.clip_id, err);
    }
  }
}

function seekPlayheadFromPointerEvent(event) {
  const targetIsClip = event.target && event.target.closest && event.target.closest(".clip");
  if (targetIsClip) return;
  const rect = arrangerEl.getBoundingClientRect();
  const x = Math.max(0, event.clientX - rect.left + arrangerEl.scrollLeft);
  playheadSeconds = quantizeSeconds(pxToSec(x));
  renderPlayhead();
  if (transportIsRunning) {
    stopAllAudioSources();
    playAudioFromPlayhead().catch((err) => {
      console.error(err);
    });
  }
}

function onArrangerWheel(event) {
  if (event.ctrlKey || event.metaKey) {
    event.preventDefault();
    const zoomFactor = event.deltaY > 0 ? 0.0065 : 0.0045; // zoom out faster per swipe
    const zoomDelta = -event.deltaY * zoomFactor;
    setZoom(zoomLevel + zoomDelta, event.clientX);
    return;
  }
  // Two-finger swipe over arranger should always scrub timeline horizontally.
  const dx = event.deltaX;
  const dy = event.deltaY;
  if (Math.abs(dx) > 0 || Math.abs(dy) > 0) {
    event.preventDefault();
    event.stopPropagation();
    const horizontal = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
    arrangerEl.scrollLeft += horizontal;
  }
}

function onGlobalWheel(event) {
  if (!isPointerOverArranger) return;
  if (event.ctrlKey || event.metaKey) return;
  const targetInsideTrackBrowser = event.target && event.target.closest && event.target.closest("#trackBrowserScroll");
  if (targetInsideTrackBrowser) return;
  const dx = event.deltaX;
  const dy = event.deltaY;
  if (Math.abs(dx) > 0 || Math.abs(dy) > 0) {
    event.preventDefault();
    const horizontal = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
    arrangerEl.scrollLeft += horizontal;
  }
}

function onGlobalMouseWheel(event) {
  if (!isPointerOverArranger) return;
  const targetInsideTrackBrowser = event.target && event.target.closest && event.target.closest("#trackBrowserScroll");
  if (targetInsideTrackBrowser) return;
  const dx = event.wheelDeltaX ? -event.wheelDeltaX : 0;
  const dy = event.wheelDeltaY ? -event.wheelDeltaY : -event.wheelDelta;
  if (Math.abs(dx) > 0 || Math.abs(dy) > 0) {
    event.preventDefault();
    const horizontal = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
    arrangerEl.scrollLeft += horizontal;
  }
}

function onTrackBrowserWheel(event) {
  const dx = event.deltaX;
  const dy = event.deltaY;
  if (Math.abs(dx) > 0 || Math.abs(dy) > 0) {
    event.preventDefault();
    trackBrowserScrollEl.scrollTop += dy;
    trackBrowserScrollEl.scrollLeft += dx;
  }
}

function refreshClipGeometryFromState() {
  const byId = new Map(projectClips.map((c) => [Number(c.clip_id), c]));
  arrangerEl.querySelectorAll(".clip").forEach((el) => {
    const clip = byId.get(Number(el.dataset.clipId));
    if (!clip) return;
    el.style.left = `${secToPx(Number(clip.timeline_start_s))}px`;
    el.style.width = `${secToPx(clipDurationTimelineSeconds(clip))}px`;
  });
}

function setZoom(nextZoom, anchorClientX = null) {
  const prevPps = pxPerSec();
  zoomLevel = Math.max(0.6, Math.min(4, Number(nextZoom) || 1));
  zoomSlider.value = String(zoomLevel);
  zoomValue.textContent = `${Math.round(zoomLevel * 100)}%`;
  if (anchorClientX !== null) {
    const rect = arrangerEl.getBoundingClientRect();
    const anchorXInView = Math.max(0, anchorClientX - rect.left);
    const secAtAnchor = (arrangerEl.scrollLeft + anchorXInView) / prevPps;
    const newScrollLeft = secAtAnchor * pxPerSec() - anchorXInView;
    arrangerEl.scrollLeft = Math.max(0, newScrollLeft);
  }
  applyTimelineWidth();
  renderTimeMarkers();
  refreshClipGeometryFromState();
  renderPlayhead();
}

loadTracksBtn.addEventListener("click", loadTracks);
saveProjectBtn.addEventListener("click", saveProjectSettings);
projectSelect.addEventListener("change", loadProjectClips);
splitClipBtn.addEventListener("click", splitSelectedClip);
trimLeftBtn.addEventListener("click", trimSelectedLeft);
trimRightBtn.addEventListener("click", trimSelectedRight);
deleteClipBtn.addEventListener("click", deleteSelectedClip);
moveClipUpBtn.addEventListener("click", () => moveSelectedClipLane(-1));
moveClipDownBtn.addEventListener("click", () => moveSelectedClipLane(1));
playBtn.addEventListener("click", startTransport);
pauseBtn.addEventListener("click", pauseTransport);
stopBtn.addEventListener("click", resetTransport);
arrangerEl.addEventListener("pointerdown", seekPlayheadFromPointerEvent);
arrangerEl.addEventListener("wheel", onArrangerWheel, { passive: false });
arrangerEl.addEventListener("mouseenter", () => { isPointerOverArranger = true; });
arrangerEl.addEventListener("mouseleave", () => { isPointerOverArranger = false; });
loopEnabledCheckbox.addEventListener("change", renderLoopRegion);
loopStartInput.addEventListener("input", () => {
  renderLoopRegion();
});
loopEndInput.addEventListener("input", () => {
  renderLoopRegion();
});
setLoopFromPlayheadBtn.addEventListener("click", () => {
  loopStartInput.value = String(Math.max(0, Math.round(playheadSeconds * 100) / 100));
  const { loopStart, loopEnd } = currentLoopBounds();
  if (loopEnd <= loopStart) {
    loopEndInput.value = String(Math.round((loopStart + 4) * 100) / 100);
  }
  renderLoopRegion();
});
setLoopEndFromPlayheadBtn.addEventListener("click", () => {
  const { loopStart } = currentLoopBounds();
  const candidate = Math.max(loopStart + 0.1, playheadSeconds);
  loopEndInput.value = String(Math.round(candidate * 100) / 100);
  renderLoopRegion();
});
zoomSlider.addEventListener("input", () => setZoom(zoomSlider.value));
zoomOutBtn.addEventListener("click", () => setZoom(Math.round((zoomLevel - 0.2) * 10) / 10));
zoomInBtn.addEventListener("click", () => setZoom(Math.round((zoomLevel + 0.2) * 10) / 10));
addLaneBtn.addEventListener("click", () => {
  laneCount += 1;
  renderLanes();
  applyTimelineWidth();
  renderTimeMarkers();
});
themeSelect.addEventListener("change", () => {
  const theme = themeSelect.value || "dark";
  document.body.classList.remove("theme-dark", "theme-light", "theme-blue", "theme-sunset");
  document.body.classList.add(`theme-${theme}`);
});
clipVolumeSlider.addEventListener("input", () => {
  const v = Number(clipVolumeSlider.value);
  clipVolumeValue.textContent = `${Math.round(v * 100)}%`;
  if (selectedClipId) {
    clipVolumeById.set(Number(selectedClipId), v);
  }
});
trackSearchInput.addEventListener("input", applyTrackSearchFilter);
trackBrowserScrollEl.addEventListener("wheel", onTrackBrowserWheel, { passive: false });
window.addEventListener("pointermove", onPointerMove);
window.addEventListener("pointerup", onPointerUp);
window.addEventListener("keydown", onGlobalKeyDown);
document.addEventListener("wheel", onGlobalWheel, { passive: false, capture: true });
document.addEventListener("mousewheel", onGlobalMouseWheel, { passive: false, capture: true });

async function boot() {
  const healthResp = await fetch(`${apiBase()}/health`);
  if (!healthResp.ok) {
    alert("Backend is not reachable. Start uvicorn first.");
    return;
  }
  document.body.classList.add("theme-dark");
  setZoom(1);
  renderLanes();
  await loadProjects();
  await loadTracks();
  applyTimelineWidth();
  renderTimeMarkers();
  renderPlayhead();
  renderLoopRegion();
  syncVolumeUiForSelection();
}

boot();
