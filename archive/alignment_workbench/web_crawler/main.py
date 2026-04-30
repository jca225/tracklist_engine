from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

# Adjust path to ensure we can import modules from current directory
sys.path.append(str(Path(__file__).parent))

from archive.alignment_workbench.web_crawler.config import GeneratorConfig, load_config
from archive.alignment_workbench.web_crawler.database import MusicDatabase
from archive.alignment_workbench.web_crawler.logging_setup import setup_logging
from archive.alignment_workbench.web_crawler.workers import run_serial


def _matches_title(title: str, needles: list[str]) -> bool:
    if not needles:
        return True
    lowered = title.lower()
    return any(needle.lower() in lowered for needle in needles)


def load_jobs(artist_set_jsons_dir: Path, generator: GeneratorConfig) -> list[dict]:
    if not artist_set_jsons_dir.exists():
        return []

    if generator.testing:
        dj_files = generator.filters.dj_files or []
        if dj_files:
            json_files = [artist_set_jsons_dir / name for name in dj_files]
        else:
            json_files = sorted(artist_set_jsons_dir.glob("*.json"))
    else:
        json_files = sorted(artist_set_jsons_dir.glob("*.json"))

    allowed_ids = set(generator.filters.set_ids or []) if generator.testing else set()
    title_filters = (generator.filters.title_contains or []) if generator.testing else []

    rng = random.Random(generator.seed)
    required: list[dict] = []
    required_ids: set[str] = set()
    remaining_by_file: dict[str, list[dict]] = {}

    for jf in json_files:
        if not jf.exists():
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        records = data if isinstance(data, list) else [data]
        file_key = jf.name
        remaining_by_file.setdefault(file_key, [])

        for obj in records:
            tracklist_id = obj.get("tracklist_id")
            url = obj.get("url")
            if not tracklist_id or not url:
                continue

            title = obj.get("title") or ""
            if generator.testing:
                matches_id = bool(allowed_ids and tracklist_id in allowed_ids)
                matches_title = bool(title_filters and _matches_title(title, title_filters))

                if matches_id or matches_title:
                    if tracklist_id not in required_ids:
                        required.append(obj)
                        required_ids.add(tracklist_id)
                    continue

                remaining_by_file[file_key].append(obj)
            else:
                remaining_by_file[file_key].append(obj)

    for file_jobs in remaining_by_file.values():
        rng.shuffle(file_jobs)

    if generator.testing:
        jobs = list(required)
    else:
        jobs = []

    if generator.limit and generator.limit > 0:
        needed = generator.limit - len(jobs)
        if needed > 0:
            file_keys = [k for k, v in remaining_by_file.items() if v]
            rng.shuffle(file_keys)

            idx = 0
            while needed > 0 and file_keys:
                key = file_keys[idx % len(file_keys)]
                bucket = remaining_by_file.get(key, [])
                if not bucket:
                    file_keys = [k for k in file_keys if remaining_by_file.get(k)]
                    idx += 1
                    continue
                jobs.append(bucket.pop())
                needed -= 1
                idx += 1
    else:
        if not generator.testing:
            jobs = [job for bucket in remaining_by_file.values() for job in bucket]

    if generator.order == "random":
        rng.shuffle(jobs)
    elif generator.order == "desc":
        jobs = list(reversed(jobs))

    if generator.limit and generator.limit > 0:
        jobs = jobs[: generator.limit]

    return jobs


def main() -> None:
    current_src_dir = Path(__file__).resolve().parent
    project_root = current_src_dir.parent
    config_file_path = project_root / "config.yaml"

    if not config_file_path.exists():
        print(f"CRITICAL: Config file not found at {config_file_path}")
        return

    try:
        cfg = load_config(config_file_path, project_root)
    except Exception as e:
        print(f"CRITICAL: Could not load config: {e}")
        return

    setup_logging(cfg)
    log = logging.getLogger("Main")
    log.info("Starting Scraper Application...")

    db = MusicDatabase(str(cfg.paths.db_path), str(cfg.paths.schema_path))

    jobs = load_jobs(cfg.paths.artist_set_jsons_dir, cfg.generator)
    log.info(f"Loaded {len(jobs)} jobs. Running serial scraper...")

    try:
        run_serial(jobs, cfg, db)
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt received! Stopping serial run...")
    finally:
        db.close()
        log.info("Serial run finished. Exiting.")


if __name__ == "__main__":
    main()
