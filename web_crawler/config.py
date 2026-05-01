from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, Generic, Literal
import yaml

# --- Type Variables ---
T = TypeVar("T")
E = TypeVar("E")


# --- Functional Core: Result Monad ---
@dataclass(frozen=True)
class Result(Generic[T, E]):
    """
        Wrapper for returning success or failure. Meant to emulate GoLang's error propogation
        and based on Category Theory.
    """
    value: T | None = None
    error: E | None = None
    is_success: bool = True

    @staticmethod
    def success(value: T) -> Result[T, E]:
        return Result(value=value, is_success=True)

    @staticmethod
    def fail(error: E) -> Result[T, E]:
        return Result(error=error, is_success=False)


@dataclass(frozen=True)
class PathsConfig:
    artist_set_jsons_dir: Path
    db_path: Path
    schema_path: Path
    log_dir: Path
    captcha_imgs_dir: Path


@dataclass(frozen=True)
class SelectionFilters:
    dj_files: list[str] = field(default_factory=list)
    title_contains: list[str] = field(default_factory=list)
    set_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratorConfig:
    testing: bool
    filters: SelectionFilters
    limit: int
    order: Literal["random", "asc", "desc"]
    seed: int
    fill: bool
    log_sets: bool
    log_set_dir: Path


@dataclass(frozen=True)
class ExecutionConfig:
    restart_every_n_sets: int
    log_every_n_sets: int


@dataclass(frozen=True)
class ProfilesConfig:
    base_dir: Path
    num_profiles: int
    retire_after_n_sites: int


@dataclass(frozen=True)
class TimingConfig:
    crawl_delay_s: float
    random_jitter_s: float


@dataclass(frozen=True)
class ViewportConfig:
    width: int
    height: int


@dataclass(frozen=True)
class BrowserConfig:
    headless: bool
    locale: str
    timezone: str
    viewport: ViewportConfig
    user_agent: str
    args: list[str]
    nav_timeout_ms: int = 60_000
    selector_timeout_ms: int = 10_000


@dataclass(frozen=True)
class FailureConfig:
    fail_fast: bool
    fail_dir: Path
    ajax_failure: Literal["continue", "kill", "wait"]
    ajax_wait_s: int
    kill_process_after_consecutive_failures: int


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int
    retry_delay_s: float
    retry_jitter_s: float


@dataclass(frozen=True)
class CaptchaConfig:
    # mode controls behavior on captcha hits:
    #   ocr       — local ddddocr OCR (no network) up to solver_max_attempts
    #   continue  — try OCR first; if it fails, skip the set and continue
    #   wait      — try OCR first; if it fails, fall back to email solver
    #   kill      — abort the whole run on any captcha
    mode: Literal["ocr", "kill", "continue", "wait"]
    captcha_wait_s: int
    solver_max_attempts: int


# --- Root Configuration ---

@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    generator: GeneratorConfig
    execution: ExecutionConfig
    profiles: ProfilesConfig
    timing: TimingConfig
    browser: BrowserConfig
    failure: FailureConfig
    retry: RetryConfig
    captcha: CaptchaConfig

    @staticmethod
    def _resolve(root: Path, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (root / path).resolve()

    @classmethod
    def from_dict(cls, data: dict, project_root: Path) -> AppConfig:
        return cls(
            paths=PathsConfig(
                artist_set_jsons_dir=cls._resolve(project_root, data['paths']['artist_set_jsons_dir']),
                db_path=cls._resolve(project_root, data['paths']['db_path']),
                schema_path=cls._resolve(project_root, data['paths']['schema_path']),
                log_dir=cls._resolve(project_root, data['paths'].get('log_dir', './logs')),
                captcha_imgs_dir=cls._resolve(project_root, data['paths'].get('captcha_imgs_dir', './captcha_imgs'))
            ),
            generator=GeneratorConfig(
                testing=data['generator']['testing'],
                filters=SelectionFilters(**data['generator']['filters']),
                limit=data['generator']['limit'],
                order=data['generator']['order'],
                seed=data['generator']['seed'],
                fill=data['generator']['fill'],
                log_sets=data['generator'].get('log_sets', False),
                log_set_dir=cls._resolve(project_root, data['generator'].get('log_set_dir', './set_logs'))
            ),
            execution=ExecutionConfig(**data['execution']),
            profiles=ProfilesConfig(
                base_dir=cls._resolve(project_root, data['profiles']['base_dir']),
                num_profiles=data['profiles']['num_profiles'],
                retire_after_n_sites=data['profiles']['retire_after_n_sites']
            ),
            timing=TimingConfig(**data['timing']),
            browser=BrowserConfig(
                headless=data['browser']['headless'],
                locale=data['browser']['locale'],
                timezone=data['browser']['timezone'],
                viewport=ViewportConfig(**data['browser']['viewport']),
                user_agent=data['browser']['user_agent'],
                args=data['browser'].get('args', []),
                nav_timeout_ms=data['browser'].get('nav_timeout_ms', 60000),
                selector_timeout_ms=data['browser'].get('selector_timeout_ms', 10000)
            ),
            failure=FailureConfig(
                fail_fast=data['failure']['fail_fast'],
                fail_dir=cls._resolve(project_root, data['failure']['fail_dir']),
                ajax_failure=data['failure']['ajax_failure'],
                ajax_wait_s=data['failure']['ajax_wait_s'],
                kill_process_after_consecutive_failures=data['failure'].get('kill_process_after_consecutive_failures', 5),
            ),
            retry=RetryConfig(**data['retry']),
            captcha=CaptchaConfig(**data['captcha'])
        )


def load_config(config_path: Path, project_root: Path) -> AppConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, 'r') as f:
        raw_data = yaml.safe_load(f)

    return AppConfig.from_dict(raw_data, project_root)
