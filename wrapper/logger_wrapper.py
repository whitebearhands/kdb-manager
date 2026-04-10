import logging
import logging.handlers
import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LogConfig(Protocol):
    level: str
    format: str
    output_target: str
    dir: str
    filename: str
    file_max_bytes: int
    file_backup_count: int


def _build_formatter(fmt: str) -> logging.Formatter:
    if fmt == "json":
        try:
            from pythonjsonlogger.json import JsonFormatter
            return JsonFormatter(
                fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        except ImportError:
            pass
    return logging.Formatter(
        fmt="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logger(config: LogConfig) -> None:
    """
    루트 로거를 설정합니다.

    Parameters
    ----------
    config : LogConfig
        level, format, output_target, dir, filename,
        file_max_bytes, file_backup_count 속성을 가진 객체
    """
    level = getattr(logging, config.level.upper(), logging.INFO)
    formatter = _build_formatter(config.format)

    root = logging.getLogger()
    # 기존 핸들러 제거 (중복 등록 방지)
    root.handlers.clear()
    root.setLevel(level)

    if config.output_target in ("console", "both"):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    if config.output_target in ("file", "both"):
        os.makedirs(config.dir, exist_ok=True)
        path = os.path.join(config.dir, config.filename)
        rotating = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=config.file_max_bytes,
            backupCount=config.file_backup_count,
            encoding="utf-8",
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)
