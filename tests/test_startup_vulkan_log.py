import logging
from unittest.mock import MagicMock, patch

import pytest

from speaches.config import Config
from speaches.main import log_vulkan_summary_if_enabled


def test_skips_when_backend_is_faster_whisper(caplog: pytest.LogCaptureFixture) -> None:
    cfg = Config(whisper_backend="faster_whisper")
    with caplog.at_level(logging.INFO, logger="speaches.main"):
        log_vulkan_summary_if_enabled(cfg)
    assert "Vulkan" not in caplog.text


def test_runs_vulkaninfo_when_backend_whisper_cpp(caplog: pytest.LogCaptureFixture) -> None:
    cfg = Config(whisper_backend="whisper_cpp")
    fake_result = MagicMock()
    fake_result.stdout = "Vulkan Instance Version: 1.3.0\n"
    fake_result.returncode = 0
    with (
        patch("speaches.main.subprocess.run", return_value=fake_result) as run,
        caplog.at_level(logging.INFO, logger="speaches.main"),
    ):
        log_vulkan_summary_if_enabled(cfg)
    run.assert_called_once()
    assert "Vulkan Instance Version" in caplog.text


def test_handles_missing_vulkaninfo(caplog: pytest.LogCaptureFixture) -> None:
    cfg = Config(whisper_backend="whisper_cpp")
    with (
        patch("speaches.main.subprocess.run", side_effect=FileNotFoundError),
        caplog.at_level(logging.WARNING, logger="speaches.main"),
    ):
        log_vulkan_summary_if_enabled(cfg)
    assert "vulkaninfo" in caplog.text.lower()


def test_handles_nonzero_exit(caplog: pytest.LogCaptureFixture) -> None:
    cfg = Config(whisper_backend="whisper_cpp")
    fake_result = MagicMock()
    fake_result.stdout = ""
    fake_result.returncode = 1
    with (
        patch("speaches.main.subprocess.run", return_value=fake_result),
        caplog.at_level(logging.WARNING, logger="speaches.main"),
    ):
        log_vulkan_summary_if_enabled(cfg)
    assert "status" in caplog.text.lower() or "exit" in caplog.text.lower()
