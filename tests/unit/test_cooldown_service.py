"""Unit tests for the cooldown service contract."""

from datetime import datetime, timedelta

from skynet_local.application.services.cooldown_service import CooldownService


def test_cooldown_allows_after_interval() -> None:
    """Verify that a cooldown key becomes available again after the configured interval."""
    service = CooldownService()
    now = datetime(2026, 1, 1, 12, 0, 0)
    assert service.allowed("hello:user", 10, now=now) is True
    assert service.allowed("hello:user", 10, now=now + timedelta(seconds=5)) is False
    assert service.allowed("hello:user", 10, now=now + timedelta(seconds=11)) is True
