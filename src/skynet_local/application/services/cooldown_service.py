"""Time-based cooldown helper used by greetings, emotion reactions and alerts."""

from datetime import datetime, timedelta


class CooldownService:
    """Store last-trigger timestamps and decide whether an event may fire again."""

    def __init__(self) -> None:
        self._state: dict[str, datetime] = {}

    def allowed(self, key: str, cooldown_seconds: int, now: datetime | None = None) -> bool:
        """Return true when the cooldown window has expired for the given key."""
        now = now or datetime.utcnow()
        last = self._state.get(key)
        if last is None or now - last >= timedelta(seconds=cooldown_seconds):
            self._state[key] = now
            return True
        return False
