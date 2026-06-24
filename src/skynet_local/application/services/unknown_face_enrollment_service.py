from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic


class UnknownTrackState(str, Enum):
    TRACKING = "tracking"
    PROMPTING = "prompting"
    ENROLLING = "enrolling"
    IGNORED = "ignored"
    ENROLLED = "enrolled"


@dataclass
class UnknownTrackSample:
    frame: object
    face_row: object
    quality: float


@dataclass
class UnknownTrackSession:
    track_id: str
    first_seen_at: float
    last_seen_at: float
    state: UnknownTrackState = UnknownTrackState.TRACKING
    prompted_at: float | None = None
    ignored_until: float | None = None
    samples: list[UnknownTrackSample] = field(default_factory=list)
    auto_person_id: str | None = None
    auto_display_name: str | None = None


class UnknownFaceEnrollmentService:
    def __init__(
        self,
        recognition_service,
        detector,
        min_dwell_seconds: float = 2.5,
        ignore_cooldown_seconds: float = 20.0,
        min_face_width: int = 120,
        samples_to_enroll: int = 5,
    ) -> None:
        self.recognition_service = recognition_service
        self.detector = detector
        self.min_dwell_seconds = min_dwell_seconds
        self.ignore_cooldown_seconds = ignore_cooldown_seconds
        self.min_face_width = min_face_width
        self.samples_to_enroll = samples_to_enroll
        self.sessions: dict[str, UnknownTrackSession] = {}
        self.pending_prompt_track_id: str | None = None
        self._person_seq = 1

    def update(self, frame, faces, key: int | None = None) -> str | None:
        now = monotonic()
        active_ids = set()
        self.pending_prompt_track_id = None

        for face in faces:
            active_ids.add(face.track_id)

            if face.label != "unknown":
                self.sessions.pop(face.track_id, None)
                continue

            x1, y1, x2, y2 = face.bbox
            face_width = x2 - x1
            if face_width < self.min_face_width:
                continue

            raw_face_row = self.detector.get_raw_face_row(face.track_id)
            if raw_face_row is None:
                continue

            session = self.sessions.get(face.track_id)
            if session is None:
                session = UnknownTrackSession(
                    track_id=face.track_id,
                    first_seen_at=now,
                    last_seen_at=now,
                )
                self.sessions[face.track_id] = session
            else:
                session.last_seen_at = now

            if session.ignored_until is not None and now < session.ignored_until:
                session.state = UnknownTrackState.IGNORED
                continue

            dwell = now - session.first_seen_at
            if dwell >= self.min_dwell_seconds and session.state == UnknownTrackState.TRACKING:
                session.state = UnknownTrackState.PROMPTING
                session.prompted_at = now

            if session.state == UnknownTrackState.PROMPTING:
                self.pending_prompt_track_id = face.track_id

                if key is not None and key == ord("i"):
                    session.state = UnknownTrackState.IGNORED
                    session.ignored_until = now + self.ignore_cooldown_seconds
                    continue

                if key is not None and key == ord("a"):
                    session.state = UnknownTrackState.ENROLLING
                    session.auto_person_id = self._build_person_id()
                    session.auto_display_name = session.auto_person_id

            if session.state == UnknownTrackState.ENROLLING:
                quality = self._estimate_quality(face)
                session.samples.append(
                    UnknownTrackSample(
                        frame=frame.copy(),
                        face_row=raw_face_row.copy() if hasattr(raw_face_row, "copy") else raw_face_row,
                        quality=quality,
                    )
                )
                session.samples.sort(key=lambda s: s.quality, reverse=True)
                session.samples = session.samples[: self.samples_to_enroll]

                if len(session.samples) >= self.samples_to_enroll:
                    self._finalize_enrollment(session)
                    session.state = UnknownTrackState.ENROLLED

        stale_ids = set(self.sessions.keys()) - active_ids
        for stale_id in stale_ids:
            session = self.sessions[stale_id]
            if session.state not in {UnknownTrackState.IGNORED, UnknownTrackState.ENROLLED}:
                del self.sessions[stale_id]

        return self.pending_prompt_track_id

    def get_prompt_text(self) -> str | None:
        if self.pending_prompt_track_id is None:
            return None
        return "Unknown face detected. Press A to add or I to ignore."

    def _finalize_enrollment(self, session: UnknownTrackSession) -> None:
        if not session.auto_person_id or not session.auto_display_name:
            return

        for sample in session.samples:
            self.recognition_service.enroll_detection(
                person_id=session.auto_person_id,
                display_name=session.auto_display_name,
                frame=sample.frame,
                face_row=sample.face_row,
                quality=sample.quality,
            )

    def _build_person_id(self) -> str:
        person_id = f"person_{self._person_seq:03d}"
        self._person_seq += 1
        return person_id

    @staticmethod
    def _estimate_quality(face) -> float:
        x1, y1, x2, y2 = face.bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        return float(area)
