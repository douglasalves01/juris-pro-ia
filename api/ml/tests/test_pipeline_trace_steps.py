from __future__ import annotations

import time
from datetime import datetime

from hypothesis import given, settings, strategies as st

from api.ml.pipeline import AnalysisPipeline
from api.schemas.analysis import TraceStep


_STEP_NAMES = st.text(min_size=1, max_size=80).filter(lambda value: value.strip() != "")


@given(
    step_name=_STEP_NAMES,
    elapsed_seconds=st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_record_step_duration_ms_eh_sempre_nao_negativo(
    step_name: str,
    elapsed_seconds: float,
) -> None:
    pipeline = object.__new__(AnalysisPipeline)
    steps: list[dict[str, object]] = []
    started = time.perf_counter() - elapsed_seconds

    pipeline._record_step(steps, step_name, started)

    assert len(steps) == 1
    parsed = TraceStep.model_validate(steps[0])
    assert parsed.durationMs >= 0


@given(
    step_name=_STEP_NAMES,
    elapsed_seconds=st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_record_step_started_at_e_finished_at_sao_iso_utc(
    step_name: str,
    elapsed_seconds: float,
) -> None:
    pipeline = object.__new__(AnalysisPipeline)
    steps: list[dict[str, object]] = []
    started = time.perf_counter() - elapsed_seconds

    pipeline._record_step(steps, step_name, started)

    parsed = TraceStep.model_validate(steps[0])
    assert parsed.startedAt
    assert parsed.finishedAt

    started_at = datetime.fromisoformat(parsed.startedAt)
    finished_at = datetime.fromisoformat(parsed.finishedAt)
    assert started_at.tzinfo is not None
    assert finished_at.tzinfo is not None
    assert started_at <= finished_at
