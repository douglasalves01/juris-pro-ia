from __future__ import annotations

import math
from datetime import datetime

from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st

from api.main import (
    _METRICS_BUFFER_SIZE,
    _metrics_buffer,
    _metrics_lock,
    app,
    compute_pipeline_metrics,
    record_pipeline_metrics,
)
from api.schemas.analysis import PipelineMetricsResponse


def _clear_metrics() -> None:
    with _metrics_lock:
        _metrics_buffer.clear()


@given(durations=st.lists(st.integers(min_value=0, max_value=100_000), min_size=1, max_size=100))
@settings(max_examples=100)
def test_pipeline_metrics_calcula_media_e_p95(durations: list[int]) -> None:
    _clear_metrics()
    for duration in durations:
        record_pipeline_metrics([{"step": "classify_document", "durationMs": duration}])

    metrics = compute_pipeline_metrics()

    assert len(metrics) == 1
    metric = metrics[0]
    sorted_durations = sorted(durations)
    p95_index = max(0, math.ceil(len(sorted_durations) * 0.95) - 1)
    assert metric["step"] == "classify_document"
    assert metric["avgDurationMs"] == round(sum(durations) / len(durations), 2)
    assert metric["p95DurationMs"] == sorted_durations[p95_index]
    assert metric["callCount"] == len(durations)


@given(
    executions=st.lists(
        st.lists(st.integers(min_value=0, max_value=10_000), min_size=1, max_size=5),
        min_size=101,
        max_size=140,
    )
)
@settings(max_examples=100)
def test_pipeline_metrics_buffer_circular_nao_excede_100_execucoes(
    executions: list[list[int]],
) -> None:
    _clear_metrics()
    for execution in executions:
        record_pipeline_metrics(
            [
                {"step": f"step_{idx}", "durationMs": duration}
                for idx, duration in enumerate(execution)
            ]
        )

    with _metrics_lock:
        assert len(_metrics_buffer) <= _METRICS_BUFFER_SIZE
        assert len(_metrics_buffer) == _METRICS_BUFFER_SIZE


def test_metrics_pipeline_endpoint_retorna_schema_com_buffer_vazio() -> None:
    _clear_metrics()

    with TestClient(app) as client:
        response = client.get("/metrics/pipeline")

    assert response.status_code == 200
    parsed = PipelineMetricsResponse.model_validate(response.json())
    assert parsed.steps == []
    assert parsed.collectedAt
    assert datetime.fromisoformat(parsed.collectedAt).tzinfo is not None
    assert parsed.semanticCache is not None
    assert parsed.semanticCache.cacheHitRate >= 0.0
    assert parsed.semanticCache.cacheHitRate <= 1.0
