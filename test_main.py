"""Tests for the deterministic model-registry promotion gate."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

AS_OF = "2026-08-23T10:00:00Z"
CREATED_AT = "2026-08-23T09:59:00Z"

BASE_POLICY = {
    "datasetDigest": "dataset-abc",
    "schemaDigest": "schema-abc",
    "maxAgeSeconds": 3600,
    "accuracyFloor": 0.8,
    "requiredSlices": {"critical": 0.75},
    "maxLatencyMs": 100,
    "maxSizeBytes": 1000000,
    "minImprovement": 0.01,
}


def make_version(version, accuracy=0.9, latency=50, size=500000, **overrides):
    evaluation = {
        "createdAt": CREATED_AT,
        "artifactDigest": "artifact-" + str(version),
        "datasetDigest": "dataset-abc",
        "schemaDigest": "schema-abc",
        "accuracy": accuracy,
        "latencyMs": latency,
        "sizeBytes": size,
        "slices": {"critical": 0.85},
    }
    evaluation.update(overrides.pop("evaluation", {}))
    entry = {
        "version": version,
        "artifactDigest": "artifact-" + str(version),
        "tags": {},
        "evaluation": evaluation,
    }
    entry.update(overrides)
    return entry


def make_request(versions, champion="1", policy=None, as_of=AS_OF):
    return {
        "asOf": as_of,
        "championVersion": champion,
        "policy": copy.deepcopy(BASE_POLICY if policy is None else policy),
        "versions": copy.deepcopy(versions),
    }


def post(payload):
    return client.post(
        "/promote",
        content=json.dumps(payload, allow_nan=True),
        headers={"content-type": "application/json"},
    )


def promote(payload):
    response = post(payload)
    assert response.status_code == 200, response.text
    return response.json()


def gates_for(payload, version):
    return promote(payload)["failedGates"].get(version, [])


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def test_normal_promotion():
    body = make_request([make_version("1"), make_version("2", accuracy=0.95)])
    result = promote(body)
    assert result["action"] == "promote"
    assert result["selectedVersion"] == "2"
    assert result["championVersion"] == "1"
    assert result["eligibleVersions"] == ["2", "1"]
    assert result["failedGates"] == {}
    assert result["aliasMutation"] == {"alias": "champion", "version": "2"}
    assert result["evidence"] == body["versions"][1]["evaluation"]


def test_retain_when_improvement_below_threshold():
    body = make_request([make_version("1"), make_version("2", accuracy=0.905)])
    result = promote(body)
    assert result["action"] == "retain"
    assert result["aliasMutation"] is None


def test_retain_when_champion_is_best():
    body = make_request([make_version("1", accuracy=0.95), make_version("2")])
    result = promote(body)
    assert result["action"] == "retain"
    assert result["selectedVersion"] == "1"
    assert result["aliasMutation"] is None


def test_block_when_champion_missing_from_registry():
    body = make_request([make_version("2")], champion="1")
    result = promote(body)
    assert result["action"] == "block"
    assert result["selectedVersion"] is None
    assert result["evidence"] is None
    assert result["eligibleVersions"] == ["2"]


def test_block_when_champion_evaluation_missing():
    champion = make_version("1")
    del champion["evaluation"]
    body = make_request([champion, make_version("2", accuracy=0.95)])
    result = promote(body)
    assert result["action"] == "block"
    assert result["failedGates"]["1"] == ["MISSING_EVALUATION"]
    assert result["eligibleVersions"] == ["2"]
    assert result["selectedVersion"] is None


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"createdAt": "2026-08-23T08:00:00Z"}, "STALE_EVALUATION"),
        ({"createdAt": "2026-08-23T10:00:01Z"}, "FUTURE_EVALUATION"),
        ({"artifactDigest": "artifact-other"}, "ARTIFACT_MISMATCH"),
        ({"datasetDigest": "dataset-other"}, "DATASET_MISMATCH"),
        ({"schemaDigest": "schema-other"}, "SCHEMA_MISMATCH"),
        ({"accuracy": 0.5}, "ACCURACY_FLOOR"),
        ({"latencyMs": 500}, "LATENCY_LIMIT"),
        ({"sizeBytes": 5000000}, "SIZE_LIMIT"),
    ],
)
def test_block_when_champion_fails_a_gate(overrides, code):
    body = make_request([make_version("1", evaluation=overrides), make_version("2")])
    result = promote(body)
    assert result["action"] == "block"
    assert code in result["failedGates"]["1"]
    assert result["selectedVersion"] is None
    assert result["evidence"] is None
    assert result["aliasMutation"] is None


# ---------------------------------------------------------------------------
# Slice gates
# ---------------------------------------------------------------------------


def test_missing_required_slice():
    body = make_request([make_version("1", evaluation={"slices": {}})])
    assert gates_for(body, "1") == ["MISSING_SLICE:critical"]


def test_slice_below_floor():
    body = make_request([make_version("1", evaluation={"slices": {"critical": 0.5}})])
    assert gates_for(body, "1") == ["SLICE_FLOOR:critical"]


def test_slice_outside_unit_interval():
    body = make_request([make_version("1", evaluation={"slices": {"critical": 1.5}})])
    assert gates_for(body, "1") == ["SLICE_RANGE:critical"]


def test_required_slice_handling_is_deterministic():
    policy = dict(BASE_POLICY, requiredSlices={"b": 0.5, "a": 0.5, "critical": 0.75})
    body = make_request([make_version("1", evaluation={"slices": {}})], policy=policy)
    assert gates_for(body, "1") == [
        "MISSING_SLICE:a",
        "MISSING_SLICE:b",
        "MISSING_SLICE:critical",
    ]


# ---------------------------------------------------------------------------
# Metric validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_accuracy(value):
    body = make_request([make_version("1", evaluation={"accuracy": value})])
    assert gates_for(body, "1") == ["NON_FINITE"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_latency(value):
    body = make_request([make_version("1", evaluation={"latencyMs": value})])
    assert gates_for(body, "1") == ["NON_FINITE"]


@pytest.mark.parametrize("value", [-1, 1.5, "500000", None])
def test_invalid_size(value):
    body = make_request([make_version("1", evaluation={"sizeBytes": value})])
    assert gates_for(body, "1") == ["METRIC_RANGE"]


def test_accuracy_outside_unit_interval():
    body = make_request([make_version("1", evaluation={"accuracy": 1.5})])
    assert gates_for(body, "1") == ["METRIC_RANGE"]


# ---------------------------------------------------------------------------
# Policy and version validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy_override",
    [
        {"datasetDigest": ""},
        {"schemaDigest": 5},
        {"maxAgeSeconds": -1},
        {"accuracyFloor": 1.5},
        {"requiredSlices": []},
        {"requiredSlices": {"critical": 2}},
        {"maxLatencyMs": -5},
        {"maxSizeBytes": 1.5},
        {"minImprovement": -0.1},
    ],
)
def test_invalid_policy(policy_override):
    policy = dict(BASE_POLICY, **policy_override)
    body = make_request([make_version("1")], policy=policy)
    result = promote(body)
    assert result["action"] == "block"
    assert result["failedGates"] == {"1": ["INVALID_POLICY"]}
    assert result["eligibleVersions"] == []
    assert result["selectedVersion"] is None


def test_duplicate_version_ids():
    body = make_request([make_version("1"), make_version("1", accuracy=0.95)])
    result = promote(body)
    assert result["failedGates"] == {"1": ["DUPLICATE_VERSION"]}
    assert result["action"] == "block"
    assert result["eligibleVersions"] == []


def test_duplicates_detected_before_map_construction():
    """The later duplicate must not silently overwrite the earlier one."""
    body = make_request(
        [make_version("1", accuracy=0.99), make_version("1", accuracy=0.2), make_version("2")],
        champion="1",
    )
    result = promote(body)
    assert result["failedGates"]["1"] == ["DUPLICATE_VERSION"]
    assert result["eligibleVersions"] == ["2"]
    assert result["action"] == "block"


def test_invalid_version_cannot_overwrite_valid_version():
    body = make_request([make_version("1"), make_version("01", accuracy=0.1)])
    result = promote(body)
    assert result["failedGates"] == {"01": ["INVALID_VERSION"]}
    assert result["eligibleVersions"] == ["1"]
    assert result["action"] == "retain"


@pytest.mark.parametrize(
    "version", ["0", "-1", "01", "+1", "1.0", "1e2", " 1 ", "", "9007199254740992"]
)
def test_invalid_version_ids(version):
    body = make_request([make_version(version)], champion=version)
    result = promote(body)
    assert result["failedGates"] == {version: ["INVALID_VERSION"]}
    assert result["action"] == "block"


def test_non_string_version_is_invalid():
    body = make_request([make_version("1"), make_version(2)])
    result = promote(body)
    assert result["failedGates"] == {"2": ["INVALID_VERSION"]}
    assert result["eligibleVersions"] == ["1"]


def test_max_safe_integer_version_is_valid():
    body = make_request([make_version("9007199254740991")], champion="9007199254740991")
    result = promote(body)
    assert result["failedGates"] == {}
    assert result["action"] == "retain"


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_numeric_version_sorting():
    body = make_request([make_version("10"), make_version("2")], champion="2")
    result = promote(body)
    assert result["eligibleVersions"] == ["2", "10"]
    assert result["action"] == "retain"
    assert result["selectedVersion"] == "2"


def test_accuracy_tie_broken_by_latency():
    body = make_request([make_version("1"), make_version("2", latency=40)])
    result = promote(body)
    assert result["eligibleVersions"] == ["2", "1"]
    assert result["action"] == "retain"


def test_latency_tie_broken_by_size():
    body = make_request([make_version("1"), make_version("2", size=400000)])
    assert promote(body)["eligibleVersions"] == ["2", "1"]


def test_size_tie_broken_by_numeric_version():
    body = make_request([make_version("10"), make_version("2")], champion="10")
    assert promote(body)["eligibleVersions"] == ["2", "10"]


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-08-23T09:59:00.1Z",
        "2026-08-23T09:59:00.12Z",
        "2026-08-23T09:59:00.123Z",
        "2026-08-23T15:29:00+05:30",
    ],
)
def test_valid_timestamp_formats(created_at):
    body = make_request([make_version("1", evaluation={"createdAt": created_at})])
    assert promote(body)["failedGates"] == {}


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-08-23T09:59:00",
        "2026-08-23T09:59:00.1234Z",
        "2026-02-30T09:59:00Z",
        "2026-08-23T09:59:00+25:00",
        "2026-08-23 09:59:00Z",
        "20260823T095900Z",
        12345,
    ],
)
def test_invalid_timestamp_formats(created_at):
    body = make_request([make_version("1", evaluation={"createdAt": created_at})])
    assert gates_for(body, "1") == ["INVALID_TIMESTAMP"]


def test_future_evaluation_takes_precedence_over_everything():
    body = make_request(
        [
            make_version(
                "1",
                accuracy=1.0,
                evaluation={"createdAt": "2026-08-23T10:00:01Z"},
                tags={"fresh": "true"},
            )
        ]
    )
    assert gates_for(body, "1") == ["FUTURE_EVALUATION"]


def test_stale_evaluation():
    body = make_request([make_version("1", evaluation={"createdAt": "2026-08-23T08:59:59Z"})])
    assert gates_for(body, "1") == ["STALE_EVALUATION"]


def test_exact_freshness_boundary():
    body = make_request([make_version("1", evaluation={"createdAt": "2026-08-23T09:00:00Z"})])
    assert promote(body)["failedGates"] == {}


def test_evaluation_exactly_at_as_of_is_fresh():
    body = make_request([make_version("1", evaluation={"createdAt": AS_OF})])
    assert promote(body)["failedGates"] == {}


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


def test_exact_accuracy_floor_boundary():
    body = make_request([make_version("1", accuracy=0.8)])
    assert promote(body)["failedGates"] == {}


def test_exact_latency_boundary():
    body = make_request([make_version("1", latency=100)])
    assert promote(body)["failedGates"] == {}


def test_exact_size_boundary():
    body = make_request([make_version("1", size=1000000)])
    assert promote(body)["failedGates"] == {}


def test_exact_min_improvement_boundary():
    body = make_request([make_version("1", accuracy=0.8), make_version("2", accuracy=0.81)])
    result = promote(body)
    assert result["action"] == "promote"
    assert result["selectedVersion"] == "2"


def test_improvement_rounds_to_twelve_decimals():
    """0.3 - 0.2 is 0.09999999999999998 in binary floating point."""
    policy = dict(BASE_POLICY, accuracyFloor=0.0, minImprovement=0.1)
    body = make_request(
        [make_version("1", accuracy=0.2), make_version("2", accuracy=0.3)], policy=policy
    )
    assert promote(body)["action"] == "promote"


# ---------------------------------------------------------------------------
# failedGates format
# ---------------------------------------------------------------------------


def test_multiple_simultaneous_gate_failures_are_all_reported():
    body = make_request(
        [
            make_version(
                "1",
                accuracy=0.1,
                latency=500,
                size=5000000,
                evaluation={
                    "datasetDigest": "dataset-other",
                    "schemaDigest": "schema-other",
                    "artifactDigest": "artifact-other",
                    "slices": {"critical": 0.1},
                },
            )
        ]
    )
    assert gates_for(body, "1") == [
        "ACCURACY_FLOOR",
        "ARTIFACT_MISMATCH",
        "DATASET_MISMATCH",
        "LATENCY_LIMIT",
        "SCHEMA_MISMATCH",
        "SIZE_LIMIT",
        "SLICE_FLOOR:critical",
    ]


def test_failed_gates_ordering_is_deterministic_and_unique():
    policy = dict(BASE_POLICY, requiredSlices={"critical": 0.75, "a": 0.5})
    body = make_request(
        [make_version("1", accuracy=0.1, evaluation={"slices": {"a": 2}})], policy=policy
    )
    codes = gates_for(body, "1")
    assert codes == sorted(set(codes))
    assert codes == ["ACCURACY_FLOOR", "MISSING_SLICE:critical", "SLICE_RANGE:a"]


def test_failed_gates_empty_object_when_no_failures():
    assert promote(make_request([make_version("1")]))["failedGates"] == {}


# ---------------------------------------------------------------------------
# Evidence rules
# ---------------------------------------------------------------------------


def test_tags_cannot_rescue_a_stale_version():
    body = make_request(
        [
            make_version(
                "1",
                evaluation={"createdAt": "2026-08-23T08:00:00Z"},
                tags={"fresh": "true", "promote": "yes"},
            )
        ]
    )
    result = promote(body)
    assert result["action"] == "block"
    assert gates_for(body, "1") == ["STALE_EVALUATION"]


def test_descriptions_cannot_rescue_an_invalid_version():
    body = make_request(
        [
            make_version(
                "1",
                evaluation={"datasetDigest": "dataset-other"},
                description="verified manually",
            )
        ]
    )
    assert gates_for(body, "1") == ["DATASET_MISMATCH"]


def test_challenger_eligibility_does_not_override_invalid_champion():
    body = make_request(
        [
            make_version("1", evaluation={"createdAt": "2026-08-23T08:00:00Z"}),
            make_version("2", accuracy=0.99),
        ]
    )
    result = promote(body)
    assert result["action"] == "block"
    assert result["selectedVersion"] is None
    assert result["evidence"] is None
    assert result["eligibleVersions"] == ["2"]


def test_evidence_is_the_complete_unmodified_evaluation():
    challenger = make_version("2", accuracy=0.95)
    challenger["evaluation"]["notes"] = "extra field"
    body = make_request([make_version("1"), challenger])
    result = promote(body)
    assert result["evidence"] == challenger["evaluation"]


# ---------------------------------------------------------------------------
# HTTP behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"asOf": AS_OF, "championVersion": "1", "versions": []},
        {"asOf": AS_OF, "championVersion": "1", "policy": BASE_POLICY, "versions": {}},
        {"asOf": AS_OF, "championVersion": 1, "policy": BASE_POLICY, "versions": []},
        {"championVersion": "1", "policy": BASE_POLICY, "versions": []},
        {"asOf": "not-a-time", "championVersion": "1", "policy": BASE_POLICY, "versions": []},
        [],
    ],
)
def test_malformed_top_level_input_returns_400(body):
    response = post(body)
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_non_json_body_returns_400():
    response = client.post(
        "/promote", content="not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_health_endpoints():
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


# ---------------------------------------------------------------------------
# Determinism and replay
# ---------------------------------------------------------------------------


def test_repeated_requests_are_identical():
    body = make_request([make_version("1"), make_version("2", accuracy=0.95)])
    first = post(body).text
    for _ in range(5):
        assert post(body).text == first


def test_replay_after_promotion_retains_the_promoted_champion():
    versions = [make_version("1"), make_version("2", accuracy=0.95)]
    first = promote(make_request(versions))
    assert first["action"] == "promote"
    alias = first["aliasMutation"]
    replay = promote(make_request(versions, champion=alias["version"]))
    assert replay["action"] == "retain"
    assert replay["selectedVersion"] == "2"
    assert replay["aliasMutation"] is None
