"""Deterministic model-registry promotion gate.

A single endpoint, ``POST /promote``, decides whether a challenger model
version may replace the current champion. The decision depends only on the
request body: there is no persistence, no wall-clock time and no randomness.
"""

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Model Registry Promotion Gate")

MAX_SAFE_INTEGER = 9007199254740991
ALIAS_NAME = "champion"

VERSION_RE = re.compile(r"^[1-9][0-9]*$")
TIMESTAMP_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?"
    r"(?:Z|([+-])(\d{2}):(\d{2}))$"
)


def invalid_input_response() -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})


class InvalidInput(Exception):
    """Raised for malformed top-level request structures."""


# ---------------------------------------------------------------------------
# Primitive validation helpers
# ---------------------------------------------------------------------------


def is_number(value: Any) -> bool:
    """True for JSON numbers (booleans are not numbers)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_finite_number(value: Any) -> bool:
    return is_number(value) and math.isfinite(value)


def is_safe_integer(value: Any) -> bool:
    if not is_finite_number(value):
        return False
    if isinstance(value, float) and not value.is_integer():
        return False
    return abs(int(value)) <= MAX_SAFE_INTEGER


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def is_unit_interval(value: Any) -> bool:
    return is_finite_number(value) and 0.0 <= float(value) <= 1.0


def parse_version(value: Any) -> Optional[int]:
    """Return the numeric value of a canonical version ID, else None."""
    if not isinstance(value, str) or not VERSION_RE.match(value):
        return None
    number = int(value)
    if number > MAX_SAFE_INTEGER:
        return None
    return number


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse ``YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)`` into a UTC instant."""
    if not isinstance(value, str):
        return None
    match = TIMESTAMP_RE.match(value)
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(match.group(i)) for i in range(1, 7))
    fraction = match.group(7)
    microsecond = int(fraction.ljust(3, "0")) * 1000 if fraction else 0
    sign, offset_hours, offset_minutes = match.group(8), match.group(9), match.group(10)
    if sign is None:
        tzinfo = timezone.utc
    else:
        hours, minutes = int(offset_hours), int(offset_minutes)
        if hours > 23 or minutes > 59:
            return None
        delta = timedelta(hours=hours, minutes=minutes)
        tzinfo = timezone(-delta if sign == "-" else delta)
    try:
        moment = datetime(year, month, day, hour, minute, second, microsecond, tzinfo)
    except ValueError:
        return None
    return moment.astimezone(timezone.utc)


def version_key(value: Any) -> str:
    """Key used for a version inside ``failedGates``."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def policy_is_valid(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False
    if not is_non_empty_string(policy.get("datasetDigest")):
        return False
    if not is_non_empty_string(policy.get("schemaDigest")):
        return False
    max_age = policy.get("maxAgeSeconds")
    if not is_safe_integer(max_age) or max_age < 0:
        return False
    if not is_unit_interval(policy.get("accuracyFloor")):
        return False
    required_slices = policy.get("requiredSlices")
    if not isinstance(required_slices, dict):
        return False
    for floor in required_slices.values():
        if not is_unit_interval(floor):
            return False
    max_latency = policy.get("maxLatencyMs")
    if not is_finite_number(max_latency) or max_latency < 0:
        return False
    max_size = policy.get("maxSizeBytes")
    if not is_safe_integer(max_size) or max_size < 0:
        return False
    if not is_unit_interval(policy.get("minImprovement")):
        return False
    return True


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def evaluation_gates(entry: Dict[str, Any], policy: Dict[str, Any], as_of: datetime) -> List[str]:
    """All gate codes failed by a structurally valid version entry."""
    gates: List[str] = []
    evaluation = entry.get("evaluation")
    if not isinstance(evaluation, dict):
        return ["MISSING_EVALUATION"]

    accuracy = evaluation.get("accuracy")
    latency = evaluation.get("latencyMs")
    size = evaluation.get("sizeBytes")

    metrics = (accuracy, latency, size)
    if any(is_number(value) and not math.isfinite(value) for value in metrics):
        gates.append("NON_FINITE")

    accuracy_ok = is_unit_interval(accuracy)
    latency_ok = is_finite_number(latency) and latency >= 0
    size_ok = is_safe_integer(size) and size >= 0
    checked = ((accuracy, accuracy_ok), (latency, latency_ok), (size, size_ok))
    if any(
        not ok and not (is_number(value) and not math.isfinite(value))
        for value, ok in checked
    ):
        gates.append("METRIC_RANGE")

    if evaluation.get("artifactDigest") != entry.get("artifactDigest"):
        gates.append("ARTIFACT_MISMATCH")
    if evaluation.get("datasetDigest") != policy["datasetDigest"]:
        gates.append("DATASET_MISMATCH")
    if evaluation.get("schemaDigest") != policy["schemaDigest"]:
        gates.append("SCHEMA_MISMATCH")

    created_at = parse_timestamp(evaluation.get("createdAt"))
    if created_at is None:
        gates.append("INVALID_TIMESTAMP")
    elif created_at > as_of:
        gates.append("FUTURE_EVALUATION")
    elif created_at < as_of - timedelta(seconds=policy["maxAgeSeconds"]):
        gates.append("STALE_EVALUATION")

    slices = evaluation.get("slices")
    slices = slices if isinstance(slices, dict) else {}
    for name in sorted(policy["requiredSlices"]):
        floor = policy["requiredSlices"][name]
        if name not in slices:
            gates.append("MISSING_SLICE:" + name)
            continue
        value = slices[name]
        if not is_unit_interval(value):
            gates.append("SLICE_RANGE:" + name)
        elif value < floor:
            gates.append("SLICE_FLOOR:" + name)

    if accuracy_ok and accuracy < policy["accuracyFloor"]:
        gates.append("ACCURACY_FLOOR")
    if latency_ok and latency > policy["maxLatencyMs"]:
        gates.append("LATENCY_LIMIT")
    if size_ok and size > policy["maxSizeBytes"]:
        gates.append("SIZE_LIMIT")

    return gates


def ranking_key(entry: Dict[str, Any]) -> Tuple[float, float, int, int]:
    evaluation = entry["evaluation"]
    return (
        -float(evaluation["accuracy"]),
        float(evaluation["latencyMs"]),
        int(evaluation["sizeBytes"]),
        parse_version(entry["version"]),
    )


def improvement_of(selected: Dict[str, Any], champion: Dict[str, Any]) -> Decimal:
    difference = Decimal(str(selected["evaluation"]["accuracy"])) - Decimal(
        str(champion["evaluation"]["accuracy"])
    )
    return round(difference, 12)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def decide(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise InvalidInput()

    policy = body.get("policy")
    if policy is None:
        raise InvalidInput()
    versions = body.get("versions")
    if not isinstance(versions, list):
        raise InvalidInput()
    champion_version = body.get("championVersion")
    if not isinstance(champion_version, str):
        raise InvalidInput()
    as_of = parse_timestamp(body.get("asOf"))
    if as_of is None:
        raise InvalidInput()

    failed_gates: Dict[str, List[str]] = {}

    def add_gates(key: str, codes: List[str]) -> None:
        failed_gates.setdefault(key, [])
        failed_gates[key].extend(codes)

    # Duplicate detection happens before any lookup map is built.
    occurrences: Dict[str, int] = {}
    for entry in versions:
        raw = entry.get("version") if isinstance(entry, dict) else entry
        key = version_key(raw)
        occurrences[key] = occurrences.get(key, 0) + 1

    valid_entries: Dict[str, Dict[str, Any]] = {}
    for entry in versions:
        raw = entry.get("version") if isinstance(entry, dict) else entry
        key = version_key(raw)
        codes: List[str] = []
        if parse_version(raw) is None or not isinstance(entry, dict):
            codes.append("INVALID_VERSION")
        if occurrences[key] > 1:
            codes.append("DUPLICATE_VERSION")
        if codes:
            add_gates(key, codes)
            continue
        valid_entries[key] = entry

    policy_valid = policy_is_valid(policy)
    if policy_valid:
        for key, entry in valid_entries.items():
            gates = evaluation_gates(entry, policy, as_of)
            if gates:
                add_gates(key, gates)
    else:
        for key in list(valid_entries) + list(failed_gates):
            add_gates(key, ["INVALID_POLICY"])
        valid_entries = {}

    eligible = [
        entry for key, entry in valid_entries.items() if not failed_gates.get(key)
    ]
    eligible.sort(key=ranking_key)
    eligible_versions = [entry["version"] for entry in eligible]

    response: Dict[str, Any] = {
        "action": "block",
        "championVersion": champion_version,
        "selectedVersion": None,
        "eligibleVersions": eligible_versions,
        "failedGates": {
            key: sorted(set(codes)) for key, codes in sorted(failed_gates.items()) if codes
        },
        "aliasMutation": None,
        "evidence": None,
    }

    champion = next(
        (entry for entry in eligible if entry["version"] == champion_version), None
    )
    if champion is None:
        return response

    selected = eligible[0]
    response["selectedVersion"] = selected["version"]
    response["evidence"] = selected["evaluation"]
    if selected["version"] == champion_version:
        response["action"] = "retain"
        return response

    if improvement_of(selected, champion) >= Decimal(str(policy["minImprovement"])):
        response["action"] = "promote"
        response["aliasMutation"] = {"alias": ALIAS_NAME, "version": selected["version"]}
    else:
        response["action"] = "retain"
    return response


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


@app.post("/promote")
async def promote(request: Request) -> JSONResponse:
    try:
        body = json.loads(await request.body())
    except (ValueError, UnicodeDecodeError):
        return invalid_input_response()
    try:
        return JSONResponse(status_code=200, content=decide(body))
    except InvalidInput:
        return invalid_input_response()
    except Exception:
        return invalid_input_response()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok", "service": "promotion-gate"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
