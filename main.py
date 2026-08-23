"""Stateful two-phase experiment gate (select / evaluate).

Feature values are treated strictly as opaque data and are never interpreted.
"""

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MAX_SAFE_INTEGER = 9007199254740991
MAX_RUN_ID_LENGTH = 128

DB_PATH = os.environ.get(
    "STATE_DB_PATH", os.path.join(os.getcwd(), "state", "state.db")
)

_db_lock = threading.Lock()

TIMESTAMP_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?"
    r"(?:Z|([+-])(\d{2}):(\d{2}))$"
)
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------
# Persistence (SQLite file, survives process restarts)
# --------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS selections ("
        "run_id TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL)"
    )
    return conn


def get_selection(run_id: str) -> Optional[Dict[str, Any]]:
    """Return the stored selection record for run_id, or None."""
    with _db_lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT request, response FROM selections WHERE run_id = ?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    return {"request": json.loads(row[0]), "response": json.loads(row[1])}


def put_selection(run_id: str, request: Dict[str, Any], response: Dict[str, Any]) -> None:
    """Persist a successful selection under run_id."""
    with _db_lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO selections (run_id, request, response) "
                "VALUES (?, ?, ?)",
                (run_id, canonical_json(request), json.dumps(response)),
            )
            conn.commit()
        finally:
            conn.close()


def clear_selections() -> None:
    """Remove all stored selections."""
    with _db_lock:
        conn = _connect()
        try:
            conn.execute("DELETE FROM selections")
            conn.commit()
        finally:
            conn.close()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Stable serialization used to fingerprint stored selection requests."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def utf8_sorted(values: List[str]) -> List[str]:
    return sorted(values, key=lambda v: v.encode("utf-8"))


def is_safe_int(value: Any, minimum: int = 0) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= MAX_SAFE_INTEGER
    )


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def parse_timestamp(ts: Any) -> Optional[datetime]:
    """Parse a strict RFC3339-style timestamp; return None when invalid."""
    if not isinstance(ts, str):
        return None
    match = TIMESTAMP_RE.match(ts)
    if match is None:
        return None

    year, month, day, hour, minute, second = (int(match.group(i)) for i in range(1, 7))
    fraction = match.group(7)
    sign, offset_hour, offset_minute = match.group(8), match.group(9), match.group(10)

    if hour > 23 or minute > 59 or second > 59:
        return None

    microsecond = int(fraction.ljust(3, "0")) * 1000 if fraction else 0

    if sign is None:
        tzinfo = timezone.utc
    else:
        offset_hour, offset_minute = int(offset_hour), int(offset_minute)
        if offset_minute > 59:
            return None
        if offset_hour > 14 or (offset_hour == 14 and offset_minute != 0):
            return None
        delta = timedelta(hours=offset_hour, minutes=offset_minute)
        tzinfo = timezone(-delta if sign == "-" else delta)

    try:
        return datetime(year, month, day, hour, minute, second, microsecond, tzinfo)
    except ValueError:
        return None


def is_valid_timestamp(ts: Any) -> bool:
    return parse_timestamp(ts) is not None


def to_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc)


def compute_dataset_digest(
    train_row_ids: List[str], eval_row_ids: List[str], feature_names: List[str]
) -> str:
    payload = (
        '{"trainRowIds":'
        + json.dumps(train_row_ids, separators=(",", ":"))
        + ',"evalRowIds":'
        + json.dumps(eval_row_ids, separators=(",", ":"))
        + ',"featureNames":'
        + json.dumps(feature_names, separators=(",", ":"))
        + "}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def error_response(code: str, status: int) -> JSONResponse:
    return JSONResponse(content={"error": code}, status_code=status)


# --------------------------------------------------------------------------
# Select phase
# --------------------------------------------------------------------------


def is_valid_select_request(data: Dict[str, Any]) -> bool:
    run_id = data.get("runId")
    if not is_non_empty_string(run_id) or len(run_id) > MAX_RUN_ID_LENGTH:
        return False

    forbidden = data.get("forbiddenFeatures", [])
    if not isinstance(forbidden, list) or not all(
        isinstance(name, str) for name in forbidden
    ):
        return False

    if not is_safe_int(data.get("numTrialsLimit"), minimum=1):
        return False

    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        return False

    seen_row_ids = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        row_id = row.get("id")
        if not is_non_empty_string(row_id) or row_id in seen_row_ids:
            return False
        seen_row_ids.add(row_id)

        if not is_non_empty_string(row.get("entity")):
            return False
        if row.get("split") not in ("TRAIN", "EVAL"):
            return False
        if not is_safe_int(row.get("version")):
            return False

        prediction_time = parse_timestamp(row.get("predictionTime"))
        if prediction_time is None or not is_valid_timestamp(row.get("eventTime")):
            return False

        features = row.get("features", {})
        if not isinstance(features, dict):
            return False
        for name, feature in features.items():
            if not is_non_empty_string(name) or not isinstance(feature, dict):
                return False
            if not is_valid_timestamp(feature.get("availableAt")):
                return False

    trials = data.get("trials", [])
    if not isinstance(trials, list):
        return False

    seen_trial_ids = set()
    for trial in trials:
        if not isinstance(trial, dict):
            return False
        trial_id = trial.get("trialId")
        if not is_safe_int(trial_id) or trial_id in seen_trial_ids:
            return False
        seen_trial_ids.add(trial_id)

        if trial.get("status") not in ("SUCCEEDED", "FAILED"):
            return False
        if not is_number(trial.get("evalMetric")):
            return False

    return True


def deduplicate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one row per [entity, UTC(eventTime)]: highest version, then smallest id."""
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["entity"], to_utc(parse_timestamp(row["eventTime"])).isoformat())
        current = groups.get(key)
        if current is None:
            groups[key] = row
            continue
        candidate_rank = (-row["version"], row["id"].encode("utf-8"))
        current_rank = (-current["version"], current["id"].encode("utf-8"))
        if candidate_rank < current_rank:
            groups[key] = row
    return list(groups.values())


def eligible_features(
    rows: List[Dict[str, Any]], forbidden_features: List[str]
) -> List[str]:
    common: Optional[set] = None
    for row in rows:
        names = set(row.get("features", {}).keys())
        common = names if common is None else common & names
    if not common:
        return []

    forbidden = set(forbidden_features)
    eligible = []
    for name in utf8_sorted(list(common)):
        if name in forbidden:
            continue
        available = all(
            to_utc(parse_timestamp(row["features"][name]["availableAt"]))
            <= to_utc(parse_timestamp(row["predictionTime"]))
            for row in rows
        )
        if available:
            eligible.append(name)
    return eligible


def select_trial(
    trials: List[Dict[str, Any]], num_trials_limit: int
) -> Tuple[Optional[int], List[str]]:
    reason_codes: List[str] = []
    if len(trials) > num_trials_limit:
        reason_codes.append("TRIAL_LIMIT_EXCEEDED")

    eligible = [
        trial
        for trial in trials
        if trial["status"] == "SUCCEEDED" and is_finite_number(trial["evalMetric"])
    ]
    if not eligible:
        reason_codes.append("NO_SUCCESSFUL_TRIAL")
        return None, reason_codes

    best = min(eligible, key=lambda t: (-t["evalMetric"], t["trialId"]))
    return best["trialId"], reason_codes


def handle_select(data: Dict[str, Any]) -> JSONResponse:
    if not is_valid_select_request(data):
        return error_response("INVALID_INPUT", 400)

    run_id: str = data["runId"]
    retained = deduplicate_rows(data["rows"])

    train_row_ids = utf8_sorted([r["id"] for r in retained if r["split"] == "TRAIN"])
    eval_row_ids = utf8_sorted([r["id"] for r in retained if r["split"] == "EVAL"])
    feature_names = eligible_features(retained, data.get("forbiddenFeatures", []))

    selected_trial_id, reason_codes = select_trial(
        data.get("trials", []), data["numTrialsLimit"]
    )

    if reason_codes:
        selected_trial_id = None
        dataset_digest = None
    else:
        dataset_digest = compute_dataset_digest(
            train_row_ids, eval_row_ids, feature_names
        )

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": utf8_sorted(list(set(reason_codes))),
    }

    stored = get_selection(run_id)
    if stored is not None:
        if canonical_json(stored["request"]) != canonical_json(data):
            return error_response("RUN_ID_CONFLICT", 409)
        return JSONResponse(content=stored["response"])

    if not reason_codes:
        put_selection(run_id, data, response)

    return JSONResponse(content=response)


# --------------------------------------------------------------------------
# Evaluate phase
# --------------------------------------------------------------------------


def is_valid_evaluate_request(data: Dict[str, Any]) -> bool:
    run_id = data.get("runId")
    if not is_non_empty_string(run_id) or len(run_id) > MAX_RUN_ID_LENGTH:
        return False

    selected_trial_id = data.get("selectedTrialId")
    if not is_safe_int(selected_trial_id):
        return False

    dataset_digest = data.get("datasetDigest")
    if not isinstance(dataset_digest, str) or DIGEST_RE.match(dataset_digest) is None:
        return False

    metric_floor = data.get("metricFloor")
    if not is_finite_number(metric_floor) or not 0 <= metric_floor <= 1:
        return False

    required_slices = data.get("requiredSlices", {})
    if not isinstance(required_slices, dict):
        return False
    for name, floor in required_slices.items():
        if not is_non_empty_string(name):
            return False
        if not is_finite_number(floor) or not 0 <= floor <= 1:
            return False

    if not is_safe_int(data.get("bytesProcessed")):
        return False
    if not is_safe_int(data.get("maxBytes")):
        return False

    if not isinstance(data.get("rows", []), list):
        return False

    return True


def is_valid_test_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    label, prediction = row.get("label"), row.get("prediction")
    if not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
        return False
    if (
        not isinstance(prediction, int)
        or isinstance(prediction, bool)
        or prediction not in (0, 1)
    ):
        return False
    return is_non_empty_string(row.get("slice"))


def handle_evaluate(data: Dict[str, Any]) -> JSONResponse:
    if not is_valid_evaluate_request(data):
        return error_response("INVALID_INPUT", 400)

    run_id: str = data["runId"]
    selected_trial_id: int = data["selectedTrialId"]
    dataset_digest: str = data["datasetDigest"]
    metric_floor: float = data["metricFloor"]
    required_slices: Dict[str, float] = data.get("requiredSlices", {})
    rows: List[Any] = data.get("rows", [])
    bytes_processed: int = data["bytesProcessed"]
    max_bytes: int = data["maxBytes"]

    reason_codes: List[str] = []
    critical_slice_pass = True

    stored = get_selection(run_id)
    if (
        stored is None
        or stored["response"]["selectedTrialId"] != selected_trial_id
        or stored["response"]["datasetDigest"] != dataset_digest
    ):
        reason_codes.append("INVALID_LINEAGE")
        critical_slice_pass = False

    rows_usable = bool(rows) and all(is_valid_test_row(row) for row in rows)
    if rows and not rows_usable:
        reason_codes.append("INVALID_TEST_ROW")
    if not rows_usable:
        critical_slice_pass = False

    test_metric: Optional[float] = None
    if rows_usable:
        correct = sum(1 for row in rows if row["label"] == row["prediction"])
        test_metric = round(correct / len(rows), 12)
        if test_metric < metric_floor:
            reason_codes.append("AGGREGATE_FLOOR")

        totals: Dict[str, int] = {}
        hits: Dict[str, int] = {}
        for row in rows:
            name = row["slice"]
            totals[name] = totals.get(name, 0) + 1
            hits[name] = hits.get(name, 0) + (1 if row["label"] == row["prediction"] else 0)

        for name in utf8_sorted(list(required_slices.keys())):
            floor = required_slices[name]
            if name not in totals:
                reason_codes.append(f"MISSING_SLICE:{name}")
                critical_slice_pass = False
                continue
            slice_metric = round(hits[name] / totals[name], 12)
            if slice_metric < floor:
                reason_codes.append(f"SLICE_FLOOR:{name}")
                critical_slice_pass = False

    if bytes_processed > max_bytes:
        reason_codes.append("BYTE_LIMIT")

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": "reject" if reason_codes else "admit",
        "bytesProcessed": bytes_processed,
        "reasonCodes": utf8_sorted(list(set(reason_codes))),
    }
    return JSONResponse(content=response)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, status_code=200)


@app.post("/bqml")
async def bqml(request: Request) -> JSONResponse:
    try:
        data = await request.json()
    except Exception:
        return error_response("INVALID_INPUT", 400)

    if not isinstance(data, dict):
        return error_response("INVALID_INPUT", 400)

    phase = data.get("phase")
    if phase == "select":
        return handle_select(data)
    if phase == "evaluate":
        return handle_evaluate(data)
    return error_response("INVALID_INPUT", 400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
