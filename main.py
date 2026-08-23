import json
import hashlib
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import pytz

app = FastAPI()

# File-based persistence for Render
STATE_DIR = os.path.join(os.getcwd(), "state")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "selections.json")


def load_state() -> Dict[str, Any]:
    """Load state from file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_state(state: Dict[str, Any]) -> None:
    """Save state to file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse ISO 8601 timestamp with timezone."""
    try:
        # Handle Z suffix
        if ts.endswith('Z'):
            ts = ts[:-1] + '+00:00'
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def is_valid_timestamp(ts: str) -> bool:
    """Validate timestamp format and timezone offset."""
    dt = parse_timestamp(ts)
    if dt is None:
        return False
    
    # Check timezone offset
    if dt.tzinfo is None:
        return False
    
    offset = dt.utcoffset()
    if offset is None:
        return False
    
    offset_seconds = offset.total_seconds()
    offset_abs = abs(offset_seconds)
    
    # Offset magnitude must be <= 14:00 (50400 seconds)
    if offset_abs > 50400:
        return False
    
    # Offset hour 14 requires minutes 00
    offset_hours = int(offset_abs // 3600)
    offset_minutes = int((offset_abs % 3600) // 60)
    
    if offset_hours == 14 and offset_minutes != 0:
        return False
    
    return True


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compact_json(obj: Any) -> str:
    """Serialize to compact JSON with sorted keys."""
    return json.dumps(obj, separators=(',', ':'), sort_keys=True)


def compute_dataset_digest(train_row_ids: List[str], eval_row_ids: List[str], feature_names: List[str]) -> Optional[str]:
    """Compute SHA-256 digest of dataset."""
    try:
        data = {
            "trainRowIds": sorted(train_row_ids),
            "evalRowIds": sorted(eval_row_ids),
            "featureNames": sorted(feature_names)
        }
        json_str = compact_json(data)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    except Exception:
        return None


def validate_select_request(data: Dict[str, Any]) -> List[str]:
    """Validate select request and return reason codes."""
    reason_codes = []
    
    # Check phase
    if data.get("phase") != "select":
        reason_codes.append("INVALID_INPUT")
        return reason_codes
    
    # Validate runId
    run_id = data.get("runId")
    if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
        reason_codes.append("INVALID_INPUT")
    
    # Validate forbiddenFeatures
    forbidden_features = data.get("forbiddenFeatures")
    if not isinstance(forbidden_features, list):
        reason_codes.append("INVALID_INPUT")
    
    # Validate numTrialsLimit
    num_trials_limit = data.get("numTrialsLimit")
    if not isinstance(num_trials_limit, int) or num_trials_limit <= 0:
        reason_codes.append("INVALID_INPUT")
    
    # Validate rows
    rows = data.get("rows", [])
    if not isinstance(rows, list) or len(rows) == 0:
        reason_codes.append("INVALID_INPUT")
    
    # Validate row IDs are unique
    row_ids = set()
    for row in rows:
        row_id = row.get("id")
        if row_id in row_ids:
            reason_codes.append("INVALID_INPUT")
        row_ids.add(row_id)
        
        # Validate version
        version = row.get("version")
        if not isinstance(version, int) or version < 0:
            reason_codes.append("INVALID_INPUT")
        
        # Validate timestamps
        event_time = row.get("eventTime")
        prediction_time = row.get("predictionTime")
        if not is_valid_timestamp(event_time) or not is_valid_timestamp(prediction_time):
            reason_codes.append("INVALID_INPUT")
    
    # Validate trials
    trials = data.get("trials", [])
    if not isinstance(trials, list):
        reason_codes.append("INVALID_INPUT")
    
    trial_ids = set()
    for trial in trials:
        trial_id = trial.get("trialId")
        if trial_id in trial_ids:
            reason_codes.append("INVALID_INPUT")
        trial_ids.add(trial_id)
        
        if not isinstance(trial_id, int) or trial_id < 0:
            reason_codes.append("INVALID_INPUT")
        
        status = trial.get("status")
        if status not in ["SUCCEEDED", "FAILED"]:
            reason_codes.append("INVALID_INPUT")
        
        eval_metric = trial.get("evalMetric")
        if not isinstance(eval_metric, (int, float)):
            reason_codes.append("INVALID_INPUT")
    
    return reason_codes


def deduplicate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate rows by [entity, UTC(eventTime)]."""
    groups = {}
    
    for row in rows:
        entity = row.get("entity")
        event_time = row.get("eventTime")
        key = (entity, to_utc(parse_timestamp(event_time)))
        
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    
    # Keep highest version, then UTF-8-smallest ID
    retained = []
    for key, group in groups.items():
        # Sort by version descending, then by ID UTF-8 bytes ascending
        group.sort(key=lambda r: (-r.get("version", 0), r.get("id").encode('utf-8')))
        retained.append(group[0])
    
    return retained


def get_eligible_features(rows: List[Dict[str, Any]], forbidden_features: List[str]) -> List[str]:
    """Get eligible feature names."""
    if not rows:
        return []
    
    # Get all feature names from first row
    all_features = set(rows[0].get("features", {}).keys())
    
    # Check each feature appears in every row
    for row in rows[1:]:
        row_features = set(row.get("features", {}).keys())
        all_features = all_features.intersection(row_features)
    
    eligible = []
    for feature in sorted(all_features, key=lambda x: x.encode('utf-8')):
        if feature in forbidden_features:
            continue
        
        # Check availableAt <= predictionTime for all rows
        feature_eligible = True
        for row in rows:
            features = row.get("features", {})
            if feature not in features:
                feature_eligible = False
                break
            
            available_at = features[feature].get("availableAt")
            prediction_time = row.get("predictionTime")
            
            if not is_valid_timestamp(available_at) or not is_valid_timestamp(prediction_time):
                feature_eligible = False
                break
            
            if to_utc(parse_timestamp(available_at)) > to_utc(parse_timestamp(prediction_time)):
                feature_eligible = False
                break
        
        if feature_eligible:
            eligible.append(feature)
    
    return eligible


def select_trial(trials: List[Dict[str, Any]], num_trials_limit: int) -> tuple[Optional[int], List[str]]:
    """Select best trial and return (trialId, reason_codes)."""
    reason_codes = []
    
    # Filter eligible trials
    eligible = []
    for trial in trials:
        if trial.get("status") != "SUCCEEDED":
            continue
        
        eval_metric = trial.get("evalMetric")
        if not isinstance(eval_metric, (int, float)) or not (abs(eval_metric) < float('inf')):
            continue
        
        eligible.append(trial)
    
    # Check trial limit
    if len(trials) > num_trials_limit:
        reason_codes.append("TRIAL_LIMIT_EXCEEDED")
    
    if not eligible:
        return None, reason_codes + ["NO_SUCCESSFUL_TRIAL"]
    
    # Select by max evalMetric, then smallest trialId
    eligible.sort(key=lambda t: (-t.get("evalMetric"), t.get("trialId")))
    return eligible[0].get("trialId"), reason_codes


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/bqml")
async def bqml(request: Request):
    """Main endpoint for select and evaluate phases."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(content={"error": "INVALID_INPUT"}, status_code=400)
    
    phase = data.get("phase")
    
    if phase not in ["select", "evaluate"]:
        return JSONResponse(content={"error": "INVALID_INPUT"}, status_code=400)
    
    if phase == "select":
        return handle_select(data)
    else:
        return handle_evaluate(data)


def handle_select(data: Dict[str, Any]) -> JSONResponse:
    """Handle select phase."""
    reason_codes = validate_select_request(data)
    
    if "INVALID_INPUT" in reason_codes:
        return JSONResponse(content={"error": "INVALID_INPUT"}, status_code=400)
    
    run_id = data.get("runId")
    forbidden_features = data.get("forbiddenFeatures", [])
    num_trials_limit = data.get("numTrialsLimit")
    rows = data.get("rows", [])
    trials = data.get("trials", [])
    
    # Deduplicate rows
    retained_rows = deduplicate_rows(rows)
    
    # Get eligible features
    feature_names = get_eligible_features(retained_rows, forbidden_features)
    
    # Select trial
    selected_trial_id, trial_reason_codes = select_trial(trials, num_trials_limit)
    reason_codes.extend(trial_reason_codes)
    
    # Separate train and eval IDs
    train_row_ids = sorted([r.get("id") for r in retained_rows if r.get("split") == "TRAIN"], key=lambda x: x.encode('utf-8'))
    eval_row_ids = sorted([r.get("id") for r in retained_rows if r.get("split") == "EVAL"], key=lambda x: x.encode('utf-8'))
    
    # Compute dataset digest
    if reason_codes:
        dataset_digest = None
    else:
        dataset_digest = compute_dataset_digest(train_row_ids, eval_row_ids, feature_names)
    
    # If any reason code, set selectedTrialId to null
    if reason_codes:
        selected_trial_id = None
    
    # Build response
    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": sorted(set(reason_codes), key=lambda x: x.encode('utf-8'))
    }
    
    # Check for runId conflict
    state = load_state()
    if run_id in state:
        stored = state[run_id]
        stored_input = stored.get("input")
        if stored_input != data:
            return JSONResponse(content={"error": "RUN_ID_CONFLICT"}, status_code=409)
        else:
            # Return identical stored response
            return JSONResponse(content=stored["response"])
    
    # Store successful selection
    if not reason_codes:
        state[run_id] = {
            "input": data,
            "response": response
        }
        save_state(state)
    
    return JSONResponse(content=response)


def validate_evaluate_request(data: Dict[str, Any]) -> List[str]:
    """Validate evaluate request and return reason codes."""
    reason_codes = []
    
    # Check phase
    if data.get("phase") != "evaluate":
        reason_codes.append("INVALID_INPUT")
        return reason_codes
    
    # Validate runId
    run_id = data.get("runId")
    if not isinstance(run_id, str) or not run_id:
        reason_codes.append("INVALID_INPUT")
    
    # Validate selectedTrialId
    selected_trial_id = data.get("selectedTrialId")
    if not isinstance(selected_trial_id, int) or selected_trial_id is None:
        reason_codes.append("INVALID_INPUT")
    
    # Validate datasetDigest
    dataset_digest = data.get("datasetDigest")
    if not isinstance(dataset_digest, str) or len(dataset_digest) != 64 or not all(c.lower() in '0123456789abcdef' for c in dataset_digest):
        reason_codes.append("INVALID_INPUT")
    
    # Validate metricFloor
    metric_floor = data.get("metricFloor")
    if not isinstance(metric_floor, (int, float)) or not (abs(metric_floor) < float('inf')) or not (0 <= metric_floor <= 1):
        reason_codes.append("INVALID_INPUT")
    
    # Validate requiredSlices
    required_slices = data.get("requiredSlices", {})
    if not isinstance(required_slices, dict):
        reason_codes.append("INVALID_INPUT")
    else:
        for slice_name, floor in required_slices.items():
            if not isinstance(slice_name, str) or not slice_name:
                reason_codes.append("INVALID_INPUT")
            if not isinstance(floor, (int, float)) or not (abs(floor) < float('inf')) or not (0 <= floor <= 1):
                reason_codes.append("INVALID_INPUT")
    
    # Validate bytes
    bytes_processed = data.get("bytesProcessed")
    max_bytes = data.get("maxBytes")
    if not isinstance(bytes_processed, int) or bytes_processed < 0:
        reason_codes.append("INVALID_INPUT")
    if not isinstance(max_bytes, int) or max_bytes < 0:
        reason_codes.append("INVALID_INPUT")
    
    # Validate test rows
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        reason_codes.append("INVALID_INPUT")
    else:
        for row in rows:
            label = row.get("label")
            prediction = row.get("prediction")
            slice_name = row.get("slice")
            
            if label not in [0, 1]:
                reason_codes.append("INVALID_TEST_ROW")
            if prediction not in [0, 1]:
                reason_codes.append("INVALID_TEST_ROW")
            if not isinstance(slice_name, str) or not slice_name:
                reason_codes.append("INVALID_TEST_ROW")
    
    return reason_codes


def handle_evaluate(data: Dict[str, Any]) -> JSONResponse:
    """Handle evaluate phase."""
    reason_codes = validate_evaluate_request(data)
    
    if "INVALID_INPUT" in reason_codes:
        return JSONResponse(content={"error": "INVALID_INPUT"}, status_code=400)
    
    run_id = data.get("runId")
    selected_trial_id = data.get("selectedTrialId")
    dataset_digest = data.get("datasetDigest")
    metric_floor = data.get("metricFloor")
    required_slices = data.get("requiredSlices", {})
    rows = data.get("rows", [])
    bytes_processed = data.get("bytesProcessed")
    max_bytes = data.get("maxBytes")
    
    # Check lineage
    state = load_state()
    if run_id not in state:
        reason_codes.append("INVALID_LINEAGE")
    else:
        stored = state[run_id]["response"]
        if stored.get("selectedTrialId") != selected_trial_id or stored.get("datasetDigest") != dataset_digest:
            reason_codes.append("INVALID_LINEAGE")
    
    # Check for invalid test rows
    has_invalid_test_row = False
    for row in rows:
        if row.get("label") not in [0, 1] or row.get("prediction") not in [0, 1] or not isinstance(row.get("slice"), str) or not row.get("slice"):
            has_invalid_test_row = True
            break
    
    if has_invalid_test_row:
        reason_codes.append("INVALID_TEST_ROW")
    
    # Initialize values
    test_metric = None
    critical_slice_pass = True
    
    # Only compute metrics if rows are valid and non-empty
    if not has_invalid_test_row and rows and "INVALID_LINEAGE" not in reason_codes:
        # Compute aggregate accuracy
        correct = sum(1 for r in rows if r.get("label") == r.get("prediction"))
        total = len(rows)
        test_metric = round(correct / total, 12)
        
        # Check aggregate floor
        if test_metric < metric_floor:
            reason_codes.append("AGGREGATE_FLOOR")
        
        # Compute slice accuracies
        slice_metrics = {}
        slice_counts = {}
        for row in rows:
            slice_name = row.get("slice")
            if slice_name not in slice_counts:
                slice_counts[slice_name] = 0
                slice_metrics[slice_name] = 0
            slice_counts[slice_name] += 1
            if row.get("label") == row.get("prediction"):
                slice_metrics[slice_name] += 1
        
        # Check required slices
        for slice_name, floor in required_slices.items():
            if slice_name not in slice_counts:
                reason_codes.append(f"MISSING_SLICE:{slice_name}")
                critical_slice_pass = False
            else:
                slice_acc = round(slice_metrics[slice_name] / slice_counts[slice_name], 12)
                if slice_acc < floor:
                    reason_codes.append(f"SLICE_FLOOR:{slice_name}")
                    critical_slice_pass = False
    
    # Check byte limit
    if bytes_processed > max_bytes:
        reason_codes.append("BYTE_LIMIT")
    
    # Set criticalSlicePass false for invalid input/lineage/test rows
    if "INVALID_INPUT" in reason_codes or "INVALID_LINEAGE" in reason_codes or "INVALID_TEST_ROW" in reason_codes:
        critical_slice_pass = False
    
    # Set criticalSlicePass false for missing/failed slices
    for rc in reason_codes:
        if rc.startswith("MISSING_SLICE:") or rc.startswith("SLICE_FLOOR:"):
            critical_slice_pass = False
            break
    
    # Determine decision
    decision = "admit"
    if reason_codes:
        decision = "reject"
    
    # Build response
    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed,
        "reasonCodes": sorted(set(reason_codes), key=lambda x: x.encode('utf-8'))
    }
    
    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
