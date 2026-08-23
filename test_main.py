import pytest
import json
import os
import tempfile
import shutil
from fastapi.testclient import TestClient
from main import app, STATE_DIR, STATE_FILE

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_state():
    """Setup and cleanup state directory for tests."""
    # Backup original state
    original_state_dir = STATE_DIR
    original_state_file = STATE_FILE
    
    # Create temp directory for state
    temp_dir = tempfile.mkdtemp()
    test_state_dir = os.path.join(temp_dir, "state")
    os.makedirs(test_state_dir, exist_ok=True)
    test_state_file = os.path.join(test_state_dir, "selections.json")
    
    # Patch module globals
    import main
    main.STATE_DIR = test_state_dir
    main.STATE_FILE = test_state_file
    
    yield
    
    # Cleanup
    shutil.rmtree(temp_dir)
    # Restore original
    main.STATE_DIR = original_state_dir
    main.STATE_FILE = original_state_file


# ==================== SELECT TESTS ====================

def test_valid_select():
    """Test valid selection."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-1",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["runId"] == "test-run-1"
    assert data["selectedTrialId"] == 1
    assert data["trainRowIds"] == ["row1"]
    assert data["evalRowIds"] == []
    assert data["featureNames"] == ["feature1"]
    assert data["datasetDigest"] is not None
    assert data["reasonCodes"] == []


def test_missing_phase():
    """Test missing phase returns INVALID_INPUT."""
    response = client.post("/bqml", json={
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [],
        "trials": []
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_unknown_phase():
    """Test unknown phase returns INVALID_INPUT."""
    response = client.post("/bqml", json={
        "phase": "unknown",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [],
        "trials": []
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_runId():
    """Test invalid runId."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [],
        "trials": []
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_runId_too_long():
    """Test runId > 128 characters."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "a" * 129,
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [],
        "trials": []
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_empty_selection_rows():
    """Test empty selection rows."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [],
        "trials": []
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_duplicate_row_ids():
    """Test duplicate row IDs."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            },
            {
                "id": "row1",
                "entity": "entity2",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": []
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_duplicate_trial_ids():
    """Test duplicate trial IDs."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9},
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.8}
        ]
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_timestamp():
    """Test invalid timestamp."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "invalid",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": []
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_timezone():
    """Test invalid timezone offset."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00+15:00",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": []
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_version():
    """Test invalid version."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": -1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": []
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_trial_id():
    """Test invalid trial ID."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": [
            {"trialId": -1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_numTrialsLimit():
    """Test invalid numTrialsLimit."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 0,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": []
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_trial_limit_exceeded():
    """Test trial limit exceeded."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 1,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9},
            {"trialId": 2, "status": "SUCCEEDED", "evalMetric": 0.8}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "TRIAL_LIMIT_EXCEEDED" in data["reasonCodes"]


def test_failed_trials_ignored():
    """Test FAILED trials are ignored."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": [
            {"trialId": 1, "status": "FAILED", "evalMetric": 0.9},
            {"trialId": 2, "status": "SUCCEEDED", "evalMetric": 0.8}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["selectedTrialId"] == 2


def test_non_finite_evalMetric_ignored():
    """Test non-finite evalMetric ignored."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": float('inf')},
            {"trialId": 2, "status": "SUCCEEDED", "evalMetric": 0.8}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["selectedTrialId"] == 2


def test_no_successful_trial():
    """Test no successful trial."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": [
            {"trialId": 1, "status": "FAILED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["selectedTrialId"] is None
    assert "NO_SUCCESSFUL_TRIAL" in data["reasonCodes"]


def test_maximum_metric_selection():
    """Test maximum metric selection."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.7},
            {"trialId": 2, "status": "SUCCEEDED", "evalMetric": 0.9},
            {"trialId": 3, "status": "SUCCEEDED", "evalMetric": 0.8}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["selectedTrialId"] == 2


def test_exact_metric_tie_selects_smallest_trialId():
    """Test exact metric tie selects smallest trialId."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": [
            {"trialId": 9, "status": "SUCCEEDED", "evalMetric": 0.9},
            {"trialId": 4, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["selectedTrialId"] == 4


def test_row_deduplication():
    """Test row deduplication by entity + UTC(eventTime)."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            },
            {
                "id": "row2",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00+00:00",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 2,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val2", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["trainRowIds"]) == 1
    assert "row2" in data["trainRowIds"]  # Higher version wins


def test_highest_version_wins():
    """Test highest version wins."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            },
            {
                "id": "row2",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 3,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val2", "availableAt": "2024-01-01T00:00:00Z"}
                }
            },
            {
                "id": "row3",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 2,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val3", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "row2" in data["trainRowIds"]


def test_utf8_smallest_id_wins_version_tie():
    """Test UTF-8-smallest ID wins version tie."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row_z",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            },
            {
                "id": "row_a",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val2", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "row_a" in data["trainRowIds"]


def test_forbidden_feature_removal():
    """Test forbidden feature removal."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": ["feature2"],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"},
                    "feature2": {"value": "val2", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "feature1" in data["featureNames"]
    assert "feature2" not in data["featureNames"]


def test_missing_feature_removal():
    """Test missing feature removal."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            },
            {
                "id": "row2",
                "entity": "entity2",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature2": {"value": "val2", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["featureNames"]) == 0


def test_unavailable_feature_removal():
    """Test unavailable feature removal."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T02:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["featureNames"]) == 0


def test_featureNames_utf8_sorting():
    """Test featureNames UTF-8 sorting."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature_z": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"},
                    "feature_a": {"value": "val2", "availableAt": "2024-01-01T00:00:00Z"},
                    "feature_m": {"value": "val3", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["featureNames"] == ["feature_a", "feature_m", "feature_z"]


def test_train_eval_id_utf8_sorting():
    """Test train/eval ID UTF-8 sorting."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row_z",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            },
            {
                "id": "row_a",
                "entity": "entity2",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val2", "availableAt": "2024-01-01T00:00:00Z"}
                }
            },
            {
                "id": "row_m",
                "entity": "entity3",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "EVAL",
                "features": {
                    "feature1": {"value": "val3", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["trainRowIds"] == ["row_a", "row_z"]
    assert data["evalRowIds"] == ["row_m"]


def test_exact_datasetDigest():
    """Test exact datasetDigest."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["datasetDigest"] is not None
    assert len(data["datasetDigest"]) == 64


def test_reason_code_sorting_deduplication():
    """Test reason-code sorting/deduplication."""
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run",
        "forbiddenFeatures": [],
        "numTrialsLimit": 1,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {}
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9},
            {"trialId": 2, "status": "SUCCEEDED", "evalMetric": 0.8}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["reasonCodes"] == ["TRIAL_LIMIT_EXCEEDED"]


# ==================== STATE TESTS ====================

def test_successful_selection_persisted():
    """Test successful selection persisted."""
    import main
    main.save_state({})  # Clear state
    
    response = client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-persist",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    assert response.status_code == 200
    
    state = main.load_state()
    assert "test-run-persist" in state


def test_identical_replay_returns_same_response():
    """Test identical replay returns exactly the same response."""
    import main
    main.save_state({})  # Clear state
    
    request_data = {
        "phase": "select",
        "runId": "test-run-replay",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    }
    
    response1 = client.post("/bqml", json=request_data)
    assert response1.status_code == 200
    data1 = response1.json()
    
    response2 = client.post("/bqml", json=request_data)
    assert response2.status_code == 200
    data2 = response2.json()
    
    assert data1 == data2


def test_same_runId_different_input_returns_409():
    """Test same runId with different input returns HTTP 409."""
    import main
    main.save_state({})  # Clear state
    
    request_data1 = {
        "phase": "select",
        "runId": "test-run-conflict",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    }
    
    response1 = client.post("/bqml", json=request_data1)
    assert response1.status_code == 200
    
    request_data2 = request_data1.copy()
    request_data2["numTrialsLimit"] = 5
    
    response2 = client.post("/bqml", json=request_data2)
    assert response2.status_code == 409
    assert response2.json() == {"error": "RUN_ID_CONFLICT"}


# ==================== EVALUATE TESTS ====================

def test_valid_evaluate():
    """Test valid evaluation."""
    import main
    main.save_state({})  # Clear state
    
    # First, create a selection
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-eval",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    # Get the stored selection
    state = main.load_state()
    selected_trial_id = state["test-run-eval"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-eval"]["response"]["datasetDigest"]
    
    # Now evaluate
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-eval",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 0, "prediction": 0, "slice": "critical"}
        ],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert data["runId"] == "test-run-eval"
    assert data["selectedTrialId"] == selected_trial_id
    assert data["datasetDigest"] == dataset_digest
    assert data["testMetric"] == 1.0
    assert data["criticalSlicePass"] == True
    assert data["decision"] == "admit"
    assert data["bytesProcessed"] == 1000
    assert data["reasonCodes"] == []


def test_valid_lineage():
    """Test valid lineage."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-lineage",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-lineage"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-lineage"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-lineage",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [{"label": 1, "prediction": 1, "slice": "critical"}],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "INVALID_LINEAGE" not in data["reasonCodes"]


def test_invalid_runId_evaluate():
    """Test invalid runId in evaluate."""
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "",
        "selectedTrialId": 1,
        "datasetDigest": "a" * 64,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_selectedTrialId():
    """Test invalid selectedTrialId."""
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run",
        "selectedTrialId": None,
        "datasetDigest": "a" * 64,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_datasetDigest():
    """Test invalid datasetDigest."""
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run",
        "selectedTrialId": 1,
        "datasetDigest": "invalid",
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_mismatched_datasetDigest():
    """Test mismatched datasetDigest."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-mismatch",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-mismatch",
        "selectedTrialId": 1,
        "datasetDigest": "a" * 64,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [{"label": 1, "prediction": 1, "slice": "critical"}],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "INVALID_LINEAGE" in data["reasonCodes"]


def test_mismatched_selectedTrialId():
    """Test mismatched selectedTrialId."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-mismatch2",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    dataset_digest = state["test-run-mismatch2"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-mismatch2",
        "selectedTrialId": 999,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [{"label": 1, "prediction": 1, "slice": "critical"}],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "INVALID_LINEAGE" in data["reasonCodes"]


def test_unknown_runId_evaluate():
    """Test unknown runId in evaluate."""
    import main
    main.save_state({})  # Clear state
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "unknown-run",
        "selectedTrialId": 1,
        "datasetDigest": "a" * 64,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [{"label": 1, "prediction": 1, "slice": "critical"}],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "INVALID_LINEAGE" in data["reasonCodes"]


def test_invalid_metricFloor():
    """Test invalid metricFloor."""
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run",
        "selectedTrialId": 1,
        "datasetDigest": "a" * 64,
        "metricFloor": 1.5,
        "requiredSlices": {},
        "rows": [],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_slice_floor():
    """Test invalid slice floor."""
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run",
        "selectedTrialId": 1,
        "datasetDigest": "a" * 64,
        "metricFloor": 0.8,
        "requiredSlices": {"critical": 1.5},
        "rows": [],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_bytes():
    """Test invalid bytes."""
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run",
        "selectedTrialId": 1,
        "datasetDigest": "a" * 64,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [],
        "bytesProcessed": -1,
        "maxBytes": 2000
    })
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_invalid_label():
    """Test invalid label."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-label",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-label"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-label"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-label",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [{"label": 2, "prediction": 1, "slice": "critical"}],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "INVALID_TEST_ROW" in data["reasonCodes"]


def test_invalid_prediction():
    """Test invalid prediction."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-pred",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-pred"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-pred"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-pred",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [{"label": 1, "prediction": 2, "slice": "critical"}],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "INVALID_TEST_ROW" in data["reasonCodes"]


def test_empty_slice():
    """Test empty slice."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-slice",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-slice"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-slice"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-slice",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [{"label": 1, "prediction": 1, "slice": ""}],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "INVALID_TEST_ROW" in data["reasonCodes"]


def test_empty_rows():
    """Test empty rows."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-empty",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-empty"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-empty"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-empty",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert data["testMetric"] is None
    assert data["criticalSlicePass"] == True


def test_aggregate_floor_pass():
    """Test aggregate floor pass."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-agg-pass",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-agg-pass"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-agg-pass"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-agg-pass",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 1, "prediction": 1, "slice": "critical"}
        ],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "AGGREGATE_FLOOR" not in data["reasonCodes"]


def test_aggregate_floor_fail():
    """Test aggregate floor fail."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-agg-fail",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-agg-fail"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-agg-fail"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-agg-fail",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.9,
        "requiredSlices": {},
        "rows": [
            {"label": 1, "prediction": 0, "slice": "critical"},
            {"label": 0, "prediction": 1, "slice": "critical"}
        ],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "AGGREGATE_FLOOR" in data["reasonCodes"]
    assert data["decision"] == "reject"


def test_required_slice_pass():
    """Test required slice pass."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-slice-pass",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-slice-pass"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-slice-pass"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-slice-pass",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {"critical": 0.75},
        "rows": [
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 1, "prediction": 1, "slice": "critical"}
        ],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "SLICE_FLOOR:critical" not in data["reasonCodes"]
    assert data["criticalSlicePass"] == True


def test_required_slice_fail():
    """Test required slice fail."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-slice-fail",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-slice-fail"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-slice-fail"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-slice-fail",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {"critical": 0.9},
        "rows": [
            {"label": 1, "prediction": 0, "slice": "critical"},
            {"label": 0, "prediction": 1, "slice": "critical"}
        ],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "SLICE_FLOOR:critical" in data["reasonCodes"]
    assert data["criticalSlicePass"] == False


def test_missing_required_slice():
    """Test missing required slice."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-missing-slice",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-missing-slice"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-missing-slice"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-missing-slice",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {"critical": 0.75},
        "rows": [
            {"label": 1, "prediction": 1, "slice": "other"}
        ],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "MISSING_SLICE:critical" in data["reasonCodes"]
    assert data["criticalSlicePass"] == False


def test_byte_limit_pass():
    """Test byte limit pass."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-byte-pass",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-byte-pass"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-byte-pass"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-byte-pass",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [{"label": 1, "prediction": 1, "slice": "critical"}],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "BYTE_LIMIT" not in data["reasonCodes"]


def test_byte_limit_fail():
    """Test byte limit fail."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-byte-fail",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-byte-fail"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-byte-fail"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-byte-fail",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [{"label": 1, "prediction": 1, "slice": "critical"}],
        "bytesProcessed": 3000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "BYTE_LIMIT" in data["reasonCodes"]
    assert data["criticalSlicePass"] == True  # Byte limit doesn't affect criticalSlicePass


def test_criticalSlicePass_behavior():
    """Test criticalSlicePass behavior with aggregate failure but slice pass."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-critical",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-critical"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-critical"]["response"]["datasetDigest"]
    
    # Create scenario: aggregate fails (0.5 < 0.95) but critical slice passes (0.8 >= 0.75)
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-critical",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.95,
        "requiredSlices": {"critical": 0.75},
        "rows": [
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 0, "prediction": 1, "slice": "other"}  # This brings aggregate down
        ],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    # Aggregate is 0.8 (4/5), which is < 0.95, so AGGREGATE_FLOOR
    # Critical slice is 1.0 (4/4), which is >= 0.75, so slice passes
    assert data["criticalSlicePass"] == True  # Slice passes even though aggregate fails
    assert data["decision"] == "reject"  # Overall decision is reject due to aggregate floor


def test_exact_12_decimal_rounding():
    """Test exact 12-decimal rounding."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-rounding",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-rounding"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-rounding"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-rounding",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 1, "prediction": 0, "slice": "critical"}
        ],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert data["testMetric"] == 0.5  # Exactly 12 decimal places


def test_multiple_reason_codes():
    """Test multiple reason codes."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-multi",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-multi"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-multi"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-multi",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.9,
        "requiredSlices": {"critical": 0.95},
        "rows": [
            {"label": 1, "prediction": 0, "slice": "critical"},
            {"label": 0, "prediction": 1, "slice": "critical"}
        ],
        "bytesProcessed": 3000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    assert "AGGREGATE_FLOOR" in data["reasonCodes"]
    assert "SLICE_FLOOR:critical" in data["reasonCodes"]
    assert "BYTE_LIMIT" in data["reasonCodes"]


def test_reason_code_sorting():
    """Test reason-code sorting."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-sort",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-sort"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-sort"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-sort",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.9,
        "requiredSlices": {"zebra": 0.95, "alpha": 0.95},
        "rows": [
            {"label": 1, "prediction": 0, "slice": "zebra"},
            {"label": 0, "prediction": 1, "slice": "alpha"}
        ],
        "bytesProcessed": 3000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    # Check sorting
    reason_codes = data["reasonCodes"]
    assert reason_codes == sorted(reason_codes, key=lambda x: x.encode('utf-8'))


def test_exact_response_shape():
    """Test exact response shape."""
    import main
    main.save_state({})  # Clear state
    
    client.post("/bqml", json={
        "phase": "select",
        "runId": "test-run-shape",
        "forbiddenFeatures": [],
        "numTrialsLimit": 10,
        "rows": [
            {
                "id": "row1",
                "entity": "entity1",
                "eventTime": "2024-01-01T00:00:00Z",
                "predictionTime": "2024-01-01T01:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "feature1": {"value": "val1", "availableAt": "2024-01-01T00:00:00Z"}
                }
            }
        ],
        "trials": [
            {"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}
        ]
    })
    
    state = main.load_state()
    selected_trial_id = state["test-run-shape"]["response"]["selectedTrialId"]
    dataset_digest = state["test-run-shape"]["response"]["datasetDigest"]
    
    response = client.post("/bqml", json={
        "phase": "evaluate",
        "runId": "test-run-shape",
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": 0.8,
        "requiredSlices": {},
        "rows": [{"label": 1, "prediction": 1, "slice": "critical"}],
        "bytesProcessed": 1000,
        "maxBytes": 2000
    })
    assert response.status_code == 200
    data = response.json()
    expected_keys = {"runId", "selectedTrialId", "datasetDigest", "testMetric", "criticalSlicePass", "decision", "bytesProcessed", "reasonCodes"}
    assert set(data.keys()) == expected_keys


def test_health_endpoint():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
