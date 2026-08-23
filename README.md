# Deterministic Model-Registry Promotion Gate

A stateless FastAPI service that decides whether a challenger model version may
replace the current champion in a model registry. Every decision is derived
purely from the request body: no database, no external calls, no wall-clock
time and no randomness. Identical JSON input always produces identical JSON
output.

## API

### `POST /promote`

Request (`application/json`):

```json
{
  "asOf": "2026-08-23T10:00:00Z",
  "championVersion": "1",
  "policy": {
    "datasetDigest": "dataset-abc",
    "schemaDigest": "schema-abc",
    "maxAgeSeconds": 3600,
    "accuracyFloor": 0.8,
    "requiredSlices": { "critical": 0.75 },
    "maxLatencyMs": 100,
    "maxSizeBytes": 1000000,
    "minImprovement": 0.01
  },
  "versions": [
    {
      "version": "1",
      "artifactDigest": "artifact-1",
      "tags": {},
      "evaluation": {
        "createdAt": "2026-08-23T09:59:00Z",
        "artifactDigest": "artifact-1",
        "datasetDigest": "dataset-abc",
        "schemaDigest": "schema-abc",
        "accuracy": 0.9,
        "latencyMs": 50,
        "sizeBytes": 500000,
        "slices": { "critical": 0.85 }
      }
    },
    {
      "version": "2",
      "artifactDigest": "artifact-2",
      "tags": {},
      "evaluation": {
        "createdAt": "2026-08-23T09:59:00Z",
        "artifactDigest": "artifact-2",
        "datasetDigest": "dataset-abc",
        "schemaDigest": "schema-abc",
        "accuracy": 0.95,
        "latencyMs": 50,
        "sizeBytes": 500000,
        "slices": { "critical": 0.9 }
      }
    }
  ]
}
```

Response (`200`):

```json
{
  "action": "promote",
  "championVersion": "1",
  "selectedVersion": "2",
  "eligibleVersions": ["2", "1"],
  "failedGates": {},
  "aliasMutation": { "alias": "champion", "version": "2" },
  "evidence": {
    "createdAt": "2026-08-23T09:59:00Z",
    "artifactDigest": "artifact-2",
    "datasetDigest": "dataset-abc",
    "schemaDigest": "schema-abc",
    "accuracy": 0.95,
    "latencyMs": 50,
    "sizeBytes": 500000,
    "slices": { "critical": 0.9 }
  }
}
```

Malformed top-level requests (missing `policy`, non-array `versions`,
non-string `championVersion`, invalid `asOf`, non-JSON body) return `400` with
exactly:

```json
{"error":"INVALID_INPUT"}
```

### Decision rules

* `action` is `promote`, `retain` or `block`.
* Versions failing any gate are ineligible; `failedGates` lists every input
  version with at least one failure, with unique, code-point-sorted codes.
* `eligibleVersions` is ordered by accuracy desc, latency asc, size asc, then
  numeric version asc — the same ranking used to pick `selectedVersion`.
* The champion must itself be eligible; otherwise the result is `block` with
  `selectedVersion: null` and `evidence: null`, even when challengers are
  eligible.
* Promotion requires `round(selected.accuracy - champion.accuracy, 12) >=
  policy.minImprovement` (computed with decimal arithmetic).
* `aliasMutation` is only present for `promote`. Replaying the same request
  with the promoted version as `championVersion` returns `retain`.
* `evidence` is the selected version's complete, unmodified evaluation object.
  Tags and descriptions are never treated as evidence.

### Gate codes

| Code | Meaning |
| --- | --- |
| `INVALID_VERSION` | Version ID is not a canonical positive safe integer string |
| `DUPLICATE_VERSION` | Version ID appears more than once (every occurrence fails) |
| `INVALID_POLICY` | The policy object failed validation |
| `MISSING_EVALUATION` | No evaluation object on the version |
| `NON_FINITE` | `accuracy`, `latencyMs` or `sizeBytes` is NaN or ±Infinity |
| `METRIC_RANGE` | Metric missing, non-numeric or out of its allowed range |
| `ARTIFACT_MISMATCH` | Evaluation `artifactDigest` differs from the version's |
| `DATASET_MISMATCH` | Evaluation `datasetDigest` differs from the policy's |
| `SCHEMA_MISMATCH` | Evaluation `schemaDigest` differs from the policy's |
| `INVALID_TIMESTAMP` | `createdAt` is not a supported instant format |
| `FUTURE_EVALUATION` | `createdAt` is after `asOf` (takes precedence over staleness) |
| `STALE_EVALUATION` | `createdAt` is older than `asOf - maxAgeSeconds` |
| `MISSING_SLICE:<name>` | A required slice is absent from the evaluation |
| `SLICE_RANGE:<name>` | A required slice value is not finite within `[0,1]` |
| `SLICE_FLOOR:<name>` | A required slice is below its floor |
| `ACCURACY_FLOOR` | `accuracy < policy.accuracyFloor` |
| `LATENCY_LIMIT` | `latencyMs > policy.maxLatencyMs` |
| `SIZE_LIMIT` | `sizeBytes > policy.maxSizeBytes` |

Accepted timestamp format: `YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)` with 1–3
optional fractional digits.

### `GET /health` and `GET /`

Health checks returning `{"status": "ok"}` with HTTP 200. Render's health check
path is configured to `/health`.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the server (defaults to port `10000`, override with `PORT`):

```bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
```

Test with curl:

```bash
curl -s -X POST http://localhost:10000/promote \
  -H 'content-type: application/json' \
  -d '{"asOf":"2026-08-23T10:00:00Z","championVersion":"1","policy":{"datasetDigest":"dataset-abc","schemaDigest":"schema-abc","maxAgeSeconds":3600,"accuracyFloor":0.8,"requiredSlices":{"critical":0.75},"maxLatencyMs":100,"maxSizeBytes":1000000,"minImprovement":0.01},"versions":[{"version":"1","artifactDigest":"artifact-1","tags":{},"evaluation":{"createdAt":"2026-08-23T09:59:00Z","artifactDigest":"artifact-1","datasetDigest":"dataset-abc","schemaDigest":"schema-abc","accuracy":0.9,"latencyMs":50,"sizeBytes":500000,"slices":{"critical":0.85}}}]}'
```

Run the tests:

```bash
pytest -q
```

## Docker

```bash
docker build -t promotion-gate .
docker run --rm -p 10000:10000 promotion-gate
```

## Render

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

`render.yaml` configures the web service with `healthCheckPath: /health`. No
disk, database or environment variables are required.
