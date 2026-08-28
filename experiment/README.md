# Unified experiment evaluation

Raw evaluation records are append-only JSONL. Dense float32 residual activations
are stored in safetensors sidecars referenced by the JSONL records. Derived CSVs
are generated only for analysis and figures.

Run the complete Gate 1/2 measurement bundle, one memory-bounded model process
at a time:

```zsh
cd /Users/satwikpandey/Dev/unlearn-fr
caffeinate -dimsu env PYTHONPATH=experiment/src .venv/bin/python -u \
  experiment/src/gates.py run
```

Run or resume individual jobs:

```zsh
PYTHONPATH=experiment/src .venv/bin/python -u \
  experiment/src/gates.py run full
```

Assess the gates without loading model weights:

```zsh
PYTHONPATH=experiment/src .venv/bin/python \
  experiment/src/gates.py assess --root .
```

Every job resumes by unique row ID and refuses to append if its frozen
configuration differs. Confirmation-author activations are not cached during
the gate pipeline; only frozen discovery-author activations are written.

`gates.py` is the sole Gate 1/2 command surface. The `evaluate.py` module only
derives tidy scalar tables from saved JSONL; it does not rerun model inference.
Each main state writes one canonical JSONL containing all requested evidence;
the older split-job artifacts in `artifacts/state_eval/` are retained only as
provenance and are ignored by gate assessment.
