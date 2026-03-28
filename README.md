# mlsys-notes

Learning notes and hands-on experiments for understanding modern Machine Learning System.

Currently focusing on LLM serving system and inference optimization.

## Layout

- `labs/`: Python source, scripts, and lab-local tests
- `notes/`: written notes and long-form explanations
- `assets/`: images and non-code research artifacts
- `pyproject.toml`, `.pre-commit-config.yaml`, `uv.lock`: project-level tooling at the repo root

Large experiment datasets live under `labs/data/` and are intentionally ignored by Git.

## Notes
- [Introduction to LLM Inference Part 1](notes/llm-inference-intro-p1.md)
- [ORCA paper review](notes/orca.md)
- [PagedAttention paper review](notes/pagedattention.md)
- [SARATHI Explained](notes/sarathi-explained.md)

## Labs
- [microengine](labs/microengine/README.md): a minimal serving engine
- [serving_bench](labs/serving_bench/README.md): serving benchmark harness
- [tinyORCA](labs/tinyorca/README.md): minimal end-to-end implementation of [ORCA](notes/orca.md)
