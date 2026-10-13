# Software checks

Automated tests remain under `tests/publication/`, grouped by workflow. They test software behavior on synthetic fixtures and do not constitute anatomical validation.

Run from the repository root with the top-level directory explicit, so the tests/publication namespace cannot shadow the runtime publication package:

```bash
python -m unittest discover -s tests -t . -p 'test_*.py'
```

Streamlit/openpyxl are needed for mask-review/cohort spreadsheet checks. Torch-dependent checks retain their existing skips when Torch is unavailable. Scientific images/checkpoints and real GPU/registration runs are not required for the lightweight suite.
