# PeakCast — Development Log

Running diary of Phase 2 (implementation). One section per working step, with the
decisions taken, problems hit and results measured. Used as the source for the
presentation slides and the final "making of" report.

---

## Step 1 — Project setup (2026-11-02)

### Decisions

- **Python 3.12.10** chosen as the target interpreter. The machine only had Python
  3.14.3 installed, which is outside the 3.10–3.12 range of the concept. Decided to
  install 3.12 via `winget` rather than use 3.14, because the planned comparison
  against sentence embeddings needs `torch`, which is the dependency most likely to
  lack wheels for a very new Python version. Avoiding that blocker now was cheaper
  than hitting it later.
- **Exact version pins** (`==`) in `requirements.txt` instead of minimum versions.
  The portfolio exam requires that the tutor can rebuild the project from the
  submitted files alone, so the environment has to be deterministic.
- **Only current dependencies** are listed: `scikit-learn`, `pandas`, `joblib`,
  `matplotlib`. `streamlit` and `requests` are added in the step where they are
  actually used, so every dependency in the file is traceable to a feature.
- **`models/*.joblib` is git-ignored.** The model is a build artefact that
  `scripts/train_intent.py` regenerates in seconds; keeping binaries out of the
  repository keeps the diff readable. `reports/` *is* tracked, because those metrics
  are evidence for the presentation.

### Problems

- `python --version` resolved to the Microsoft Store stub (`python3.exe` in
  `WindowsApps`) and reported nothing useful. Used the Python launcher `py --list`
  instead, which reliably lists the real installations.
- The concept PDF could not be rendered page-by-page (no poppler on the machine).
  Extracted the text with a short stdlib-only script (`zlib` + regex over the PDF
  content streams) to confirm the implementation matches the approved concept.

### Results

- Virtual environment created at `.venv` on Python 3.12.10, pip 26.2.2.
- Resolved and pinned: scikit-learn 1.9.1, pandas 3.1.0, joblib 1.6.0,
  matplotlib 3.11.2.
- Folder structure created: `data/`, `models/`, `reports/`, `scripts/`, `peakcast/`.
- Git repository initialised with a first commit.

### Detail re-read from the concept that shapes later steps

- Wind **gusts**, not just mean wind speed, are a mandatory part of the answer —
  they decide whether a paraglider launches.
- Time words include **"afternoon"**, not only "today"/"tomorrow"/weekdays.
- The dialogue manager must **remember the last place** so that follow-ups like
  *"And what about tomorrow?"* work, and must end the conversation after a goodbye
  or roughly ten answers.
