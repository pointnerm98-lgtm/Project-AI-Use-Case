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

---

## Step 2 — Training and test data for the intent classifier (2026-11-02)

### Decisions carried over from the step 1 review

- **`ask_precipitation` stays merged** (rain *and* snow in one intent). Splitting it
  would force the classifier to separate "will it rain" from "will it snow", but that
  distinction is mostly **seasonal and altitude-driven, not linguistic** — the same
  sentence means rain in July and snow in January. The weather service returns both
  values anyway, so the split belongs in the *response generator*, not in the NLU.
  Planned refinement: when the question explicitly mentions snow, the response
  generator reports snowfall first, otherwise rainfall first.
- **TO-DO for the app step:** `models/*.joblib` stays git-ignored, so the Streamlit app
  must **train the model automatically if the file is missing** (check path on startup,
  otherwise call the training routine once and cache it). This keeps the bot runnable
  straight out of the submitted ZIP without the tutor running a script by hand.
- README now documents both setup variants (Windows `py -3.12 -m venv .venv` and
  macOS/Linux `python3.12 -m venv .venv`), plus a warning that the bare `python`
  command on Windows can resolve to the Microsoft Store stub.

### Dataset design

- 8 intents x 40 training sentences = **320 training sentences**; 8 x 10 = **80 test
  sentences**. Perfectly balanced, so accuracy is directly interpretable and no class
  weighting is needed for the baseline.
- The test set was written **independently with fresh phrasings and mostly different
  places**. Verified programmatically: **0 overlap** between train and test, and 0
  duplicates inside either file. This matters because hand-written NLU data is easy to
  accidentally leak, which would inflate the reported accuracy.
- Variation was built in deliberately: one-word queries (`"temperature please"`) up to
  14-word sentences, casual (`"whats up with the weather in Obertauern"`) and formal
  (`"Could you tell me the expected temperature in Lienz on Saturday"`), with and
  without a place, with and without a day word.
- **Day and time words** cover `today`, `tomorrow`, `tonight`, `this morning/
  afternoon/evening`, all seven weekdays, `right now`, `this weekend` — the vocabulary
  the rule-based day parser will later have to resolve.
- **Typos** are included on purpose (`temperatur`, `tempature`, `clody`, `snwo`,
  `rainig`, `wheater`, `forcast`, `tomorow`, `wil`, `helo`, `tanks`, `thx`, `hoow`,
  `sunnyy`, `gooed`, and `whether` for `weather`). Real users on a phone at a trailhead
  mistype. Character n-grams in the vectorizer should absorb these — step 3 will show
  whether that holds.
- **ASCII spellings of umlauts** (`Soelden`, `Kitzbuehel`, `Grossglockner`) sit next to
  the correct forms, because users on a non-German keyboard type them that way.

### Avoiding a place-name shortcut

The risk with hand-written data is that the model learns a **place name as an intent
signal** — if "Gerlitzen" only ever appeared in wind questions, the classifier would
predict `ask_wind` for any sentence mentioning it. Austrian places (towns, ski resorts,
peaks, paragliding sites) were therefore spread across all intents on purpose, including
into `out_of_scope`.

Verified with a check script: of **71 distinct capitalised place tokens**, **not one**
occurs twice or more while staying inside a single intent. The most frequent names
(Innsbruck, Dachstein, Wildspitze, Ischgl, Salzburg, Kitzsteinhorn, Hochschwab,
Zell am See) each appear across **6 different intents**.

### Borderline cases and how they were resolved

| Case | Decision | Reasoning |
| --- | --- | --- |
| `"hows the weather"`, `"whats it like tomorrow"` | `ask_overview` | No single variable named. `ask_sky` is reserved strictly for sun, clouds, blue sky, fog, visibility and the view. |
| `"is the weather good for hiking on the Rax tomorrow"` | `ask_overview` | Asks for an overall judgement, not one value. |
| `"conditions on the Hochkönig for a ski tour tomorrow"` | `ask_overview` | Same — "conditions" is unspecific, even though a ski tour hints at snow. |
| `"will I see the stars from the Krippenstein tonight"` | `ask_sky` | Cloud cover question in disguise; no precipitation or wind asked. |
| `"is the Krippenstein above the inversion today"` | `ask_sky` | An inversion question is about fog and cloud layers, i.e. visibility. |
| `"how are the thermals at the Stoderzinken tomorrow"` | `ask_wind` | Thermals are an air-movement question; paragliders phrase wind this way. |
| `"foehn wind in Innsbruck tomorrow"` | `ask_wind` | Named wind phenomenon, not a general forecast. |
| `"do I need a rain jacket in Salzburg today"` | `ask_precipitation` | Equipment phrasing, but the underlying variable is clearly rain. |
| `"how cold is it on the Hochkönig tomorrow I need to pick a jacket"` | `ask_temperature` | Same pattern: the named variable wins over the equipment framing. |
| `"will the trail be wet on the Rax tomorrow"` | `ask_precipitation` | Wet ground is a consequence of rain; no other variable involved. |
| **Avalanche questions** (`"what is the avalanche risk on the Dachstein"`, `"whats the avalanche bulletin for Tirol"`) | `out_of_scope` | Deliberate. The concept states PeakCast **does not replace official mountain weather or avalanche reports**. Routing these to a weather answer would be the single most safety-relevant failure the bot could make, so they must be refused, not answered. |
| **Mountain infrastructure** (`"is the cable car open on the Kitzsteinhorn"`, `"is there a hut open near the Hochschwab"`, `"are the lifts running at the Turracher Höhe"`) | `out_of_scope` | Mountain context and a place name, but no weather variable. These are the hardest negatives in the set: they look exactly like weather questions to a bag-of-words model. |
| **Mountain facts** (`"what is the elevation of the Wildspitze"`, `"how high is the Dachstein"`, `"how long is the hiking trail to the Grimming"`) | `out_of_scope` | Static geography, not a forecast. |
| `"do I need crampons for the Großglockner"` | `out_of_scope` | Gear advice. It *depends* on conditions, but answering it would mean giving mountaineering advice the bot is not scoped for. |

The infrastructure and avalanche cases are the ones to watch in the step 3 confusion
matrix: if anything leaks out of `out_of_scope`, it will be these.

### Results

- `data/intents_train.csv` — 320 rows, `data/intents_test.csv` — 80 rows, columns
  `text,intent`, UTF-8, no missing values.
- Sentence length: mean 5.8 words, median 6, range 1–14.
- Not yet trained — data handed over for review first.
