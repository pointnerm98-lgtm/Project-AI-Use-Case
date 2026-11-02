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

### Safety TO-DO (decided during the step 2 review)

**The classifier must not be the only line of defence for avalanche questions.** Two
hand-written `out_of_scope` examples cannot cover the phrasing space, and a sentence
like *"avalanche risk after the fresh snow"* contains strong `ask_precipitation`
vocabulary ("fresh snow") that would very likely outweigh the single token "avalanche".
A statistical model gives no guarantee here, and this is the one error class with a
real-world safety consequence.

**Required for the dialogue manager step:** a **rule-based safety guard that runs
BEFORE the intent classifier**. If the message matches `avalanche` / `lawine` or close
variants (including `avalanch`, `avalance`, `lawinen-`, `lawinengefahr`, and the
abbreviation `LWD`), the bot skips classification entirely and returns a **fixed
referral** to the official avalanche warning service (avalanche.report / the regional
Lawinenwarndienst), with no forecast values attached.

Design notes:
- Order matters: guard first, classifier second. A guard behind the classifier would
  only ever see messages the model already mislabelled.
- Matching is done on the lower-cased message with a tolerant pattern, not on exact
  tokens, so `"Avalanche-Risk"` and `"lawinengefahr"` are caught too.
- The avalanche examples **stay in `out_of_scope`** regardless. The guard and the
  training data are two independent layers: if the guard pattern ever misses a variant,
  the classifier should still not route it to a weather answer.
- This is a good slide: it shows where a rule beats a learned model, and why a safety
  requirement is an architecture decision rather than a data problem.

---

## Step 3 — Training and evaluating the baseline intent model (2026-11-02)

### Code structure decision

The shared logic went into **`peakcast/intent_classifier.py`** instead of being copied
into the three scripts. Reasons: the pipeline is defined exactly **once**, so training
and evaluation can never drift apart and invalidate the reported metrics; and the
dialogue manager and Streamlit app will import the same functions later —
`train_and_save()` is already written so the app can satisfy the
"train automatically if the model file is missing" to-do. The scripts in `scripts/`
are thin runners around it.

### Results of the baseline

| Metric | Value |
| --- | --- |
| 5-fold cross-validation accuracy (train) | **0.678** (std 0.069) |
| Test accuracy | **0.775** (62/80) |
| Macro F1 (test) | 0.778 |
| TF-IDF features | 1298 |

Per-intent F1 on the test set: `goodbye` 0.95, `ask_overview` 0.84, `greet` 0.84,
`ask_sky` 0.82, `ask_precipitation` 0.74, `ask_wind` 0.70, `ask_temperature` 0.67,
`out_of_scope` 0.67.

**Read per-intent numbers with care.** With only 10 test sentences per intent, a single
error moves that intent's recall by a full 10 percentage points, so the gap between
`ask_sky` (0.90 recall) and `ask_wind` (0.70) is two sentences, not a stable property of
the model. The cross-validation mean is the more trustworthy figure, because it averages
over 320 sentences in five different splits. Note that test accuracy (0.775) sits *above*
the CV mean (0.678) — with samples this small, that is sampling noise, not evidence that
the model generalises better than it trains.

### Problem found: the baseline cannot support a confidence threshold

The task was to recommend a probability below which the bot says it did not understand.
Measuring it exposed a defect in the baseline rather than a threshold choice:

- The most confident *correct* prediction on the whole test set reaches only **p = 0.35**.
- Mean confidence is **0.345 for correct** and **0.223 for wrong** predictions — barely
  separated, and both close to the 1/8 = 0.125 of pure guessing.

Threshold analysis on out-of-fold cross-validation predictions (deliberately **not** on
the test set, because a threshold is a tuned parameter and tuning it on test data would
make the reported test accuracy optimistic):

| threshold | errors caught | correct answers lost |
| --- | --- | --- |
| 0.20 | 52 % | 13 % |
| 0.25 | 85 % | 42 % |
| 0.30 | 98 % | 60 % |
| 0.40 | 100 % | 79 % |

There is no usable operating point: intercepting most errors means rejecting roughly
half of the answers the bot got right. **The cause is the model, not the threshold.**
`LogisticRegression` defaults to `C=1.0`, which over-regularises L2-normalised TF-IDF
vectors of very short sentences; all weights are shrunk toward zero and the softmax
output stays nearly flat.

`CONFIDENCE_THRESHOLD = 0.25` is set in `peakcast/intent_classifier.py` as a
**provisional** value so the fallback path can be built, and flagged there for
re-derivation.

### Diagnosis: what actually fixes it (measured, cross-validated, not yet adopted)

| Features | C | CV accuracy | mean p correct | mean p wrong |
| --- | --- | --- | --- | --- |
| word 1-2 grams (**current baseline**) | 1 | 0.678 | 0.31 | 0.20 |
| word 1-2 grams | 10 | 0.706 | 0.60 | 0.35 |
| word + char_wb 3-5 grams | 1 | 0.781 | 0.42 | 0.25 |
| **word + char_wb 3-5 grams** | **3** | **0.791** | **0.56** | **0.31** |

Two independent findings:

1. **Character n-grams are worth ~+11 accuracy points** (0.678 → 0.791). This confirms
   the prediction written down in step 2: the deliberate typos (`sunnyy`, `tomorow`,
   `wheater`, `gday`, `hiya`) share no *word* with any training sentence, so a word-level
   vectorizer sees them as entirely unknown input. Character n-grams still match
   `sunny`/`sunnyy`. Four of the 18 test errors are exactly such typo or short-form cases.
2. **Raising `C` to 3 roughly doubles the confidence of correct predictions**, which is
   what makes a threshold possible at all.

The baseline was **kept unchanged on purpose** — `reports/metrics_baseline.json` is the
reference point the next iteration has to beat, and the whole point of the portfolio is
to show the optimisation path rather than a single tuned number.

### Error analysis (all 18 errors in `reports/misclassified_baseline.csv`)

- Confusions are spread thinly across the weather intents; no systematic pair dominates.
  The largest single cell is `ask_wind` → `ask_sky` with 2.
- The step 2 prediction that `out_of_scope` would leak was **half right**. It has the
  joint-lowest F1 (0.67), and `"are the lifts running at the Turracher Höhe"` did leak to
  `ask_wind` as expected. But the other two errors were *not* the mountain-infrastructure
  cases: `"how many people live in Linz"` and `"how high is the Dachstein"` both went to
  `ask_temperature`, apparently on the quantity phrasing "how many"/"how high".
- `greet` lost `"gday"` and `"hiya"` to `out_of_scope` — both unknown tokens at word
  level, i.e. the character n-gram finding again.
- The safety case was reproduced by hand in `try_intent.py`:
  **`"avalanche risk after the fresh snow"` → `ask_precipitation` with p = 0.252**,
  *above* the provisional threshold. The bot would answer it confidently with a snow
  forecast. This is direct empirical evidence for the rule-based avalanche guard
  specified in the step 2 safety to-do — a statistical classifier will not reliably
  catch this, and the fallback threshold does not catch it either.

### Artefacts produced

- `models/intent_model.joblib` (git-ignored, regenerated by the training script)
- `reports/confusion_matrix.png`
- `reports/metrics_baseline.json`
- `reports/misclassified_baseline.csv` (18 rows, with the probability of both the
  predicted and the true intent, so near-misses can be told from confident errors)
