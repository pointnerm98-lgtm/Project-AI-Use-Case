# PeakCast — Development Log

Running diary of Phase 2 (implementation). One section per working step, with the
decisions taken, problems hit and results measured. Used as the source for the
presentation slides and the final "making of" report.

---

## Step 1 — Project setup (2026-11-02)

### Decisions

- **Python 3.12.10** chosen as the target interpreter. The machine only had Python
  3.14.3 installed, which is outside the 3.10–3.12 range of the concept. Decided to
  install 3.12 via `winget` rather than use 3.14, because an optional later comparison
  against sentence embeddings would need `torch`, which is the dependency most likely to
  lack wheels for a very new Python version. Avoiding that blocker now was cheaper
  than hitting it later. (Correction made in step 5: that comparison is **not** part of
  the submitted concept — see the step 5 scope correction. Staying on 3.12 is still the
  right call, because it is the range the concept names.)
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
   `sunny`/`sunnyy`. (Correction made in step 4: the first draft of this entry claimed
   "four of the 18 test errors" were typo cases. Measured properly against the
   baseline's own vocabulary, **15 of the 18** errors contain at least one token the
   baseline had never seen — the problem is unknown words in general, of which the
   deliberate typos are only one kind. See step 4.)
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

---

## Step 4 — Iteration 1: grid search, better features, usable threshold (2026-11-03)

### What was changed and why

The step 3 baseline had two separate defects: it could not read misspelled or unseen
words, and its probabilities were too compressed to support a fallback threshold.
Iteration 1 addresses both, and **every choice was made on cross-validation** — the
test set was evaluated exactly once, after the model was already fixed.

- **Features:** a `FeatureUnion` of the original word 1-2 grams **and** `char_wb`
  character n-grams. `char_wb` keeps n-grams inside word boundaries, so it never
  creates a feature spanning two unrelated words.
- **Hyper-parameters:** `GridSearchCV` over `C ∈ {0.3, 1, 3, 10, 30}` and
  `char_ngram_range ∈ {(2,4), (3,5), (2,5)}` — 15 combinations, 5-fold stratified
  shuffled CV with the project seed, 75 fits. The grid was kept small on purpose:
  with 320 sentences, a large grid mostly fits noise and makes the winner more likely
  to be a lucky split.
- **Winner:** `C = 10.0`, `char_ngram_range = (2, 4)`, mean CV accuracy **0.8031**.
  Full grid in `reports/grid_search_iter1.csv`.
- Note how flat the top of the grid is: ranks 2–5 all sit at 0.7969, within one
  standard deviation (~0.058) of the winner. The honest reading is "word+char with
  C somewhere between 3 and 30", not "C=10 with a (2,4) range is optimal".

### Results

| Metric | Baseline | Iteration 1 | Change |
| --- | --- | --- | --- |
| Cross-validation accuracy (train) | 0.6781 | **0.8031** | **+12.5 pts** |
| Test accuracy | 0.7750 | **0.8125** | +3.7 pts |
| Macro F1 (test) | 0.7775 | **0.8118** | +3.4 pts |
| Mean confidence of correct answers | 0.3451 | **0.7354** | +39.0 pts |
| Test errors | 18 / 80 | **15 / 80** | −3 |
| Features | 1 298 | 4 313 | ×3.3 |

Chart for the slide: `reports/comparison_baseline_vs_iter1.png`.

**The CV gain (+12.5) is the trustworthy number; the test gain (+3.7) is three
sentences.** Both are reported because they disagree in size, and that disagreement is
itself the lesson: 80 test sentences cannot resolve a difference this small. The
confidence gain is the largest and the most consequential one in practice, because it
is what makes a fallback possible at all.

### The threshold, re-derived

Re-derived from out-of-fold CV predictions of the *new* model (same method as step 3,
still never on the test set):

| threshold | errors caught | correct answers lost |
| --- | --- | --- |
| 0.30 | 20.6 % | 4.7 % |
| **0.35** | **42.9 %** | **7.8 %** |
| 0.40 | 58.7 % | 15.6 % |
| 0.45 | 77.8 % | 19.8 % |

`CONFIDENCE_THRESHOLD` is now **0.35** (was a provisional 0.25). It is the knee of the
curve: moving to 0.40 catches 10 more errors but sacrifices 20 more correct answers,
and 0.45 would make the bot ask back on roughly a third of all messages — unusable for
someone standing at a trailhead. The reasoning is written into the constant's comment
in `peakcast/intent_classifier.py` so it does not have to be rediscovered later.

For contrast, the baseline had **no** usable operating point at all.

### Spot-check 1 — the avalanche sentence got *worse*

| | baseline | iteration 1 |
| --- | --- | --- |
| `"avalanche risk after the fresh snow"` | `ask_precipitation` p = 0.25 | `ask_precipitation` **p = 0.62** |

Still wrong, and now **confidently** wrong: 0.62 is far above the 0.35 fallback
threshold, so the improved bot would answer a safety question with a snow forecast
without hesitating, where the baseline at least sat near the uncertainty region.

This is the key finding of the step. **A better classifier made the safety case
worse**, because "fresh snow" is strong precipitation evidence and the model has no
concept of what the question is *for*. It confirms that the rule-based avalanche guard
specified in the step 2 safety to-do is not a nice-to-have and cannot be replaced by
either a better model or a confidence threshold. Good slide material: accuracy went up,
safety went down.

### Spot-check 2 — unseen words, measured rather than assumed

Checked against the baseline's own vocabulary using its own preprocessor and tokenizer:
**15 of the 18 baseline errors contain at least one token the baseline had never seen**
(`sunnyy`, `gday`, `hiya`, but also `thermometer`, `drizzling`, `umbrella`, `powder`,
`lifts`, and many place names). So the step 3 framing was too narrow — the weakness was
unknown words in general, and the deliberate typos are just the most visible kind.

How iteration 1 handled those 15:

- **Fixed outright:** `"will it be sunnyy at the Katschberg on Friday"` → `ask_sky`
  0.41 (was `ask_precipitation`), and `"hiya"` → `greet` 0.42 (was `out_of_scope`).
  Both are exactly the character n-gram effect that was predicted in step 2.
- **Still wrong but now safely rejected:** 5 of the remaining errors fall below the
  0.35 threshold, so the bot asks back instead of answering. `"gday"` is one of them —
  it moved to a different wrong intent but at p = 0.20, below the threshold.
- **Still wrong and now more confident:** `"how high is the Dachstein"` → 0.62,
  `"have a good evening"` → `greet` 0.80. Higher confidence cuts both ways: when the
  model is wrong, the fallback no longer rescues it.

### Code structure

- `build_pipeline()` (baseline) is **unchanged and frozen** as the reference point;
  `build_pipeline_iter1()` is a separate function. Re-running step 3 reproduces
  0.6781 CV / 0.7750 test exactly, which was verified after the refactor.
- `scripts/evaluate_intent.py` now takes `--variant baseline|iter1`. Both go through
  one code path, so the two metric files are genuinely comparable rather than
  produced by two drifting copies of the same code.
- `analyse_thresholds()` moved into `peakcast/intent_classifier.py`, next to the
  constant it derives.
- `ACTIVE_MODEL_PATH` names the model the bot should use, so the dialogue manager will
  not need to know which iteration is currently best.

### Open points for the next step

- The remaining errors are mostly **genuinely ambiguous phrasings**
  (`"what kind of day is it in Hermagor"`, `"is it a nice day in Dornbirn today"`,
  `"do I need an umbrella in Gmunden today"`) rather than vocabulary gaps. More
  training sentences in those styles would help more than more hyper-parameter search.
- `out_of_scope` remains the weakest intent (F1 0.70) and the mountain-infrastructure
  case `"are the lifts running at the Turracher Höhe"` still leaks to `ask_wind`.
- A comparison against **sentence embeddings** would be a possible iteration 2.
  (Corrected in step 5: this is **optional**, not a deliverable — see the step 5
  scope correction.)

---

## Step 5 — Safety guard, place list, and slot extraction (2026-11-04)

### Two scope corrections made at the start of this step

- **Sentence embeddings are not a deliverable.** Earlier entries called the
  comparison against sentence embeddings "planned" and "the natural iteration 2".
  That was wrong: it is **not** part of the concept actually submitted
  (`docs/PeakCast Wheater Chatbot - Concept.pdf` is the submitted version). It is
  **optional** and will only be done if time remains after the bot works end to
  end. The two affected entries (steps 1 and 4) were corrected in place rather
  than silently edited, so the change of plan stays visible.
- **New training data must not be modelled on test-set errors.** The step 4 open
  points suggested adding training sentences in the styles the model got wrong
  (`"what kind of day is it in Hermagor"`, `"is it a nice day in Dornbirn today"`).
  Doing that would be **test-set leakage**: the test set would stop being an
  unbiased estimate and every later number would be optimistic, because the
  training data would have been shaped by the answers. For any later data
  iteration, new phrasings get collected from **real test conversations with the
  running bot**, not from `reports/misclassified_*.csv`.

### 5a — The avalanche safety guard

`peakcast/safety_guard.py`, a plain function with no model and no state. The
design decisions:

- **Order: guard first, classifier second.** A guard running after the classifier
  would only ever see the messages the model already mislabelled — exactly the
  set it cannot be relied on to produce correctly.
- **Tolerant matching, deliberately.** Substring matching on a normalised message
  (lower-cased, accents stripped, punctuation to spaces) plus an explicit
  misspelling list. The asymmetry justifies erring towards catching: a false
  positive sends someone on a winter mountain to the official avalanche bulletin,
  which is good advice anyway, while a false negative is the one failure this
  project must not have. German compounds come free from the substring approach —
  the stem `lawin` covers *Lawine*, *Lawinengefahr*, *Lawinenlage*,
  *Lawinenwarnstufe* and *lawinengefährlich*.
- **No forecast values in the referral**, not even ones the bot could legitimately
  produce. Mixing "20 cm of fresh snow" into a safety referral would undo the
  point of referring. There is a test asserting the referral string contains no
  units.
- `LWD` and `slab` are matched as **whole words** only; three-letter substrings
  would fire inside unrelated words.

**37 unit tests**, including the named case the classifier provably gets wrong.

### 5b — The place list from GeoNames

`scripts/build_places.py` downloads `AT.zip` (52 912 rows) and writes the compact
`data/places_at.csv` (32 958 rows, 2.2 MB). The raw dump is git-ignored; the
compact file is committed so the bot runs from the submitted ZIP without a
download. GeoNames CC BY attribution added to the README.

**Feature codes were chosen from the data, not from the GeoNames code list.**
Measuring which codes are needed to cover the place names that actually occur in
`data/intents_*.csv` gave the set: populated places (`PPL*`), terrain (`PK`, `MT`,
`MTS`, `RGN`, `HLL`, `PASS`, `SDL`, `RDGE`, `VAL`, `CRQ`, `SLP`), water and ice
(`GLCR`, `LK`) and mountain huts (`HUT`). Huts earn their place because a hut is
very often *the* destination of a tour, and some well-known paragliding launch
sites are mapped only as huts (*Emberger Alpe*).

**The administrative divisions are excluded on purpose**, and this is the single
most consequential decision of the step. `ADM2`/`ADM3` rows look attractive
because they carry the municipality population, but:

| | code | population | elevation |
| --- | --- | --- | --- |
| the municipality of Sölden | ADM3 | 3 145 | **2 856 m** |
| the village of Sölden | PPLA3 | 2 205 | **1 368 m** |
| *Politischer Bezirk* Zell am See | ADM2 | 86 923 | 761 m |
| the town of Zell am See | PPLA3 | 7 619 | 764 m |

The ADM row's coordinates and elevation describe an *area*, so a forecast fetched
for them can sit ~1 500 m above the village the user meant — a different
temperature, a different snow line, a different answer. And because ambiguous
names are resolved by population, the district would have outranked the town on
every single lookup. For a *mountain* weather bot, elevation is not a detail.

#### Two bugs found by measuring instead of assuming

1. **Blind truncation of the alternate-name field threw away the names that
   matter.** The field is alphabetically sorted, so keeping "the first 6 Latin
   names" gave Vienna → `Bec, Bech, Becs, Beç, Beč, Bienne` and discarded **"Wien"
   at position 40**. Fixed by ranking candidates by **similarity to the primary
   name** before applying the cap: the spelling variants a user actually types
   (`Soelden`, `Solden`, `Sölden` all score 1.00 after normalisation) now survive
   and the foreign exonyms are pushed down. Similarity cannot rescue "Wien"
   itself (it scores 0.60 against "Vienna", below several Slavic
   transliterations, and the country file carries no language codes), so the few
   such cases are listed explicitly in `EXTRA_NAME_ALIASES` — as *extra spellings
   for existing rows*, never as new places, so an alias can never point a
   forecast at unverified coordinates.

2. **Dropping rows without a federal state deleted some of the best-known
   mountains in Austria.** GeoNames uses admin1 `00` for features that do not lie
   inside one single state, and the first version dropped all 277 of them —
   including the **Raxalpe**, the **Hohe Tauern**, the **Totes Gebirge**, the
   **Rätikon**, the **Lechtaler Alpen** and the **Wolfgangsee**. Now kept with an
   empty state; `Place.label` degrades to the bare name, which costs only the
   disambiguation hint in the answer.

### 5c — Place extraction

`peakcast/slots.py`. Normalisation collapses case, `ß`→`ss`, umlauts→bare vowels,
the `ae`/`oe`/`ue` digraphs→bare vowels, and `Sankt`/`St.`→`st`. Both the
gazetteer names and the user message go through the *same* function, so the three
spellings a user might type (`Sölden`, `Soelden`, `Solden`) meet on one key.

Matching runs in three stages, exact → partial → fuzzy:

**Stage 1, longest match wins.** Not leftmost-then-longest — the difference is a
real bug. `"forecast for Sankt Anton am Arlberg"` contains the window
`for` + `st` (from *Sankt*), and removing its space gives `forst` — the village of
**Forst**, which a leftmost-first scan returned without ever looking at the
four-token `st anton am arlberg` further right. Two changes fixed it: collect all
matching windows and take the longest, and apply the space-free fallback only
when no token in the window is an ordinary English word.

The space-free fallback itself is needed because GeoNames is inconsistent about
compounds: the pass everyone writes *Turracher Höhe* is stored as the single word
*Turracherhöhe*.

**Stage 2, partial names — added after measuring, and it replaced work that fuzzy
matching was doing badly.** Users shorten compound mountain names while GeoNames
stores the long form. On the 400 intent sentences, fuzzy matching resolved
`Bischling` → **Bichling** and `Sonnwendjoch` → **Sonnenjoch**: *different real
places*, which is the worst kind of error because the answer looks entirely
plausible. Word-boundary containment (with a minimum length, a coverage
requirement, and the shortest matching name winning) gets all of them right:

| typed | fuzzy matching gave | containment gives |
| --- | --- | --- |
| Bischling | Bichling | **Bischlinghöhe** ✓ |
| Sonnwendjoch | Sonnenjoch | **Hinteres Sonnwendjoch** ✓ |
| Krippenstein | *(nothing)* | **Hoher Krippenstein** ✓ |

The Krippenstein row corrects something from earlier in this step: it was first
recorded as a gazetteer gap, on the strength of an exact-match lookup. It is in
the dump all along, as *Hoher Krippenstein*.

**Stage 3, fuzzy matching** for genuine typos (`Innsbruk` → Innsbruck), kept
conservative — a high cut-off, a minimum length, and only for tokens that already
look like a place. A wrong place produces a plausible forecast for somewhere the
user never asked about, which is worse than admitting the name was not recognised.

#### Place names that are also English words

The most dangerous failure mode of a gazetteer matcher, and not hypothetical:

| name | how many places | largest |
| --- | --- | --- |
| **Rain** | 24 entries | a village of 1 081 |
| **See** | 15 entries | 1 265 |
| **Alm** | 20 entries | — |
| **Hard** | 2 entries | a town of **13 495** |
| **Going** | 2 entries | 358 |
| also *Still*, *First*, *Quick*, *Not*, *Fell*, *Bad*, *Fall*, *Pack* | | |

Dropping these names is not an option — Hard is a real town of 13 000 people and
someone there will ask for its weather. They are kept but matched only **with
evidence**: a preposition directly in front, or an initial capital in the user's
own text.

Two findings from this:

- **"to" had to be removed from the preposition list.** With it,
  `"is it going to rain in Wien"` read `to rain` as `to <place>` and answered for
  the village of Rain in Kärnten. "going to rain" is one of the most common
  phrasings a weather bot sees, so the trade was not close. `from`, `over` and
  `above` were dropped for the same reason.
- **Capitalisation is not evidence for words English always capitalises.**
  `"can you speak German"` matched the village of **Germanns** — a prefix hit with
  an initial capital, where the capital says nothing, because *German* is
  capitalised in every sentence it ever appears in. Languages, nationalities and
  month names were added to the ambiguous list; weekday names were already there.

Ambiguous names resolve to the **most populous** match, falling back to a
feature-code preference and then to elevation for the same-named peaks that have
no population to compare. The place carries name, state, lat, lon and elevation,
so it is directly usable as a forecast query.

When no place is found the extractor returns `None`. Restoring the place from
conversation history is the **dialogue manager's** job — it owns conversation
state, and putting that memory here would make this module untestable in
isolation.

### 5d — Day extraction

Rules over a closed vocabulary, resolved in `Europe/Vienna` rather than the
machine's local timezone — otherwise "today" silently shifts for anyone running
the bot outside Austria, the tutor included.

Covers `today`, `tonight`, `this morning/afternoon/evening`, `tomorrow` (+ parts
of day), `the day after tomorrow`, all seven weekday names with
`next`/`on`/`this`, `this weekend`, `in N days`, `in a week`, `right now`, and the
typo `tomorow` that is present in the training data. Parts of day are a **flag on
the day**, not separate days, because the weather service fetches a whole day and
the response generator narrows the hours.

Decisions worth recording:

- **Default is today.** A bare *"how windy is it on the Hochkönig"* means now.
  The slot carries an `is_default` flag so the dialogue manager can tell an
  assumption from a statement.
- **"this weekend" → Saturday.** A weekend spans two days and the forecast is
  per-day, so one has to be chosen; Saturday is the first of the two and the more
  likely tour day. The dialogue manager can offer Sunday as a follow-up.
- **Out-of-range days are flagged, not clamped.** The horizon is today + 6 days
  (Open-Meteo's free range). `"next Friday"` is +9 and `"in 10 days"` is +10, both
  returned with `in_range=False`. Clamping would make the bot answer for the wrong
  day while sounding confident; the flag lets it say it cannot forecast that far.
- **"next Friday" is read as the Friday of the following week.** Genuinely
  ambiguous in English, so the convention is documented in the code and pinned by
  a test — predictable rather than correct, and it usually lands outside the
  horizon where the bot says so anyway.
- **Bare "morning" and "evening" are deliberately not patterns**: *"good morning"*
  and *"have a good evening"* are greetings, and the classifier already has
  `greet`/`goodbye` for them.

### 5e — Results

**112 unit tests** (`python -m pytest`), 37 for the guard and 75 for the slots.
Every day test passes an explicit reference date; without it the suite would pass
or fail depending on which weekday it ran on, which is exactly the flakiness that
makes a test suite worthless as evidence.

Accuracy on the hand-written `data/slots_test.csv` (52 sentences, fixed reference
date Wednesday 2026-11-04):

| Slot | Accuracy | |
| --- | --- | --- |
| Place | **0.981** | 51/52 |
| Day | **1.000** | 52/52 |
| Part of day | **1.000** | 52/52 |
| Both slots | **0.981** | 51/52 |

Of the 52 sentences, 8 name no place at all, and all 8 were correctly left empty —
that is the ambiguous-name handling doing its job, and it is scored separately
because getting those right means *not* matching anything.

The one remaining error is a genuine gazetteer limitation, not a matcher bug:
*"will the trail be wet on the Rax tomorrow"* → **Rax, Burgenland** (a village of
615) instead of the **Raxalpe** massif in Niederösterreich. "Rax" is the
colloquial short name climbers use, it is only three letters, and it is an exact
match for a different real place — so neither the containment nor the fuzzy stage
can help. The honest fix is a small curated alias file, which is noted below
rather than quietly bolted on.

#### Sweep over the intent datasets

Running the extractor over all 400 intent sentences (`reports/slot_extraction_check.csv`)
is a free false-positive sweep: those sentences were written for the *intent*
classifier and never designed as slot tests, so they contain plenty of ordinary
English for a gazetteer matcher to trip over.

| | |
| --- | --- |
| sentences | 400 |
| a place was found in | 258 (64.5 %) |
| of those, non-exact matches | 7 |
| a day word was found in | 187 (46.8 %) |
| **places matched from a lower-case word** | **0** |

The zero is the number that matters: across 400 sentences, not one ordinary
English word was read as a place. Six of the seven non-exact matches are correct
(Krippenstein ×3, Bischling ×2, Sonnwendjoch); the seventh, `Riesneralm` →
*Griesneralm*, is wrong because the Riesneralm ski area is simply not in GeoNames.

### Open points for the next step

- **A curated alias file** for colloquial short names GeoNames does not carry:
  "the Rax" → Raxalpe, "the Glockner" → Großglockner. Small, hand-written,
  with a `source` column so hand-added rows stay distinguishable from GeoNames
  rows. This is the one fix the measured errors actually ask for.
- `Place.name` echoes the gazetteer's canonical name, which occasionally differs
  from what the user typed (*Mölltaler Gletscher* → **Wurtenkees**). The matched
  text is kept on the slot, so the response generator can prefer the user's
  wording and mention the official name once.
- `data/places_at.csv` is 2.2 MB. Acceptable for the submission, but if it needs
  to shrink, the obvious candidates are `HLL` (1 092 hills) and `SLP` (714 named
  ski runs), which contributed nothing to the measured accuracy.
- The dialogue manager still owes: calling the guard **first**, remembering the
  last place, confirming back when `Place.fuzzy` is set, and saying "I can only
  forecast 7 days ahead" when `DaySlot.in_range` is false.
