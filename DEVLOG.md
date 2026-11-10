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
Lawinenwarndienst), with no forecast values attached. *(Corrected in step 6:
avalanche.report covers only Tyrol, South Tyrol and Trentino, not Austria. The
implemented guard points at lawinen.at, the joint portal of the state services,
and adds avalanche.report only for places in Tyrol - see the step 6 entry.)*

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

---

## Step 6 — Weather service, plus two corrections to step 5 (2026-11-04)

### Correction 1 — the avalanche referral named the wrong service

The step 2 plan and the step 5 implementation both pointed the user at
**avalanche.report**. That was factually wrong in a way that matters for a
safety referral: avalanche.report is the **Euregio** bulletin and covers only
**Tyrol, South Tyrol and Trentino**. In Austria avalanche warning is organised
**per federal state**, and the joint portal of the state services, with a map
for the whole country, is **lawinen.at**.

Sending someone in Styria or Carinthia to a bulletin that does not cover their
mountain is exactly the failure the guard exists to prevent — it would have
looked like authoritative advice while being useless.

The guard now:

- points at **lawinen.at** by default, for every message;
- **additionally** names avalanche.report when a place in **Tirol** was
  extracted, where it genuinely applies.

`check_message(message, state=None)` takes the state as an **optional** second
argument, and the ordering decision from step 5 is unchanged: the guard still
runs first and never waits for place extraction. The state only *refines* the
referral. If extraction fails or is skipped, the user still gets the correct
nationwide portal — the safety behaviour does not depend on the NLU succeeding.

Tests: the default referral is asserted **not** to contain avalanche.report,
the Tyrol case is asserted to contain both, and one end-to-end test runs the
real extractor over "avalanche risk on the Wildspitze" (Tirol → both) and
"avalanche danger on the Hochschwab" (Steiermark → lawinen.at only). That last
test also pins the contract between the two modules: the guard reads the state
exactly as `Place.state` spells it.

The step 2 entry has been corrected in place with a visible note, as with the
embeddings correction in step 5.

### Correction 2 — the alias file, and two more bugs it uncovered

`data/place_aliases.csv` holds the colloquial short names GeoNames does not
carry. Each row is `alias, geonames_name, state, reason`; the alias **selects
an existing gazetteer row** and can never introduce coordinates of its own, so
it cannot point a forecast at an unverified location. Checked **before** the
gazetteer, because the gazetteer is what gets these wrong. Every row carries a
reason — an undocumented alias list is indistinguishable from a list of
arbitrary preferences. An alias that fails to resolve raises at index build
time rather than silently doing nothing.

The eight rows, and what each was answering before:

| alias | resolved to (wrong) | now |
| --- | --- | --- |
| the Rax / Rax | **Rax, Burgenland** (village, 615) | Raxalpe |
| the Glockner / Glockner | **Glöcknerin, Salzburg** (a different mountain) | Großglockner |
| Schmitten / die Schmitten | **Schmitten** (village at the valley station, 1 200 m lower) | Schmittenhöhe |
| Sonnblick | **Stubacher Sonnblick, Tirol** (a different 3 000er) | Hoher Sonnblick |
| Emberger Alm | **Maria Alm am Steinernen Meer** | Emberger Alpe |

Auditing my own intent data for "other short names that resolve wrongly"
turned up two further problems that aliases were the wrong fix for:

1. **"at the Hohe Wand" → the hamlet "Höhe" in Kärnten.** The Hohe Wand is a
   well-known climbing and hiking plateau in Niederösterreich, and it *is* in
   the dump — filed under the feature code **`CLF`** (cliff), which was not in
   the kept set. Adding `CLF` is the right fix rather than an alias: cliffs are
   legitimate destinations for this bot's users, and the alias would have
   papered over a whole missing category. (+196 rows.)
2. **"Emberger Alm" → Maria Alm.** GeoNames spells it *Alpe*, so the full name
   missed, and the scan fell back to the single token "Alm" — which is in the
   ambiguous-word list but was capitalised, so the capitalisation evidence let
   it through to Maria Alm's alternate name "Alm". This one *is* an alias case,
   because the spelling difference is in the data, not in the matcher.

Not changed, and noted rather than patched:

- **"Montafon" → Sankt Anton im Montafon** (715 inhabitants). The Montafon is a
  valley; resolving it to one arbitrary village in that valley is defensible
  but not ideal. Not *wrong* in the way the alias cases were, so it stays out
  of a file meant to be small and curated.
- **"Annaberg" → Annaberg im Lammertal.** There are three Annabergs; the
  documented population rule picks the largest. Inherent ambiguity, correctly
  handled by the documented rule.
- **"Mittagskogel" → Kepa.** The correct mountain, under its Slovene name,
  which is what GeoNames stores as canonical. An alias cannot rename a row, so
  this stays on the display-name list below with *Mölltaler Gletscher →
  Wurtenkees*.

#### Slot metrics, before and after

| | before (step 5) | after |
| --- | --- | --- |
| Place accuracy | 0.981 (51/52) | **1.000 (52/52)** |
| Day accuracy | 1.000 | 1.000 |
| Part-of-day accuracy | 1.000 | 1.000 |
| Places matched with no capitalised word (400 intent sentences) | 0 | **0** |

The one remaining step 5 error ("the Rax") is gone, and the false-positive
count stayed at zero — the aliases bought recall without costing precision.

**The false-positive metric itself had to be fixed to stay meaningful.** The
step 5 audit counted "places matched from a lower-case word" by looking at the
first character of the matched text. After the aliases, that counted 4 — all of
them the alias `"the Rax"`, where the lower-case word is the *article* and
"Rax" is capitalised. The metric now checks whether **any** word in the matched
text is capitalised, and is computed inside `evaluate_slots.py` rather than by
an ad-hoc command, so the number in this table is reproducible.

### Step 6 — the weather service

`peakcast/weather.py`. One public entry point, `get_forecast(place, day)`,
returning a `WeatherReport`.

#### The decision that matters: send the elevation

Open-Meteo otherwise answers for the mean elevation of its grid cell, which
near a summit is a different mountain:

| Request for the Großglockner | Model elevation | Max temperature |
| --- | --- | --- |
| no elevation parameter | 3 650 m | −8.9 °C |
| `elevation=3798` | 3 798 m | **−9.8 °C** |

A degree is the difference between rain and snow, so this is not cosmetic. It
is also the retrospective justification for two earlier decisions: keeping the
elevation of every gazetteer row, and **excluding the administrative divisions**
in step 5 — their elevation describes an area, and `ADM3 Sölden` at 2 856 m
would have produced a forecast for 1 500 m above the village.

#### Daily values, and the part-of-day path

Daily: temperature max/min, WMO code, mean cloud cover, max wind, **max
gusts**, rain (mm), snowfall (cm), max precipitation probability. Units are
requested explicitly rather than taken from the API defaults, so a change on
their side cannot silently turn km/h into m/s in the answers. Open-Meteo
already returns snowfall in cm and rain in mm, which is the pair a ski tourer
wants.

When a part of day was extracted, the values are computed from **hourly** data
for that window instead, because averaging the whole day would answer a
different question. Each variable is reduced by its own rule, and the rule
matters:

| variable | reduction | why |
| --- | --- | --- |
| temperature | max / min | as the daily block does |
| wind, gusts | **max** | what matters is the strongest gust someone is exposed to, not the average |
| rain, snowfall | **sum** | they are per-hour amounts |
| probability | max | matches `precipitation_probability_max` |
| cloud cover | mean | matches `cloud_cover_mean` |
| WMO code | **max** | "the worst it gets in that window", which is what a tour planner needs |

The WMO ordering is only approximately severity-ordered, which is why the code
is used for the sky category and never printed as a number.

Windows, in local Vienna hours: morning 06–12, afternoon 12–18, evening 18–23,
night **21–06 crossing midnight**. The night window is the interesting one:
someone asking "how cold tonight" at 18:00 means the hours they will actually
be out in, which run into the small hours. There is a test that puts the
coldest hour at 03:00 the *next* day and a sentinel at 14:00 the same day, so
it fails both if the window stops at midnight and if it leaks into the
afternoon.

#### Sky mapping

Three rules, in order, documented on the function:

1. WMO **45/48 → `foggy`**. Fog is its own category rather than part of
   "cloudy", because for this use case it is a different answer: fog on a
   summit means no view and poor orientation, which is what `ask_sky` is
   actually about.
2. Any code from **51 (drizzle) upwards → `cloudy`**. Precipitation implies
   cloud, so the bot can never report "sunny" for an hour it also says is
   raining — an inconsistency the cloud thresholds alone would allow, since a
   shower can fall on a day averaging 20 % cloud.
3. Otherwise **cloud cover decides**: `< 25 %` sunny, `25–75 %` partly cloudy,
   `≥ 75 %` cloudy. These follow the okta convention (the sky is reported in
   eighths; "clear" is up to 2/8 = 25 %). The upper bound is 75 % rather than
   the textbook 87.5 % because this value is a *mean* over a day or a window,
   and a daily mean of 75 % already means very little sun.

#### Robustness

Six-second timeout, **one** retry, a ten-minute in-memory cache, and any
failure falls back to the offline dataset.

- Six seconds because a user on mobile data at a trailhead will not wait, and
  the fallback should feel like an answer rather than a hang.
- One retry, not three: the common failure is a single dropped packet, and
  further attempts only lengthen the wait before the fallback that was always
  coming.
- The cache is keyed by rounded coordinates plus elevation and stores the whole
  payload, which covers the full week. That makes the concept's
  *"And what about tomorrow?"* follow-up cost **zero** requests — there is a
  test asserting two different days at one place produce one HTTP call.
- A plain dict rather than `functools.lru_cache`, because entries have to
  expire: a cached forecast is useful for minutes and wrong by tomorrow.

**Out-of-range days raise instead of falling back.** This is the one deliberate
asymmetry. The network being down is a reason to show demo data; a question
about nine days out has an honest answer ("I can only forecast 7 days ahead"),
and quietly inventing numbers would be worse than saying so.

#### The offline fallback

`data/offline_forecasts.json`, fictional values for 8 demo places — 4 valley
towns (Innsbruck, Salzburg, Zell am See, Lienz) and 4 summits (Schmittenhöhe,
Hochkönig, Hoher Dachstein, Großglockner) — **keyed by day offset 0–6, never by
date**, so the demo cannot go stale. A test asserts no ISO date string appears
anywhere in the values.

Two things that made it better than a pile of plausible-looking numbers:

- **One coherent weather story across every place**: settled high pressure with
  valley inversion fog on day 0, a front crossing on day 3 with rain in the
  valleys and snow higher up, clearing by day 5. Temperatures follow the
  standard 6.5 °C/km lapse rate from a common sea-level value, and wind
  increases with exposure. So the 8 places agree with each other, which a
  per-place invention would not have.
- **Unlisted places get an elevation-banded generic profile** (valley /
  mid-mountain / high-mountain) rather than one flat default. Elevation is what
  decides whether a plausible answer is +12 °C and rain or −6 °C and snow;
  without the bands, offline mode would have answered a question about a
  3 000 m summit with valley numbers.

An error caught by writing the consistency test rather than by reading the
data: the first draft used a **fixed snow line per day**, which produced
**rain at −5.8 °C on the Schmittenhöhe**. The split now follows the computed
temperature at that elevation. Fictional does not mean careless — implausible
demo data undermines the demo it exists for, so the property is pinned by a
test that checks every value in the file: no rain at or below freezing, no snow
above +4 °C, gusts never below mean wind, min never above max, percentages in
range.

#### `source` is part of the report

`WeatherReport.source` is `"live"` or `"offline"`, with an `is_offline`
convenience property. The response generator is required to say when sample
data was used: **a fictional forecast must never be presentable as a real
one.** `try_weather.py` prints it in capitals for the offline case, and the
dataset's own header says the same thing.

Both paths go through one construction function, so the live and offline
reports can never drift into exposing different fields or a different sky rule.

### Testing

**178 tests, no network required**, plus 2 live smoke tests behind a `--live`
flag (`tests/conftest.py`).

Skipped by default because a suite that needs the internet is not a suite — it
fails on a train and in a CI container without egress. Kept because a fully
mocked suite can pass while the real API has renamed a field, which is the one
thing mocks cannot catch. The live tests check a valley town and a 3 798 m
summit, and assert that the elevation requested is the elevation the model
used.

The mocked tests concentrate on the failure paths and the aggregation, since
those are where a bug would be invisible in the output: timeout, HTTP 500, and
a 200 response with no `daily` block (checked separately, because a status-code
check alone would miss it) must each produce an offline report; the retry must
recover a single dropped request; the cache must not serve one place's forecast
for another.

One test bug worth recording, because it is the kind that produces false
confidence: the hourly fixture filled unlisted hours with a `−99` sentinel so a
wrong window would be obvious. That works for the afternoon window, but the
night window legitimately spans hours the test did not set, so the night test
failed with `−99` against the expected `−12`. The fixture now takes a
`default_temperature`, and the night test sets a neutral baseline plus an
explicit out-of-window sentinel. The test was wrong, not the code — but a
fixture that cannot distinguish those two is a liability.

`scripts/try_weather.py` runs the whole chain for one typed message — guard,
slots, forecast — so a single screenshot shows every component built so far,
including the guard refusing an avalanche question with the Tyrol-specific
referral and the out-of-range day being declined.

### Open points for the next step

- **Display names.** `Place.name` echoes the gazetteer's canonical name, which
  is occasionally not what an Austrian user would say: *Mölltaler Gletscher* →
  **Wurtenkees**, *Mittagskogel* → **Kepa**. `matched_text` already carries the
  user's own wording, so the response generator should prefer it and mention
  the official name once.
- The response generator owes: the `source == "offline"` disclosure, the
  `Place.fuzzy` confirmation ("I assumed you meant Innsbruck"), the
  out-of-range message, and the step 2 decision that **snow is reported before
  rain when the question mentioned snow**.
- `precipitation_probability_max` is fetched but not yet used in any answer.
- The dialogue manager still owes: guard first, remembering the last place, and
  ending the conversation after a goodbye or roughly ten answers.

---

## Step 7 — Dialogue manager and response generator: the bot works end-to-end (2026-11-05)

### Small fix first — display name vs canonical name

GeoNames' canonical name is sometimes not the name an Austrian user knows the
place by, and answering with it reads as a *different mountain*:

| user types | GeoNames canonical | what the bot now says |
| --- | --- | --- |
| Mittagskogel | **Kepa** (the Slovene name) | Mittagskogel |
| Mölltaler Gletscher | **Wurtenkees** | Mölltaler Gletscher |
| Soelden / Solden | Sölden | Sölden |

`Place` now carries a `display_name` and a `shown` property; `name` stays the
canonical form and is used for **resolution only** — gazetteer lookups, row
comparison, the forecast cache key.

The selection rule turned out to be simple once framed properly. All the
spellings of a row that normalise to the same key are, by definition, the
different ways of writing *the one name the user typed*. Among those:

1. the canonical name, if it is one of them (so "Soelden" is answered as
   *Sölden*, not echoed back);
2. otherwise a spelling with diacritics (this is what rescues *Mittagskogel*,
   whose row's canonical name does not normalise to what the user typed);
3. otherwise the longest, as a stable tie-break.

What falls through to the canonical name is exactly what should: a **typo**
("Innsbruk" → Innsbruck, not the user's mistake), an **alias** ("the Glockner"
is not a spelling of anything → Großglockner), and a **shortened compound**
("Bischling" → Bischlinghöhe).

Also added `Place.ambiguous`, set when more than one *distinct* place answers
to the name, so the response generator can name the federal state for
"See", "Annaberg" or "Mittagskogel" without appending it to every answer.

All four match stages (alias, exact, partial, fuzzy) now build their result
through **one** `_resolved()` helper, so they cannot drift apart on how the
display name is chosen or when a name counts as ambiguous.

### 7a — Dialogue manager

`peakcast/dialogue.py`. `Bot.respond(state, message)` runs one turn:

    safety guard → intent classifier → slot extraction
                 → context handling → weather service → response generator

**Why the guard is still first**, restated because this is the step where it
could quietly have been moved: the classifier labels "avalanche risk after the
fresh snow" as `ask_precipitation` at p = 0.62, above the 0.35 threshold, and
the *better* model was the more confidently wrong one. A guard placed after the
classifier would only ever see messages the model had already mislabelled —
exactly the set it cannot be trusted on. It therefore runs on the raw message
and never waits for the NLU. The extracted place is used **only** to pick the
regional bulletin, and falls back to the remembered place, so
*"and what about the avalanche risk?"* after talking about a Tyrolean peak
still offers avalanche.report.

#### Context handling — the core of the step, and the hard part

A pipeline can answer "how windy is it on the Hochkönig tomorrow". It cannot
answer *"and tomorrow?"*, which contains no weather word, no place, and
nothing for a classifier to work with. Four cases:

1. **An answer to an ask-back.** The pending intent resumes, so the user does
   not repeat the question.
2. **A confident weather question**, taken at face value; a missing place falls
   back to the remembered one.
3. **A follow-up with no weather word**, which reuses the remembered intent
   with whatever new slot the message carries.
4. **Nothing usable** → the help reply.

Three decisions inside that are worth recording, because each was a measured
problem rather than a design preference.

**"What about Kaprun?" was being declined as out of scope.** Measured:

| message | intent | p |
| --- | --- | --- |
| `what about Kaprun?` | `out_of_scope` | **0.88** |
| `can you book me a hotel in Innsbruck` | `out_of_scope` | **0.91** |

Both are confident `out_of_scope` **with a place in them**, so confidence
cannot separate them — and the first is one of the follow-ups the step is
supposed to support. What does separate them: the follow-up is *nothing but a
slot*. Strip out the place and day it names and only filler remains, where the
hotel question still says "book" and "hotel". `is_bare_slot_followup()`
implements exactly that, and `out_of_scope` is believed unless the message is a
bare slot **and** a weather conversation is already running. The filler list is
kept deliberately short: every word added to it is a word the bot stops
treating as content.

**"And in the afternoon?" needed its own slot flag.** `DaySlot` gained
`part_only`, set only by the `in the <part>` patterns. The distinction is real:
*"this afternoon"* means today whatever came before, while *"in the afternoon"*
leaves the day open and should inherit the day under discussion. Resolving the
bare form to today would silently answer about the wrong day halfway through a
conversation about tomorrow.

**A bug the inheritance introduced, caught by a stray hint.** The first version
re-ran `find_day(matched_text, reference=state.last_day.target_date)`, which
looks tidy and is wrong: it resolves "in the afternoon" to offset 0 *relative
to that date*, producing a `DaySlot` whose `target_date` was tomorrow and whose
`offset_days` was 0. The live path keys off the date and the offline dataset
keys off the offset, so the two answered for **different days** — which showed
up as an inversion-fog hint appearing on a clear afternoon. The inherited day
is now built explicitly, keeping `offset_days` relative to today, and there is
a test asserting the two stay in step.

**Day inheritance for ordinary questions.** A weather question with no day word
at all also inherits the remembered day: after "how windy on the Hochkönig
tomorrow", *"will it snow there"* means tomorrow, which is how people talk. The
risk is answering about Saturday when the user meant now; it is accepted
because an explicit "today" or "right now" still overrides it (both set
`is_default=False`), and **every answer names the day it used**, so a wrong
inheritance is visible and correctable rather than silent.

#### The conversation limit

Every reply goes through one `_finish_turn()`, which is what makes the limit
reliable — there is a single place that increments the counter, so no path can
answer without being counted, including greetings, refusals and help replies.
That matters for the case the limit most needs to cover: a conversation of
nothing but unanswerable messages would otherwise never end. The closing is
**appended to the tenth answer** rather than sent as an eleventh message, so
the user still gets the answer they asked for.

#### The step 1 to-do, finally closed

`ensure_model()` trains the model on first start if the file is missing.
`models/*.joblib` is git-ignored because it is a build artefact, so a fresh
checkout — or the submitted ZIP — has none, and "run this script first" is not
an acceptable first experience for the tutor.

It rebuilds the iteration-1 pipeline from **named constants** (`ITER1_C = 10.0`,
`ITER1_CHAR_NGRAM_RANGE = (2, 4)`) rather than re-running the 75-fit grid
search to rediscover numbers that are already known. Verified: the
auto-trained model reproduces the committed metrics exactly — test accuracy
**0.8125**, macro F1 0.81. `scripts/optimize_intent.py` remains the *source*
of those values and would reveal a change.

### 7b — Response generator

`peakcast/responses.py`. Templates, not generation — and that is not a
compromise. The answers must contain exact values with units, and a template
guarantees the number in the sentence is the number the service returned. Each
intent has three phrasings, picked with an **injected** `random.Random`, so the
tests are deterministic without monkeypatching anything.

**A wording bug worth recording: "Wind in Großglockner".** You are *in*
Innsbruck but *on* the Großglockner, and a bot that gets this wrong sounds like
it has never been outside. English picks the preposition from what kind of
place it is — and the gazetteer already records exactly that in the GeoNames
feature code, so the fix cost nothing: `PPL*` → "in", `MT`/`PK`/`GLCR` →
"on the", `PASS`/`HUT`/`LK` → "at the", `VAL`/`MTS` → "in the".

**Hints are rules with named thresholds, marked as heuristics.** Every constant
carries that marking in its comment, because a number in a safety-adjacent
sentence invites being read as authoritative. Gusts (25 / 40 km/h — the
concept's named decision variable for paragliders), fresh snow (1 / 15 cm),
thunderstorm WMO codes, fog, freezing (0 / −10 °C), heat (25 °C).

Two decisions about hints:

- **Ordered by how much they should change someone's plan**, and only **one** is
  appended. Two is a lecture, and the ordering means the one that survives is
  the most consequential: thunderstorms outrank strong gusts, which outrank
  deep cold.
- **Substantial fresh snow points at the avalanche bulletin.** The one hint
  that defers to the real service, in the one situation where it matters most.

**Precipitation ordering closes the step 2 decision.** Rain and snow
deliberately share one intent, because the same sentence means rain in July and
snow in January — the distinction is seasonal, not linguistic. The ordering was
deferred to this module and is now implemented: snowfall first when the user's
own message mentions snow, rainfall first otherwise. A day with no measurable
precipitation is called **"dry"**, not "0 mm of rain", which reads as a
measurement; but "dry, though there is a 60 % chance of showers" is a different
plan from "dry" and is said that way.

**The offline disclosure is graded.** A fictional forecast may never read as a
real one — but repeating the same paragraph on all ten answers turns it into
wallpaper a reader skips, which defeats the point. The full note is used the
first time and a short tag afterwards, tracked on the conversation state.

### 7c — Testing

**274 tests, no network** (276 with `--live`). New: 39 dialogue, 46 response, 8
display-name.

The dialogue tests are the ones that could not be written before: every earlier
suite checks one message, and the behaviour that makes this a bot — remembering
a place, continuing a question, asking back and resuming — only exists across
turns. They all run with `allow_live=False` and a fixed `today`, so "tomorrow"
is the same date on every machine on every day.

**A live test broke, and for an instructive reason.** `test_live_api_still_
returns_what_we_parse` asked the real API for the fixed constant
`TODAY = 2026-11-04`. It passed on the day it was written and failed the next
morning, because a forecast service does not serve yesterday. The mocked tests
are right to use a fixed date — their fake payloads are built around the same
constant — but a *live* test has to use the real clock. Fixed with a separate
`a_real_day()` helper and the reason written next to it. This is the second
time a test fixture, rather than the code, has been the bug; both times the
fixture could not distinguish "the code is wrong" from "the fixture is wrong",
which is the property to watch for.

**One test pins a defect on purpose.** "how many people live in Linz" is
labelled `ask_temperature` at p = 0.48 and "how high is the Dachstein" at
p = 0.62 — both recorded as errors back in step 3, where the model latched onto
the quantity phrasing. The dialogue manager cannot second-guess a confident
classifier, and writing a rule for these two sentences would be fitting the
test set. So `test_known_limitation_quantity_questions_leak_to_a_forecast`
asserts the *current, wrong* behaviour, with a message telling the next person
to delete it if a later data iteration fixes the classifier. A known defect
that fails loudly when fixed is better than one that passes unnoticed.

### Example conversations

All three exactly as `scripts/chat_cli.py` produced them.

**1 — Follow-ups (live, `--seed 17`).** Neither follow-up contains a weather
word; the day and then the place change under a remembered intent.

```
You: How windy is it on the Hochkoenig today?

PeakCast: Today on the Hochkönig expect 13 km/h of wind and gusts to 23
          km/h. Around or below freezing, so expect icy patches early and
          late.

You: And tomorrow?

PeakCast: Wind on the Hochkönig tomorrow is around 14 km/h, with gusts up to
          29 km/h. Gusts reach 29 km/h, worth checking before you launch.

You: What about Kaprun?

PeakCast: Wind in Kaprun tomorrow is around 3 km/h, with gusts up to 6 km/h.

You: And in the afternoon?

PeakCast: Wind in Kaprun tomorrow in the afternoon is around 2 km/h, with
          gusts up to 5 km/h.

You: thanks, bye

PeakCast: Take care up there - have a good tour!
```

**2 — Ask-back, safety guard, forecast horizon (live, `--seed 5`).**

```
You: hi

PeakCast: Hi! I'm PeakCast. I can tell you the temperature, sky, wind or
          precipitation forecast for places in Austria - towns, peaks and
          huts. Try: "How windy is it on the Hochkönig tomorrow?"

You: How cold is it tomorrow?

PeakCast: Which place do you mean?

You: Soelden

PeakCast: Tomorrow in Sölden it should be between 1.9 °C and 12.1 °C.

You: What is the avalanche risk there?

PeakCast: I can't answer avalanche questions - and I shouldn't, because
          getting that wrong is dangerous. Avalanche warning in Austria is
          run per federal state; the joint portal with a map for the whole
          country is https://lawinen.at. I can still tell you the
          temperature, sky, wind or precipitation forecast for a place if
          that helps your planning. For Tyrol you can also use
          https://avalanche.report, the joint bulletin for Tyrol, South
          Tyrol and Trentino.

You: How about next Friday?

PeakCast: I can only forecast about a week ahead, so I can't tell you about
          Friday.
```

Three things in one screen: the ask-back keeps the day the user already gave
(the answer is for *tomorrow*, not today); the guard reads the remembered place
to add the Tyrol bulletin; and the horizon is declined in words rather than
answered with invented numbers.

**3 — Offline disclosure and a real refusal (`--offline --seed 8`).**

```
You: Will it snow on the Schmitten in 3 days?

PeakCast: Schmittenhöhe on Sunday: 10.3 cm of fresh snow (90 % chance).
          Gusts to 102 km/h are strong - not flying weather, and exposed
          ridges will be unpleasant. Heads-up: the live forecast service is
          unreachable right now, so these are sample values from the offline
          demo dataset, not a real forecast.

You: And the wind?

PeakCast: Wind on the Schmittenhöhe on Sunday is around 50 km/h, with gusts
          up to 102 km/h. Gusts to 102 km/h are strong - not flying weather,
          and exposed ridges will be unpleasant. (Sample data, not a real
          forecast.)

You: Can you book me a hotel in Innsbruck?

PeakCast: That's outside what I do - I only answer weather questions for
          places in Austria. Ask me about temperature, sky, wind or
          precipitation and I'm happy to help.
```

Note "the Schmitten" resolving through the curated alias to the Schmittenhöhe,
snow reported first because the question said "snow", the full disclosure
followed by the short tag, and the hotel question still being refused despite
naming a place — the counterpart to "What about Kaprun?" in conversation 1.

### Open points for the next step

- The Streamlit UI is all that remains of the planned architecture. It needs:
  `Bot` held in session state, `DialogueState` per session, the
  `ensure_model()` startup path, and a visible marker when
  `WeatherReport.source` is offline.
- `precipitation_probability_max` is fetched and used in the precipitation
  phrase, but not in any hint. A "70 % chance, take the jacket" hint would be
  cheap.
- The `out_of_scope` quantity leak above is the one classifier error now
  visible in normal use. The honest fix is more `out_of_scope` training
  sentences in that phrasing style, collected from real test conversations -
  **not** from `reports/misclassified_*.csv`, which would leak the test set.
  The chat CLI is now the tool that makes collecting them possible.
- `Place.matched_text` is kept but the response generator does not use it; the
  display name covers every case met so far. Worth deleting if nothing needs
  it by the end.

---

## Step 8 — Streamlit interface, plus three fixes to step 7 (2026-11-10)

### Fix 1 — hints no longer repeat a value the sentence just gave

The step 7 answers stuttered:

> Wind on the Hochkönig tomorrow is around 14 km/h, with gusts up to 29 km/h.
> **Gusts reach 29 km/h**, worth checking before you launch.

Two values carried numbers in their hints: gusts and substantial fresh snow.
`build_hints()` now takes a `stated` set of value kinds, and
`VALUES_STATED_BY_INTENT` maps each intent to what its template already says —
`ask_wind` states gusts, `ask_precipitation` states snow depth, `ask_overview`
states both, `ask_temperature` and `ask_sky` state neither.

The important part of the rule: the hint drops **only the figure**, never the
advice, because the advice is the part the user cannot derive from the numbers.

> Wind on the Hochkönig tomorrow is around 16 km/h, with gusts up to 35 km/h.
> **Worth checking before you launch.**

And the figure is still given where it is news rather than a repeat — a
temperature answer says nothing about wind, so the gust hint keeps its number
there. There is a test for each direction, plus one asserting the intent map is
**complete**: the lookup defaults to "states nothing", which is the safe
direction but would quietly undo the fix for a new intent.

### Fix 2 — out-of-range days name both dates

Before: *"I can only forecast about a week ahead, so I can't tell you about
Friday."* Which Friday? "next Friday" is genuinely ambiguous in English and the
bot resolves it by a **documented convention** (the Friday of the following
week), not by a fact — so the reading it chose has to be visible.

Now:

> Friday 20 November is too far ahead - I can forecast up to Monday 16 November.

Naming the last forecastable date as well turns a refusal into something
actionable: the user learns what they *can* ask for, not only what they cannot.
`format_calendar_date()` spells dates as "Friday 13 November", with the day
number unpadded because that is how a person writes it.

### Fix 3 — the known-defect test is now a strict xfail

`test_known_limitation_quantity_questions_leak_to_a_forecast` asserted the
*wrong* behaviour, which worked but read as if the project endorsed it. It is
now `test_quantity_questions_are_out_of_scope`, asserting what the bot
**should** do, under `@pytest.mark.xfail(strict=True)` with the full
explanation as the reason.

`strict=True` is the point: if a later data iteration fixes the classifier, the
test XPASSes and **fails the suite**, forcing the marker to be removed rather
than letting a fixed defect sit behind a stale exemption.

*(Note: the example transcripts in the step 7 entry above were captured before
fixes 1 and 2 and show the older hint wording. The README transcript has been
regenerated; the step 7 entry is left as it was, because it is a record of what
the bot did that day.)*

### Step 8 — the Streamlit interface

`app.py`, started with `streamlit run app.py`.

**The file is thin on purpose, and that is the main design decision.** It owns
the screen and nothing else: no intent logic, no slot handling, no forecast
decisions, no wording. Every reply comes from `Bot.respond`, the same code path
the terminal client and 279 tests already use. The rule written into the module
docstring is that `app.py` may *render* what a `BotTurn` contains and may not
derive anything `dialogue.py` has not already decided — because if the UI could
decide anything about an answer, there would be two bots, one tested and one on
screen, and the tested one would not be the one the tutor sees.

#### Built for a phone

The concept's users are at a trailhead, not at a desk. So: `layout="centered"`
and single column, short messages, nothing needing horizontal space, and the
sidebar collapsed by default — which is why **nothing essential lives only in
the sidebar**. A centred layout also reads better on a laptop, where `wide`
would stretch chat bubbles into unreadable lines.

#### State, and why it lives where it does

Streamlit re-executes the whole script on every interaction, so anything that
must survive a rerun goes in `st.session_state`. Two separate things are kept
there, deliberately:

- `history` — the rendered transcript;
- `dialogue` — the bot's own `DialogueState` (last place, last day, answer
  count).

They are not the same thing: the transcript is what the screen shows, the
`DialogueState` is what the bot remembers. Reset has to clear **both**, and
there is a test for that: clearing only the transcript would leave the next
question inheriting the old place and answer count.

The bot itself is cached with `st.cache_resource`, not `st.cache_data` — it
holds a fitted scikit-learn pipeline, which is a live object to share rather
than a value to copy. Without the cache, Streamlit would reload (or retrain)
the model on every keystroke. The cache key is `allow_live`, so flipping the
offline toggle hands back a bot with the new setting instead of silently
keeping the old one.

#### The example buttons, and a Streamlit detail

The four buttons were chosen to show one capability each rather than to look
full: summit wind (the paragliding case from the concept), snow at a ski
resort, a temperature question with a part of day, and a **follow-up** that
only works because the bot remembers the previous turn.

A clicked button cannot answer in place: Streamlit reruns the script on the
click, and the answer has to be appended *before* the transcript is drawn. So a
click stores the question in `session_state.queued` and triggers a rerun, which
handles it on the way through. The buttons disappear once the conversation is
under way, because by then they are clutter — and the last one only makes sense
as a follow-up anyway.

#### The data-source badge

Every answer carries a caption under it: `● live forecast` in green or
`▲ sample data - not a real forecast` in orange. This is the UI half of a rule
the whole project is built around, and the reason `WeatherReport` has carried a
`source` field through three modules: **a fictional forecast must never be able
to look real.** A caption under the message is harder to skip than a sentence
inside it.

#### End of conversation

On a goodbye or the tenth answer the input is rendered **disabled rather than
removed**, so the end of the conversation reads as a state of the page rather
than as something having broken, and a restart button appears. The sidebar also
has a reset button that works mid-conversation. A caption warns when three or
fewer answers remain, so the limit does not arrive as a surprise.

#### First start

If `models/intent_model_iter1.joblib` is missing — which it is on a fresh
checkout, because it is a git-ignored build artefact — the app trains it inside
an `st.spinner` so the first second does not look like a hang. This is the
step 1 to-do reaching the screen it was written for.

#### A bug the smoke test caught immediately

The sidebar offers **"offline demo mode"**; `Bot` takes **`allow_live`**. The
first version passed `load_bot(offline_mode)` — inverted — so *live* mode
served sample data and *offline* mode tried the network. It was invisible in
the code and obvious the moment the badge was asserted: the first smoke-test
run came back `▲ sample data` with the toggle off. Both directions are now
pinned by tests, one of them `--live`, and the inversion has a comment at the
call site.

This is the argument for the badge in miniature: a mislabelled data source is
exactly the kind of error that a user cannot see and a test can.

#### `.streamlit/config.toml`

Two settings. `gatherUsageStats = false` — Streamlit sends anonymous usage
statistics by default, and coursework shown on a university machine should not
make outbound calls nobody asked for; it also keeps the startup output clean
for a live demo. `toolbarMode = "minimal"` hides the developer menu ("Rerun",
"Clear cache", "Deploy"), which a hiker on a phone has no use for.

### Testing

**292 tests with no network** (295 with `--live`), of which 13 are the new UI
smoke tests, plus 1 strict xfail.

They are *smoke* tests on purpose. The bot's behaviour is covered by the 279
tests against `dialogue.py` and below; repeating any of it through the UI would
mean two places to update whenever an answer changes. What is left for the UI
to get wrong is **wiring**, and that is what is checked: the page starts, a
button and a typed message both reach the bot, a follow-up keeps its context
(the test that would catch `session_state` being used wrongly), the badge says
the right thing in both modes, the NLU toggle shows and hides, a goodbye locks
the input, both reset buttons clear both kinds of state, and the ten-answer
limit reaches the screen.

Two tests exist because the project has obligations rather than because the
code might break: one asserts the **disclaimer** is on screen (not a
replacement for official mountain weather or avalanche reports, with the
lawinen.at link) and one asserts the **attribution** for Open-Meteo and
GeoNames CC BY. Those are promises the README makes; a test is how they stay
true.

Verified beyond `AppTest` as well: `streamlit run app.py` starts the real
server cleanly, which `AppTest` (an in-process runner) does not prove on its
own.

### Where the project stands

Every component in the concept's architecture is built and tested:

| Component | Module | Status |
| --- | --- | --- |
| Chat UI | `app.py` | ✔ |
| Dialogue manager | `peakcast/dialogue.py` | ✔ |
| Safety guard | `peakcast/safety_guard.py` | ✔ |
| Intent classifier | `peakcast/intent_classifier.py` | ✔ CV 0.803 / test 0.813 |
| Slot extraction | `peakcast/slots.py` | ✔ place 1.000 / day 1.000 |
| Weather service | `peakcast/weather.py` | ✔ live + offline fallback |
| Response generator | `peakcast/responses.py` | ✔ |

### Open points

- **The `out_of_scope` quantity leak** is the one classifier error now visible
  in normal use, and it is behind a strict xfail. The honest fix is more
  `out_of_scope` training sentences in that phrasing style, collected from real
  conversations with the running bot — **not** from
  `reports/misclassified_*.csv`, which would leak the test set. The app is now
  the tool that makes collecting them realistic.
- The optional **sentence-embedding comparison** is still optional, and still
  not part of the submitted concept. It is now the only thing left that could
  use remaining time.
- `Place.matched_text` is still unused by the response generator; the display
  name has covered every case met so far. Worth deleting if nothing needs it.
- A **"70 % chance, take the jacket"** hint would use
  `precipitation_probability_max`, which is fetched and reported but never
  drives a hint.

---

## Step 9 — Deployment prep and user-test tooling (2026-11-10)

Two jobs: make the app deployable on Streamlit Community Cloud from a private
repository, and make a user test produce data that can be summarised rather
than remembered. The ZIP/README route for the tutor is unchanged — verified,
not assumed (see the clean-clone check below).

### Deployment prep

#### The requirements split

`requirements.txt` is what the deploy host installs on every build, so it now
contains **only** what `streamlit run app.py` needs:

| | |
| --- | --- |
| `requirements.txt` | streamlit, scikit-learn, pandas, numpy, joblib, requests |
| `requirements-dev.txt` | `-r requirements.txt` plus pytest and matplotlib |

Decided by reading the imports rather than by guessing:

- **pandas stays runtime.** It is tempting to call it an analysis dependency,
  but `load_dataset` uses `pd.read_csv`, and `ensure_model` trains from the
  CSV on first start — which on a cloud host happens on *every* cold start.
- **numpy is now declared explicitly.** It arrives with scikit-learn anyway,
  but `intent_classifier.py` does `import numpy as np` directly, and a package
  you import is a package you declare. Relying on a transitive dependency is
  how a working app breaks when an upstream pin changes.
- **matplotlib is dev-only.** Its only users are
  `scripts/compare_iterations.py` and `scripts/evaluate_intent.py`, which
  produce the charts in `reports/`. Nothing in `peakcast/` or `app.py` touches
  it.
- **pytest is obviously dev-only**, and shipping a test runner to a web host
  is pure cost.

#### The clean-clone check

Rather than reason about whether a fresh deploy would work, I built one:
`git archive HEAD` into an empty folder (so **only tracked files**), a new
virtual environment, and `pip install -r requirements.txt` alone. Then started
the app there.

| Check | Result |
| --- | --- |
| `pytest` / `matplotlib` importable? | **No** — correctly absent |
| Model file present? | **No** — `models/` is git-ignored by design |
| App starts? | Yes, no exception |
| Model trained on first page load? | Yes — `intent_model_iter1.joblib` appeared |
| Answers with a live forecast? | Yes, badge `● live forecast` |
| With `requests.get` dead? | Answers from the demo dataset, badge `▲ sample data` |
| `build_places.py` needed at runtime? | No — never imported |

So the committed `data/places_at.csv`, `data/place_aliases.csv` and
`data/offline_forecasts.json` are sufficient, and the GeoNames download script
is a build-time tool only. That was the thing most likely to be wrong, because
it is exactly the kind of dependency that is invisible on a machine where the
file already exists.

#### What goes in the README

A full Deployment section: the GitHub steps, the Community Cloud steps,
**Python 3.12** in Advanced settings (the pins have no wheels guarantee on a
newer Python, and that failure appears as an opaque build error), no secrets
needed, and what to expect afterwards — the first-load training, the ~12-hour
sleep and the ~1-minute wake, the offline fallback, the ~1 GB memory limit,
and that **a private repo does not make the app private**. The Open-Meteo
non-commercial term is restated there, next to the deployment it applies to.

### User-test tooling

#### `peakcast/transcript.py` — writes and parses

The transcript is the only artefact in this project that must be **readable by
a person and parsable by a script**, and those pull against each other: a
format that is pleasant to read is easy to parse sloppily. So the writer and
the parser live in one module, next to each other, with a round-trip test as
the anchor.

**Plain text, not JSON** — and the reason is not aesthetic. The tester has to
be willing to open the file. A `.txt` that reads like a conversation can be
checked at a glance before it is sent, which is the difference between
informed consent and forwarding a blob. The file says so itself:

> This file was created in your browser. Nothing was stored on a server. You
> can read it, edit it or delete it before sending it.

That sentence is in the file rather than only in the app, because the file is
what actually gets sent on.

The parser is deliberately **forgiving**: participants are told they may edit
the file, so an edited or hand-trimmed block is skipped rather than aborting,
and unknown keys are ignored. A strict parser would mean one tidied file
costing a whole session.

Two decisions about what gets written:

- **Only fields that apply.** A greeting carries no empty `place:` line.
  Readability is not a side concern here; a tester who cannot skim the file
  cannot meaningfully consent to sending it.
- **No `confidence: 0.00` on a guarded turn.** The safety guard never runs the
  classifier, so writing a zero would read as "the model was certain it was
  nothing" when the model was never asked. The field is simply absent.

Confidence is stored to two decimals, so the round-trip test compares it with
a tolerance — 0.9237 tells a reader nothing that 0.92 does not.

#### `records_from_history` is a pure function

The mapping from a live conversation to transcript records takes plain data
and returns plain data, and knows nothing about Streamlit. `app.py` only reads
`st.session_state` and hands it over. That is what makes the mapping testable
without starting an app — and the mapping is precisely where a field would
quietly go missing from an export, which is invisible in the UI and caught by
a test.

#### Feedback

`st.feedback("thumbs")` under each answer, optional and unlabelled. A rating
control that nagged would change the behaviour it exists to measure.

One bug worth recording, because it would have corrupted the data rather than
breaking anything: Streamlit keeps a widget value for as long as its key
exists, so after a reset, answer 1 of the *new* conversation inherited answer
1 of the *old* one's thumbs-down — and that stale rating would have been
exported as if it were real. `reset_conversation` now deletes every
`feedback_*` key, with a test.

#### `scripts/analyze_transcripts.py`

Reports per-intent counts, forecast sources, thumbs, and — deliberately — the
unflattering figures: turns below the 0.35 threshold, ask-backs, "I didn't
catch that" fallbacks, refusals, guard hits, and every thumbs-down with the
question that produced it. A summary showing only intent counts would be a
presentation, not an evaluation.

Three choices behind it:

- **Every example quoted is a real sentence from a transcript**, so the
  summary can be read next to the files it came from.
- **A bot turn is reported with the question before it.** "The bot asked which
  place" is far less useful than knowing *which question* made it ask.
- **Satisfaction is a share of the *rated* answers, not of all answers.** The
  control is optional, so most turns carry no rating; dividing by every answer
  would understate it and make the figure meaningless.

The three canned replies are matched by **exact text imported from
`dialogue.py`**, never retyped — so rewording a reply cannot silently stop it
being counted. There is a test that reads this script's own source and fails
if someone replaces the import with a literal.

Run against three generated transcripts, the summary immediately surfaced the
defect that is already behind a strict xfail: *"how high is the Dachstein"* →
`ask_temperature` p = 0.62, with a thumbs down. That is the tooling working —
the known classifier error showing up as a user complaint, in the report, from
the data.

`data/user_tests/` is git-ignored apart from its README. Those are other
people's conversations: working material for the evaluation, not part of a
repository that gets submitted.

### Two bugs found while building this

#### A mid-conversation "hmm" was answered instead of being questioned

Generating sample transcripts put gibberish into a running conversation for
the first time, and the bot answered it with a repeat of the previous
forecast.

The cause was the step 7 day inheritance. `_apply_context` replaces an
unspecified day with the remembered one — and then tested
`not day.is_default` to decide whether the message was a follow-up. After the
inheritance that is always false, so **every** message looked as though it had
named a day, and any unrecognised input became a follow-up.

A follow-up is a message that brings something *new*, so the test has to be on
what the user actually supplied. `named_place` and `named_day` are now captured
from the original slots, before any inheritance. Gibberish gets the help reply
again; follow-ups and day inheritance both still work.

This one only showed up because the user-test tooling made it easy to run a
long, messy conversation — which is the argument for the tooling in miniature.

#### A test that passed alone and failed under `--live`

`test_a_failing_weather_api_still_answers_and_says_it_is_sample_data`
monkeypatches `requests.get` to fail and expects the sample-data badge. It
passed on its own and failed with `--live`.

`peakcast.weather` caches whole payloads in a module-level dict for ten
minutes — right in production, where a conversation asks about the same place
repeatedly, and a trap in a suite, because the cache outlives a test. The live
tests had already fetched Lienz, so the cache answered from the warm payload,
`requests.get` was never called, and the badge truthfully said "live forecast".
**The test was right; the isolation was wrong.**

The cache-clearing fixture moved from `test_weather.py` into `conftest.py` as
autouse, so no future test file can reintroduce the leak by forgetting it.
Worth noting that this is the third time a *fixture* rather than the code has
been the bug — the pattern to watch for is a fixture that cannot distinguish
"the code is wrong" from "the setup is wrong".

### Testing

**325 tests with no network** (328 with `--live`), plus 1 strict xfail. New:
14 transcript, 13 analysis, 6 app.

The deployment-relevant one is
`test_a_failing_weather_api_still_answers_and_says_it_is_sample_data`: it runs
the app in *live* mode with a dead `requests.get`, which is the cloud-host
case — no outbound access, or Open-Meteo down. The app must not produce an
error page; it must answer from the demo dataset and say so. The fallback is
only acceptable *because* it is labelled.

### Open points

- The user tests themselves. The tooling is ready; the data is not collected
  yet.
- The `out_of_scope` quantity leak (`"how many people live in Linz"`,
  `"how high is the Dachstein"`) is still the one classifier error visible in
  normal use, still behind a strict xfail. The user tests are now the
  intended source of new `out_of_scope` phrasings — from real conversations,
  never from `reports/misclassified_*.csv`, which would leak the test set.
- The optional sentence-embedding comparison remains optional and remains not
  part of the submitted concept.
- `Place.matched_text` is still unused by the response generator.
