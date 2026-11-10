# User-test transcripts

Put the `.txt` files that user-test participants send you in this folder, then
run:

```bash
python scripts/analyze_transcripts.py
```

The `.txt` files themselves are **git-ignored**. They are other people's
conversations: working material for the evaluation, not part of the project,
and they have no place in a repository that gets submitted. Only this README is
tracked, so the folder exists in a fresh checkout.

## Where the files come from

A participant opens the app, has a conversation, optionally gives answers a
thumbs up or down, and presses **Download conversation** in the sidebar. The
file is written in their browser; nothing is stored on the server. They then
send it on themselves, which means they can read, edit or discard it first.

## What is in a file

A readable transcript plus, under each answer, what the bot recognised:

```
[2] 14:21:03 PEAKCAST
Wind on the Hochkönig tomorrow is around 16 km/h, with gusts up to 35 km/h.
    intent: ask_wind
    confidence: 0.92
    place: Hochkönig, Salzburg
    day: 2026-11-11
    source: live
    feedback: up
```

`peakcast/transcript.py` both writes and parses this format, and a round-trip
test keeps the two in step.
