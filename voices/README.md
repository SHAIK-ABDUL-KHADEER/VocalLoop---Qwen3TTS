# Vocalist voice library

Curated reference clips that users can pick from and clone — separate from the 9
built-in voices that ship with the CustomVoice model.

## Adding a voice

Drop an audio file in this folder. That is the whole requirement:

```
voices/
  aria_soul.wav      ->  voice_id "aria_soul"
  marcus_baritone.mp3 -> voice_id "marcus_baritone"
```

The **filename stem becomes the `voice_id`**. `.wav`, `.mp3`, `.flac`, `.ogg`,
`.m4a` and `.opus` are all recognised. It appears in `GET /v1/voices` → `vocalist`
immediately — no restart needed.

Then generate with it:

```json
POST /v1/speak
{ "text": "Sing me something new.", "voice_id": "aria_soul" }
```

## Clip guidance

- **3–10 seconds** of clean, single-speaker audio. Longer is not better.
- No background music, echo, or overlapping voices — the clone copies whatever is there.
- One consistent delivery. A clip that shifts between whispering and belting produces
  an unstable voice.

## voices.json — optional metadata

Without it, a voice still works; the API just shows a title-cased filename and no
description. Add entries to control what your picker displays:

```json
{
  "voices": [
    {
      "id": "aria_soul",
      "name": "Aria",
      "description": "Soulful female vocalist, warm mid-range",
      "language": "english",
      "ref_text": "Exact transcript of aria_soul.wav"
    }
  ]
}
```

| Field | Required | Purpose |
|---|---|---|
| `id` | yes | Must match the filename stem |
| `name` | no | Display name; defaults to a title-cased id |
| `description` | no | Shown in the picker |
| `language` | no | Advisory only; callers still pass their own `language` |
| `ref_text` | no | Transcript of the clip — see below |

## About `ref_text`

Cloning quality depends on the reference transcript being accurate. You do not have
to supply one: the first request for a voice transcribes the clip with Whisper and
**writes the result back into `voices.json`**, so it is only ever computed once.

Supplying `ref_text` yourself is still better when you know the exact words —
Whisper misreads unusual names, lyrics, and heavy accents. Compare:

```
Whisper:  "Hello, this is the Gwen 3 text-to-speech engine speaking."
Actual:   "Hello, this is the Qwen3 text to speech engine speaking."
```

That is a real transcription from testing. Good enough to clone from, but the exact
text is better.

> If the folder is read-only (some container setups), transcripts cannot be cached.
> Generation still works — it just re-transcribes each time and logs a warning.

## Licensing

You are responsible for having the right to clone any voice you add here. Get
explicit permission from the vocalist.
