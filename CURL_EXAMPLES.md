# curl examples — every flow

Every command below was run against a live server and returns the status shown.

Set these first:

```bash
API=http://127.0.0.1:9000
REF=https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav
```

If the server has `QWEN_TTS_API_KEY` set, add `-H "X-API-Key: $QWEN_TTS_API_KEY"` to every
request except `/health`.

> ## ⚠️ On Windows PowerShell, these will NOT work as written
>
> PowerShell 5.1 strips the double quotes out of native-command arguments before `curl.exe`
> ever sees them. `-d '{"text":"hi"}'` arrives as `{text:hi}`, and you get
> `400 Body must be JSON`. Verified with an echo server — this is not a server bug.
>
> Use **Git Bash** for the commands below, or see [PowerShell](#powershell) at the end.

---

## 1. Described voice — invent a voice from words

No voice source, so `instruct` is required.

```bash
curl -X POST $API/v1/speak \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Welcome, traveller. You have journeyed far.",
        "instruct": "An elderly wizard, deep gravelly voice, slow and wise.",
        "language": "english",
        "response_format": "wav"
      }' \
  --output described.wav
```

Minimal version — `language` defaults to auto-detect:

```bash
curl -X POST $API/v1/speak -H "Content-Type: application/json" \
  -d '{"text":"Bonjour, comment allez-vous?","instruct":"A warm, friendly woman.","response_format":"wav"}' \
  --output described_fr.wav
```

> **`response_format` defaults to `json`.** Leave it out and you get base64 in a JSON body —
> which `--output foo.wav` will happily save as a `.wav` your player cannot open. Every
> command here that writes a `.wav` sets `"response_format": "wav"`; keep it when you edit them.

## 2. Built-in voice

```bash
curl -X POST $API/v1/speak \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Hello, this is the Qwen3 text to speech engine speaking.",
        "speaker": "ryan",
        "language": "english",
        "response_format": "wav"
      }' \
  --output builtin.wav
```

## 3. Built-in voice + style instruction

Same endpoint; `instruct` restyles the speaker instead of inventing a voice.

```bash
curl -X POST $API/v1/speak \
  -H "Content-Type: application/json" \
  -d '{
        "text": "I cannot believe you did that. This is unacceptable.",
        "speaker": "vivian",
        "instruct": "Speak in a furious, sharp tone.",
        "language": "english",
        "response_format": "wav"
      }' \
  --output builtin_angry.wav
```

## 4. Vocalist voice from your library

```bash
curl -X POST $API/v1/speak \
  -H "Content-Type: application/json" \
  -d '{
        "text": "This line is spoken by a voice from the library.",
        "voice_id": "aria_soul",
        "language": "english",
        "response_format": "wav"
      }' \
  --output vocalist.wav
```

List the available ids first:

```bash
curl -s $API/v1/voices | python -m json.tool
```

## 5. Clone a voice by URL

`ref_text` omitted → the clip is transcribed automatically.

```bash
curl -X POST $API/v1/speak \
  -H "Content-Type: application/json" \
  -d "{
        \"text\": \"This sentence is spoken in a cloned voice.\",
        \"audio\": \"$REF\",
        \"language\": \"english\",
        \"response_format\": \"wav\"
      }" \
  --output clone_url.wav
```

With an exact transcript (better quality, and skips the ASR step):

```bash
curl -X POST $API/v1/speak \
  -H "Content-Type: application/json" \
  -d "{
        \"text\": \"Explicit transcript supplied.\",
        \"audio\": \"$REF\",
        \"ref_text\": \"Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.\",
        \"response_format\": \"wav\"
      }" \
  --output clone_reftext.wav
```

## 6. Clone from an uploaded file (multipart)

This is what a browser file input does. Note `-F`, not `-d`.

```bash
curl -X POST $API/v1/speak \
  -F "text=Cloned from an uploaded file." \
  -F "audio=@voices/aria_soul.wav" \
  -F "language=english" \
  -F "response_format=wav" \
  --output clone_upload.wav
```

With a transcript:

```bash
curl -X POST $API/v1/speak \
  -F "text=With an explicit transcript." \
  -F "audio=@voices/aria_soul.wav" \
  -F "ref_text=Every song starts somewhere quiet, long before anybody hears it." \
  -F "response_format=wav" \
  --output clone_upload_ref.wav
```

## 7. Clone from base64 / data URI

**What this is for:** callers that already hold the audio *bytes* and want to send one JSON
document instead of a multipart upload — a browser that read the file with `FileReader`, a
backend forwarding an upload it received, a clip pulled from a database or S3. It is the
same `voice_clone` mode as §5 and §6, just a different way to carry the clip.

**Not for the command line.** A 263 KB clip becomes ~358 KB of base64, and an OS caps the
total command line (~32 KB on Windows, ~2 MB on Linux), so inlining it fails with:

```
bash: /mingw64/bin/curl: Argument list too long
```

That is the shell refusing to launch the process — the request never reaches the server.
Put the body in a file and point curl at it with `-d @file`:

```bash
# build the request body (bare base64)
./venv/Scripts/python.exe -c "
import base64, json
b = base64.b64encode(open('voices/aria_soul.wav','rb').read()).decode()
json.dump({'text':'From base64.','audio':b,'response_format':'wav'}, open('body_b64.json','w'))"

curl -X POST $API/v1/speak -H "Content-Type: application/json" \
  -d @body_b64.json --output clone_b64.wav
```

The `data:` URI form is identical apart from the prefix:

```bash
./venv/Scripts/python.exe -c "
import base64, json
b = base64.b64encode(open('voices/aria_soul.wav','rb').read()).decode()
json.dump({'text':'From a data URI.','audio':'data:audio/wav;base64,'+b,
           'response_format':'wav'}, open('body_uri.json','w'))"

curl -X POST $API/v1/speak -H "Content-Type: application/json" \
  -d @body_uri.json --output clone_datauri.wav
```

Both accepted — the server rewraps bare base64 into a data URI internally, because the
upstream library misreads base64 containing `/` as a filename.

**From a browser, prefer multipart (§6)** — it streams the file instead of inflating it by
33% and buffering the whole thing as a string. Reach for base64 only when the bytes are
already in memory and multipart is awkward.

## 8. Batch — JSON only

`response_format: "wav"` cannot return several files, so batches come back as base64 JSON.
Other fields take one value or a list of matching length.

```bash
curl -X POST $API/v1/speak \
  -H "Content-Type: application/json" \
  -d '{
        "text": ["The first sentence is spoken here.",
                 "And the second follows after it."],
        "speaker": ["ryan", "aiden"],
        "language": "english"
      }' \
  -o batch.json
```

Decode the clips:

```bash
python -c "
import base64, json
for i, a in enumerate(json.load(open('batch.json'))['audio'], 1):
    open(f'batch_{i}.wav','wb').write(base64.b64decode(a))
print('written')"
```

## 9. Sampling parameters

Accepted by every mode.

```bash
curl -X POST $API/v1/speak \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Testing sampling parameters end to end.",
        "speaker": "aiden",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "do_sample": true,
        "repetition_penalty": 1.1,
        "response_format": "wav"
      }' \
  --output params.wav
```

Full list: `max_new_tokens`, `do_sample`, `top_k`, `top_p`, `temperature`,
`repetition_penalty`, `subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`,
`subtalker_temperature`.

## 10. JSON response instead of raw WAV

Drop `response_format` (defaults to `json`) to get metadata alongside the audio — duration,
which mode ran, and the transcript used when cloning.

```bash
curl -s -X POST $API/v1/speak -H "Content-Type: application/json" \
  -d "{\"text\":\"Metadata please.\",\"audio\":\"$REF\"}" \
  | python -c "
import json,sys
d = json.load(sys.stdin)
print('mode      :', d['mode'])
print('duration  :', d['durations'])
print('gen secs  :', d['generation_seconds'])
print('ref_text  :', d.get('ref_text'))
print('auto ASR  :', d.get('ref_text_auto_generated'))
print('audio b64 :', len(d['audio'][0]), 'chars')"
```

Show the mode header on a `wav` response:

```bash
curl -s -D - -o out.wav -X POST $API/v1/speak -H "Content-Type: application/json" \
  -d '{"text":"Check the headers.","speaker":"ryan","response_format":"wav"}' \
  | grep -i "^x-"
```

## 11. Metadata

```bash
curl -s $API/v1/voices | python -m json.tool     # built-in + vocalist + languages
curl -s $API/health   | python -m json.tool     # device, VRAM, resolved config
```

`/health` never requires the API key. Confirm `config.device` and `config.asr_device` are
`cuda:0` to be sure you are on the GPU.

## 12. Error cases (all expected)

```bash
# 422 - two voice sources
curl -s -X POST $API/v1/speak -H "Content-Type: application/json" \
  -d "{\"text\":\"x\",\"speaker\":\"ryan\",\"audio\":\"$REF\"}"

# 422 - no voice source and no instruct
curl -s -X POST $API/v1/speak -H "Content-Type: application/json" -d '{"text":"x"}'

# 404 - unknown vocalist (response lists the valid ids)
curl -s -X POST $API/v1/speak -H "Content-Type: application/json" \
  -d '{"text":"x","voice_id":"nope"}'

# 400 - wav format with a batch
curl -s -X POST $API/v1/speak -H "Content-Type: application/json" \
  -d '{"text":["a","b"],"speaker":"ryan","response_format":"wav"}'

# 401 - only when the server has QWEN_TTS_API_KEY set
curl -s -X POST $API/v1/speak -H "Content-Type: application/json" \
  -d '{"text":"x","speaker":"ryan"}'

# 400 - more than 32 items in one request (QWEN_TTS_MAX_ITEMS)
./venv/Scripts/python.exe -c "
import json; json.dump({'text':['hi']*40,'speaker':'ryan'}, open('too_many.json','w'))"
curl -s -X POST $API/v1/speak -H "Content-Type: application/json" -d @too_many.json

# 400 - a single item over 3000 characters (QWEN_TTS_MAX_CHARS)
./venv/Scripts/python.exe -c "
import json; json.dump({'text':'x'*4000,'speaker':'ryan'}, open('too_long.json','w'))"
curl -s -X POST $API/v1/speak -H "Content-Type: application/json" -d @too_long.json

# 502 - reference URL the server cannot fetch
curl -s -X POST $API/v1/speak -H "Content-Type: application/json" \
  -d '{"text":"x","audio":"https://example.invalid/nope.wav"}'
```

Two more you will not trigger by hand, but your site should handle:

- **`503 Server is at capacity`** with a `Retry-After: 5` header — the queue is full. Back off
  and retry; do not hammer.
- **`504 Generation exceeded 600s`** — a single request too large to finish. Split the text.

---

## PowerShell

PowerShell 5.1 destroys quotes in native-command arguments, so `curl.exe -d '{"a":"b"}'`
sends `{a:b}`. Two ways round it.

### Option A — put the body in a file (works with plain curl)

```powershell
'{"text":"Hello there.","speaker":"ryan","response_format":"wav"}' |
  Set-Content body.json -Encoding ascii

curl.exe -X POST http://127.0.0.1:9000/v1/speak `
  -H "Content-Type: application/json" `
  -d "@body.json" --output out.wav
```

`-d "@file"` tells curl to read the body from disk, so PowerShell never touches the JSON.

### Option B — use PowerShell's own client (cleaner)

```powershell
$body = @{
  text            = "Hello there."
  speaker         = "ryan"
  language        = "english"
  response_format = "wav"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:9000/v1/speak -Method Post `
  -ContentType "application/json" -Body $body -OutFile out.wav
```

With an API key, add `-Headers @{ "X-API-Key" = $env:QWEN_TTS_API_KEY }`.

Multipart upload is fine either way, because no JSON quoting is involved:

```powershell
curl.exe -X POST http://127.0.0.1:9000/v1/speak `
  -F "text=Cloned from an uploaded file." `
  -F "audio=@voices/aria_soul.wav" `
  -F "response_format=wav" --output clone.wav
```

---

See [ENDPOINTS.md](ENDPOINTS.md) for the full field reference and
[DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md) for calling the deployed server.
