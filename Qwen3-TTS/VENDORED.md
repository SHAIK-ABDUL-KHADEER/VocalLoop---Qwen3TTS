# Vendored copy of Qwen3-TTS

This folder is a checkout of the upstream Qwen3-TTS repository, committed here so
the whole project installs from one clone with nothing fetched from elsewhere.

| | |
|---|---|
| Upstream | <https://github.com/QwenLM/Qwen3-TTS> |
| Commit | `022e286b98fbec7e1e916cb940cdf532cd9f488e` |
| Date | 2026-03-17 |
| Subject | fix finetuning bug |
| Package version | `qwen-tts` 0.1.1 |
| License | Apache-2.0 — see [LICENSE](LICENSE) |

**This code is unmodified.** The working tree was clean at the commit above, and it
matches `qwen-tts` 0.1.1 on PyPI. Every workaround for the library's rough edges
lives in [`../server/app.py`](../server/app.py), never in here — so this folder can
be replaced wholesale with a newer upstream checkout without losing anything.

Install it in editable mode from the repo root:

```bash
pip install -e ./Qwen3-TTS
```

`pip install -U qwen-tts` from PyPI works equally well and is interchangeable at
this version. This copy exists so the repo is self-contained, and so the Gradio
demo (`qwen-tts-demo`), the examples, and the fine-tuning scripts come along too.

## Updating

```bash
rm -rf Qwen3-TTS
git clone https://github.com/QwenLM/Qwen3-TTS.git
rm -rf Qwen3-TTS/.git          # required, or git records a gitlink and it clones empty
pip install -e ./Qwen3-TTS
```

Then update the commit recorded above, and re-run `python test_endpoints.py` — the
API leans on library behaviour that upstream could change, particularly the audio
input handling described in [`../ENDPOINTS.md`](../ENDPOINTS.md).
