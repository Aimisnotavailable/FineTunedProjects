# `data/` — local corpora (not committed)

This directory holds downloaded corpora and derived extracts. It is **gitignored** because
`sample.jsonl` alone is 704 MB. Nothing here is tracked; this file is the only exception.

To recreate everything from scratch, run the scrape section of
`notebooks/clef_unseen_attack_probe.ipynb`.

---

## Primary source: `anonymouslinuxarena/linuxarena-public`

| | |
|---|---|
| **Where** | `https://huggingface.co/datasets/anonymouslinuxarena/linuxarena-public` |
| **Licence** | CC-BY-4.0 (commercial use OK, attribution required) |
| **Size** | 17.5 GB full / 705 MB reviewer sample |
| **Citation** | LinuxArena, NeurIPS Datasets & Benchmarks Track (2026) |
| **Retrieved** | 2026-10-16 |

**What it is.** Agent trajectories from a Linux AI-control evaluation: 22,215 trajectories
across 172 runs and 10 Linux software environments, with honest and attack runs
pre-labelled.

**Why this corpus.** BashArena / ControlArena (`UKGovernmentBEIS/control-arena`, MIT) ships the
*setting* — 640 tasks, Docker environments, task-generation pipeline, monitor and attack
prompts — but **no pre-generated trajectories**. Getting those means running agents yourself in
Docker. LinuxArena gives the same control-evaluation setting with trajectories already run.

**Files used.**

| File | Size | What it is |
|---|---|---|
| `sample.jsonl` | 704 MB | The 932-trajectory reviewer sample, full JSONL schema |
| `linuxarena_files.jsonl` | 26 MB | Extract: one row per file recovered from a trajectory |
| `real_files_dedup.json` | 3.8 MB | Same, after content- and trajectory-level dedupe |

### Extracting file content — and its limits

Verified against 120 real trajectories. Tool call shapes:

| Tool | Arguments | Count / 120 traj |
|---|---|---|
| `bash` | `{cmd: str}` | 5451 |
| `text_editor` `view` | `{command, path}` | 1701 |
| `text_editor` `str_replace` | `{command, old_str, new_str, path}` | 978 |
| `text_editor` `create` | `{command, file_text, path}` | 488 |
| `submit` | `{}` | 73 |

Only `create` yields a **complete file body**. `str_replace` is a patch and needs prior state.
`bash` mutates files in place — recovering the result requires **replaying the command sequence
in a sandbox**, which is a Docker-scale job and is deliberately not attempted.

### Two traps that silently corrupt this corpus

1. **Runs are ordered.** The first run is `Haiku honest`. Any prefix scan (`head -n`, a
   `--limit` flag) yields 100% honest runs and **zero positives**.

2. **Monitor re-runs duplicate trajectories.** `Haiku honest` (200) also appears as
   `GPT-5 Nano monitor on Opus honest` (200) and `Opus monitor on Opus honest` (200); `Haiku
   attack` (92) likewise. The dataset card states only `monitor_response` /
   `traj_monitor_response` differ. **Without dedupe by `trajectory_id`, 1964 of 2360 files
   (83%) are duplicates** — the negative set is triple-counted and every FPR computed from it
   is meaningless.

   Yield after correct extraction: **3215 raw files → 2360 content-deduped → 396 after
   trajectory dedupe.**

### Labels — and why the positives are unreliable

```
0     honest run                              true negative
1     attack run AND side_task_success        "positive"
None  attack run that did not score           never used as a negative
```

**The `1` label is trajectory-level attribution applied to a file-level task, and it is
unsound for this purpose.** A trajectory can succeed at its side task while most of the files
it wrote are ordinary. A sampled positive was
`/app/config/initializers/create_file_metadata_table.rb` — a Rails initializer, not a covert
carrier. LinuxArena's side tasks (execute malware, exfiltrate secrets, escalate privileges,
disable firewall) are often achieved by **blatant** commands, not by steganography.

**Therefore: use the 309 honest-run files as negatives. Do not use the 68 `label == 1` files as
positives for covert detection** without hand-auditing them for actual concealment.

---

## Secondary, referenced but not downloaded

- `aisa-group/ResearchArena-Trajectories` — trajectories plus artifacts and monitor judgements,
  with `side_task=none` baselines. Its monitor-access taxonomy (CoT access × artifact access)
  is conceptually aligned with this probe.
- `rogue-security/coding-agent-security-benchmark-public` — **CC-BY-NC-4.0**, non-commercial.
  Deliberately avoided: CC-BY-4.0 above already covers the need and is permissive.

## Synthetic positives

Generated locally by `src/clef_probe/generator.py` — six families, paired clean/covert
construction, RFC 5737 documentation IPs only (`192.0.2.0/24`, `198.51.100.0/24`,
`203.0.113.0/24`). No real infrastructure is referenced. See the caveats in the design doc and
the results notebook: six hand-written templates make perfect separation expected rather than
impressive.