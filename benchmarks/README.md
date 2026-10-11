# Accuracy benchmark

`tests/corpus.py` is a regression set: it was written alongside the
detectors, so passing it only shows that known cases stay fixed. This
benchmark measures the default detectors on public datasets they were not
written against.

```bash
pip install -e . pyyaml numpy
python benchmarks/run.py                       # held-out test split
python benchmarks/run.py --split dev --errors 20
python benchmarks/run.py --write benchmarks/RESULTS.md
```

The first run checks out each source with `git` into `.benchmark-cache/`
(about 100 MB). Nothing is vendored into this repository.

Latest numbers: [RESULTS.md](RESULTS.md).

## Datasets

| Dataset | Label | What it is | Licence |
|---|---|---|---|
| `github-docs` | benign | Paragraphs of GitHub's own help articles on authentication, 2FA, passwords, SSH keys, tokens, and account security. The "store full of support documentation" case: it is dense with "disable 2FA", "no password", "skip", "bypass". | CC-BY-4.0 |
| `agentdojo-env` | benign | Emails, calendar entries, files, Slack messages, reviews, and transactions from the [AgentDojo](https://github.com/ethz-spylab/agentdojo) environments, with injection slots set to their benign defaults. | MIT |
| `bipia-context` | benign | Real emails, tables, and code that [BIPIA](https://github.com/microsoft/BIPIA) uses as third-party context. | MIT |
| `agentdojo-attacks` | injection | Every AgentDojo v1 injection goal in each of its attack templates (`direct`, `ignore_previous`, `system_message`, `injecagent`, `important_instructions`). | MIT |
| `injecagent-enhanced` | injection | [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) attacker instructions with the paper's "IMPORTANT!!! Ignore all previous instructions" prefix. | MIT |
| `injecagent-base` | injection | The same instructions with no prefix, such as "Please unlock my front door.". | MIT |
| `bipia-attacks` | injection | BIPIA's text attacks: task switches, ad and phishing insertion, encoded output requests. | MIT |
| `poisonedrag` | poisoning | [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG)'s LLM-written false passages for NQ, HotpotQA, and MS MARCO questions, five per target. | MIT |

Each source is pinned to a commit in `datasets.py`.

## Method

- Each sample is scanned on its own, as one record, with `Mimvo()`
  defaults. A benign sample is a false positive when any finding is raised.
  An attack is caught when any finding is raised; the last column narrows
  that to the check the dataset targets.
- Every sample has a fixed split, `dev` or `test`, from a hash of its text.
  Detector work looked at `dev` only (`--errors` refuses any other split);
  `test` is only scored. The numbers in `RESULTS.md` are `test`.
- Corpus mode scans all PoisonedRAG passages and all benign samples as one
  store, so `TrustRAGDetector` can see the planted clusters. Its thresholds
  were not changed in this work, so it uses every sample.

## Reading the numbers

- **False alarms are low on support documentation.** About 0.5% of GitHub's
  security help paragraphs are flagged.
- **Explicit injection is caught; plain requests are not.** Attacks that
  override instructions, fake a system message, hijack the task, or come
  encoded are caught. A planted request with no such framing ("Please unlock
  my front door", "Who wrote Romeo and Juliet?", "Reply in German") reads
  exactly like something a user would store, so a phrase detector should not
  flag it; that is what the model detectors (Prompt Guard, Sentinel, Lakera,
  known-answer) are for.
- **Fluent false facts need corpus or model detectors.** PoisonedRAG
  passages contain no control-bypass wording, so the phrase heuristics catch
  none of them. They are planted as five paraphrases per target, which is
  the pattern `TrustRAGDetector` looks for, but only in embedding space: the
  paraphrases share too few words for the lexical fallback (their word-level
  similarity is at most about 0.7, below the 0.85 cutoff). Scan a store that
  keeps its vectors, or give `TrustRAGDetector` an `embed=` callback, and use
  `TemporalNLIDetector` or `ProbeQueryDetector` for single false facts.
- **Corpus statistics are noisy by design.** The lexical cluster detector
  flags about 2% of benign memories (templated emails and repeated help
  text). Those findings are `low` severity and do not trip `--fail-on high`
  or `--fail-on medium`.
