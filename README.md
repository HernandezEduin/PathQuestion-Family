# PathQuestion-Family

**A reproducible, family-controlled reconstruction of PathQuestion (PQ) and PathQuestion-Large (PQL) for multi-hop knowledge-graph question answering and explicit graph navigation.**

This repository supplies the reconstruction code used for the study **_How Benchmark Graph Construction Shapes Multi-Hop KG Navigation: A Controlled Audit of PathQuestion and PathQuestion-Large_**. The associated public arXiv preprint will be linked here when available.

The study compares the original **task-filtered** PQ/PQL knowledge graphs with the larger **source/original** graph artifacts distributed by the benchmark authors. It holds the natural-language questions, topic entities, released valid-answer sets, annotated relation chains and paths, and family-level train/development/test splits fixed. Only the active navigation graph changes. Graph-consistent answers and path realizations are then recomputed for that graph; they are kept distinct from the released evaluation targets.

> **Distribution model:** This is a **code-first reconstruction repository**, not a downloadable dataset archive or a complete reproduction of the paper's model experiments. The original IRN benchmark is referenced through a pinned Git submodule, and reconstructed data are generated locally and excluded from this repository's Git history. See [Data provenance and licensing](#data-provenance-and-licensing).

## Contents

- [Installation and reconstruction](#installation-and-reconstruction)
- [Original benchmark files](#original-benchmark-files)
- [Reconstruction protocol](#reconstruction-protocol)
- [Generated files](#generated-files)
- [QA schema](#qa-schema)
- [Reference reconstruction statistics](#reference-reconstruction-statistics)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Reproducibility and validation](#reproducibility-and-validation)
- [Scope of this release](#scope-of-this-release)
- [Citation](#citation)
- [Data provenance and licensing](#data-provenance-and-licensing)

## Installation and reconstruction

### Environment

The environment specified for the manuscript's reconstruction is:

| Component | Version |
| --- | --- |
| Python | **3.12.12** |
| pandas | **2.2.2** |

`pandas` is the only third-party Python dependency of the converter. The reconstruction has also been run successfully on Windows with Python 3.9, but **Python 3.12.12 is the reference environment** for reproducing the study.

Clone the repository **including its original-data submodule**:

```bash
git clone --recurse-submodules https://github.com/HernandezEduin/PathQuestion-Family.git
cd PathQuestion-Family
python --version
python -m pip install -r requirements.txt
```

If you already cloned without the submodule:

```bash
git submodule update --init --recursive
```

The upstream IRN revision is pinned to:

```text
32f70d311978bfc1b911c60dff3e9d291b9a2447
```

[View the pinned original PathQuestion release](https://github.com/zmtkeke/IRN/tree/32f70d311978bfc1b911c60dff3e9d291b9a2447/PathQuestion).

### Reproduce the manuscript's dataset reconstruction

Run **both** graph conditions with the manuscript's **80/10/10 family split**, seed **42**, and **shared entity/relation ID encoding** within each dataset:

```bash
python scripts/reconstruct.py --dataset all --graph both --seed 42 --split-ratio 0.8 --encode-ids
```

This works as a single command in Linux/macOS shells and in **Windows PowerShell**, provided the active Python environment has the dependency installed.

To run a subset:

```bash
# PQ only, filtered and source graphs
python scripts/reconstruct.py --dataset pq --graph both --seed 42 --encode-ids

# PQL only, filtered and source graphs
python scripts/reconstruct.py --dataset pql --graph both --seed 42 --encode-ids

# Smaller filtered-graph-only run
python scripts/reconstruct.py --dataset pq --graph filtered --seed 42 --encode-ids
```

Use `python scripts/reconstruct.py --help` to inspect the wrapper's flags:

| Option | Purpose |
| --- | --- |
| `--dataset {pq,pql,all}` | Select benchmark(s); default `all` |
| `--graph {filtered,source,both}` | Select active graph condition(s); default `both` |
| `--seed` | Seed for reproducible family splitting and canonical-question selection; default `42` |
| `--split-ratio` | Train proportion (the remainder is divided into dev and test); default `0.8` |
| `--encode-ids` | Encode entity and relation identifiers in navigation-facing outputs |
| `--allow-source-coverage-gaps` | Explicitly allow source-graph coverage gaps after reviewing the resulting diagnostics |

**Important:** `--encode-ids` is **not enabled by default**. Include it to produce the encoded navigation artifacts and shared ID mappings described in the paper. When comparing filtered and source graph conditions, use `--graph both` in the **same run** so the two conditions share one entity/relation encoding for the selected dataset.

### The two reconstruction scripts

- [`scripts/reconstruct.py`](scripts/reconstruct.py) is the public entry point. It finds the pinned upstream release, prepares the PQL source archive when needed, invokes the converter, and saves a run manifest with output SHA-256 checksums.
- [`scripts/pathqa_2_kgqa.py`](scripts/pathqa_2_kgqa.py) is the underlying benchmark converter used for parsing, question-family construction, graph restoration, reference bookkeeping, split assignment, structural validation, and QA/graph export.

The wrapper automatically extracts the `FB2M_TRI_NEW.txt` member of the upstream PQL `.zip` archive to the local `generated/source/` directory. It does **not** modify the pinned IRN submodule. PQ's `Freebase13.txt` is read directly from that submodule.

The converter may also be run directly; unlike the wrapper, it expects the PQL source `.txt` file to have been extracted or supplied through `--source-kg-file`. See its help text:

```bash
python scripts/pathqa_2_kgqa.py --help
```

## Original benchmark files

PQ and PQL were introduced by Zhou, Huang, and Zhu (COLING 2018) with the Interpretable Reasoning Network (IRN). They were constructed from Freebase subsets using relation paths and template-based questions with paraphrase/synonym variation.

The pinned original [`external/IRN/PathQuestion/`](external/IRN/PathQuestion) supplies the following reconstruction inputs:

| Dataset | Purpose | Upstream file(s) |
| --- | --- | --- |
| PQ | Released questions, answers, and annotated paths | `PQ-2H.txt`, `PQ-3H.txt` |
| PQ | Hop-specific task-filtered graphs | `2H-kb.txt`, `3H-kb.txt` |
| PQ | Distributed source/original graph | `Freebase13.txt` |
| PQL | Released questions, answers, and annotated paths | `PQL-2H.txt`, `PQL-3H.txt` |
| PQL | Hop-specific task-filtered graphs | `PQL2-KB.txt`, `PQL3-KB.txt` |
| PQL | Distributed source/original graph | `FB2M_TRI_NEW.txt.zip` |

The released 2-hop and 3-hop tasks are combined to form the **mixed-hop** PQ and PQL evaluation setting used here. The larger source graphs are *distributed benchmark artifacts*, **not complete Freebase**. “Large” in PQL concerns the graph setting rather than a larger number of QA records.

## Reconstruction protocol

### Matched graph conditions

For each dataset, the converter constructs two directed KG conditions:

- **Filtered (`GF`):** the merged hop-specific task-filtered graphs distributed with the benchmark.
- **Source (`GS`):** the corresponding larger source/original graph distributed with the benchmark, after the required normalization (particularly for PQL identifiers).

The graph construction intervention holds the **question, topic entity, released answer annotations, released reference relation chains and paths, family identity, and split assignment fixed**. Source restoration can add outgoing actions, reachable intermediates, endpoints, and alternative path realizations. These effects are measured rather than repaired away.

### Question families, QA instances, and splitting

Structural families group related released records before splitting so that paraphrased or structurally shared examples do not leak across train/dev/test boundaries.

- Both PQ and PQL group released records that share **topic entity, relation chain, and hop count**, merging answer-specific released references as needed.
- For PQL, identical **question + topic entity + hop count** records with more than one released relation chain also remain within one family and one evaluation instance.
- Family-aware 80/10/10 splitting is stratified within hop counts. Related textual QA variants remain in the same split.
- Evaluation **instances** preserve the released textual problems; **families** provide the grouping for split control and family-macro evaluation. The two counting units are deliberately distinct.

The original release includes multiple valid answers and annotated paths, repeated-node walks, immediate backtracking, and (in a small number of PQL cases) more than one valid released relation chain for the same textual QA problem. This reconstruction retains those cases rather than forcing a single path or excluding cycles.

### Fixed released targets vs graph-relative realizations

The original released answers and annotated paths are the **fixed primary evaluation references** in either graph. Independently, the converter follows each retained released relation chain over the **active** graph to enumerate:

- **Graph-consistent answers:** endpoints reachable through those relations.
- **Graph-consistent paths:** entity-level realizations of those relations.

These graph-relative objects are **structurally consistent with the relation chains, not independently human-verified semantic answers to the natural-language questions**. Replacing the released gold answers with all newly reachable endpoints would change the evaluation target and invalidate the intended controlled comparison.

All graph-relative paths follow the released **unrestricted-walk** semantics: entity revisits and immediate backtracking are permitted.

## Generated files

All reconstruction outputs are written into **`generated/`**, which is excluded by `.gitignore`. This is an output generated on your machine, **not a dataset directory committed to this repository**.

For `--dataset all --graph both --encode-ids`, the relevant layout is:

```text
generated/
├── link_prediction/
│   ├── PQ/
│   │   ├── filtered/
│   │   │   ├── triplets.txt
│   │   │   ├── node_data.csv
│   │   │   └── relation_data.csv
│   │   └── source/             # Same file types; shared ID mapping
│   └── PQL/
│       ├── filtered/
│       └── source/
├── qa/
│   ├── PQ/
│   │   ├── filtered/           # QA tables, QA metadata, regime audit, JSON metadata
│   │   ├── source/
│   │   ├── pq_question_families.csv
│   │   ├── pq_question_family_membership.csv
│   │   ├── pq_released_path_answer_provenance.csv
│   │   ├── pq_source_normalization_collisions.csv
│   │   └── pq_conversion.log
│   └── PQL/                   # Corresponding PQL files and regime folders
├── manifests/
│   ├── pq_manifest.json
│   └── pql_manifest.json
└── source/
    └── FB2M_TRI_NEW.txt       # Local extraction from upstream archive
```

The converter produces additional QA provenance/diagnostic CSVs at the respective dataset root, including annotated-path validity, ambiguity checks, and evaluation-reference-path exports. The table below lists the main types of files.

| Generated file | Location and meaning |
| --- | --- |
| `triplets.txt` | `link_prediction/<dataset>/<regime>/`; active directed triples, encoded when requested |
| `node_data.csv`, `relation_data.csv` | Same KG-condition directory, when `--encode-ids` is set; entity `Title`/`EID` and relation `Title`/`RID` mappings, identical between filtered/source within a dataset in a combined run |
| `<dataset>_qa_nhop.csv` | `qa/<dataset>/<regime>/`; compact mixed-hop QA instance table |
| `<dataset>_qa_nhop_metadata.csv` | Same folder; fuller per-instance annotations, reference provenance, split/family weights, graph-reachability and cycle diagnostics |
| `<dataset>_qa_2hop.csv`, `<dataset>_qa_3hop.csv` | Same folder; hop-specific subsets and companion metadata CSVs |
| `<dataset>_cycle_answer_audit.csv` | Same folder; topic-answer identity, cycles, and graph-dependent answer behavior |
| `conversion_metadata.json` | Same folder; configuration, count summaries, output-column definitions, path nesting semantics, and validation information |
| `<dataset>_question_families.csv` | `qa/<dataset>/`; family-level data and aggregation references |
| `<dataset>_question_family_membership.csv` | `qa/<dataset>/`; released source-row membership and retained annotation provenance |
| `<dataset>_released_path_answer_provenance.csv` | `qa/<dataset>/`; provenance connecting annotated chains/paths to designated answers and original rows |
| `<dataset>_source_normalization_collisions.csv` | `qa/<dataset>/`; source identifier-normalization collision audit (for source-graph runs) |
| `<dataset>_conversion.log` | `qa/<dataset>/`; converter run log and validation details |
| `{pq,pql}_manifest.json` | `manifests/`; Python version, run options, upstream revision, and SHA-256 checksums of local outputs |

`<dataset>` denotes lowercase `pq` or `pql` in filenames, and uppercase `PQ` or `PQL` in directory names.

**Single-regime runs:** When `--graph filtered` or `--graph source` is selected instead of `both`, the underlying converter writes graph and QA files directly into `link_prediction/<dataset>/` and `qa/<dataset>/` without additional `filtered/` or `source/` subdirectories. Use a fresh output directory or a separate checkout if you need to avoid mixing artifacts from different run modes.

**Older wrapper versions:** Before the output-root correction, runs could write nested `PQ/PQ` or `PQL/PQL` directories. The current wrapper avoids that nesting; it does not delete previous local outputs.

## QA schema

The compact `*_qa_nhop.csv` export contains the following important fields:

| Column(s) | Semantics |
| --- | --- |
| `Question-Number`, `Question`, `Question-Paraphrased` | Reconstructed instance number, the released question, and recorded textual variants |
| `Question-Family-ID`, `Question-Family-Size`, `SplitLabel` | Family identity/size and family-controlled train/dev/test assignment |
| `Source`, `Source-Entity` | Topic entity label and navigation-facing topic entity (ID when encoded) |
| `Answer`, `Answer-Entity` | **Released** valid-answer set, in label and navigation-facing forms; the fixed primary target |
| `Graph-Answer`, `Graph-Answer-Entity` | Endpoints realized by released relation chains over the **active** graph, not independently verified semantic gold |
| `Multi-Paths`, `Multi-Paths-Label`, `Multi-Paths-Key` | Released annotated paths (encoded/labeled) grouped by the unique retained relation-chain keys |
| `Path-Count` | Number of distinct released annotated reference paths across groups |
| `Graph-Multi-Paths`, `Graph-Multi-Paths-Label`, `Graph-Path-Count` | Active-graph path realizations grouped by those same relation-chain keys, and their total count |
| `Reference-ID`, `Hops` | Released source-row/path provenance and annotated hop count |

**Path nesting matters:** For one QA row, `Multi-Paths-Key[i]` identifies relation-chain group `i` in the corresponding `Multi-Paths` and `Graph-Multi-Paths` collections. Each path is an ordered list of `[head, relation, tail]` triples. `Reference-ID` follows the grouped annotated path structure and records contributing released source-row numbers. Consult the generated `conversion_metadata.json` for the precise nesting definition.

List-like columns may contain nested collections serialized inside CSV cells. First parse CSV fields properly; **do not split those fields on commas** to recover nested objects. The companion `*_qa_nhop_metadata.csv` stores additional detailed diagnostics and stable join identifiers.

PQ and PQL have distinct entity and relation ID spaces. **Do not mix IDs between datasets**; shared IDs are guaranteed only *within* the filtered/source conditions of a dataset when encoded together.

## Reference reconstruction statistics

The following counts are reported in the accompanying study for **seed 42**, family-controlled reconstruction with both graph regimes. They are provided as **reference checks** for the local output, not as precomputed data files distributed by this repository.

| Quantity | PQ | PQL |
| --- | ---: | ---: |
| Original released QA records | 7,106 | 2,625 |
| Reconstructed QA instances | 7,105 | 2,114 |
| Structural question families | 2,923 | 1,530 |
| Train families | 2,337 | 1,223 |
| Dev families | 292 | 153 |
| Test families | 294 | 154 |
| Filtered KG: unique triples | 3,377 | 5,597 |
| Source/original KG: unique triples | 375,444 | 831,940 |
| Filtered KG: entities | 2,256 | 6,505 |
| Source/original KG: entities | 75,043 | 416,249 |
| Filtered KG: relation types | 13 | 411 |
| Source/original KG: relation types | 13 | 3,076 |

The counts distinguish *released records*, *reconstructed textual QA instances*, and *structural families*. For example, PQ merges one duplicated exact QA group into a single multi-answer instance; PQL retains multi-chain references for several same-question cases. Counts refer to the reconstruction protocol here, not necessarily to the preprocessing used in earlier PQ/PQL publications.

## Interpretation and limitations

- **Released vs graph-relative evaluation:** `Answer-Entity` and the released `Multi-Paths` are primary fixed references. `Graph-Answer-Entity` and `Graph-Multi-Paths` depend on the graph and should not automatically replace them.
- **Family macro vs instance micro:** Family-macro evaluation gives each structural family equal weight; instance-micro evaluation reflects the distribution of textual QA instances. The two aggregations answer different questions.
- **Paths need not be simple:** The original annotations include revisits, immediate backtracking, and self-returning paths; this release preserves them.
- **Source graphs are benchmark artifacts:** Neither `Freebase13.txt` nor `FB2M_TRI_NEW.txt` is presented here as complete Freebase. “Source restoration” refers to these available distributed graphs.
- **Question wording:** Template-generated text and released relations are not guaranteed to be semantically aligned in every case. Exploratory wording-audit flags discussed in the paper should not be interpreted as exhaustively human-validated benchmark error rates. The reconstruction leaves those questions and annotations intact.

## Reproducibility and validation

Run synthetic tests without the original datasets:

```bash
python -m unittest discover -s tests -v
```

For a complete run, inspect `generated/qa/PQ/pq_conversion.log` and `generated/qa/PQL/pql_conversion.log`, the corresponding `conversion_metadata.json` files, the question-family tables, and the output manifests.

The reconstruction performs checks for annotated-path validity, released answer/path correspondence, family consistency, graph coverage, and (where applicable) source identifier-normalization collisions. By default, missing released path triples or unsafe graph-normalization conditions are not silently accepted. Only use `--allow-source-coverage-gaps` if you understand the exceptions that the conversion reports.

A fixed seed alone is not an exact reproduction specification. Keep the **pinned submodule commit**, **Python and pandas versions**, **encoding choice**, and **conversion arguments** fixed as well. Output manifests record hashes for auditing; platform-specific paths, serialized metadata, and timestamped logs may differ across operating systems, so hash equality for *every* output file is not a cross-platform guarantee.

## Scope of this release

**Included in Git:** the reconstruction scripts, dependency specification, pinned upstream submodule reference, documentation, and synthetic tests.

**Generated locally:** active graph triplets, optional ID mappings, filtered/source QA records, family and membership tables, reference bookkeeping, coverage and cycle diagnostics, conversion logs, and manifests.

**Not included:** redistributed copies of the original PQ/PQL dataset; generated dataset CSVs or graph files; the anonymous supplement's separately packaged `results/kg_stats/` and `results/struct_cal/` summaries; its exploratory `wording_audit/` prompts and reports; trained-policy or frozen-model inference code; checkpoints; full model predictions; or full experiment outputs. Reproducing the manuscript's **model-based evaluations and structural calibration experiments** therefore requires additional software and configurations beyond this dataset-construction repository.

## Citation

If you use this reconstruction, cite both:

1. **The original benchmark:** Mantong Zhou, Minlie Huang, and Xiaoyan Zhu (2018), [_An Interpretable Reasoning Network for Multi-Relation Question Answering_](https://aclanthology.org/C18-1171/), COLING 2018.
2. **The associated benchmark audit:** _How Benchmark Graph Construction Shapes Multi-Hop KG Navigation: A Controlled Audit of PathQuestion and PathQuestion-Large_. A public arXiv link and full citation will be added once available.

The original IRN authors are responsible for the underlying benchmark release. The contributors to this repository are responsible for the reconstruction code and associated documentation, not for authorship of the original PQ/PQL datasets.

## Data provenance and licensing

The benchmark inputs come from the original [IRN repository](https://github.com/zmtkeke/IRN), referenced here as a pinned submodule, **not checked into this repository as original data files**.

As of October 2026, we could not identify an explicit license governing the original PathQuestion benchmark in the upstream repository. The source graphs are derived from Freebase, whose data has separate attribution conditions; these do not automatically license the benchmark authors' question text, curation, or other contributions.

The [MIT license](LICENSE) applies to **our independently licensed code and documentation only**, not to the upstream IRN repository, PQ/PQL data, or generated datasets incorporating them. Cloning a submodule does **not** resolve or confer additional redistribution permissions. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for the full provenance and licensing notice.
