# PathQuestion Family

Reproducible reconstruction of the **PathQuestion (PQ)** and **PathQuestion-Large (PQL)** knowledge-graph question-answering benchmarks for graph-navigation research.

This repository contains a reconstruction pipeline and a **pinned Git submodule** pointing to the original [IRN/PathQuestion release](https://github.com/zmtkeke/IRN/tree/master/PathQuestion). It intentionally does **not** commit the original benchmark files or regenerated QA/graph outputs.

## Requirements

- **Python 3.12.12** (`.python-version`)
- `pandas==2.2.2` (`requirements.txt`)
- Git with submodule support

## Quick start

```bash
git clone --recurse-submodules https://github.com/HernandezEduin/PathQuestion-Family.git
cd PathQuestion-Family
python --version             # recommended: Python 3.12.12
python -m pip install -r requirements.txt
python scripts/reconstruct.py --dataset all --graph both --seed 42
```

All generated files are written under **`generated/`** (ignored by Git). The command reconstructs each dataset's **filtered** (released task-graph) and **source** (original knowledge graph) conditions with the same random seed and family-based data splits.

If you cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

### Common options

```bash
# Only PQ, on the released filtered graph (much smaller/faster):
python scripts/reconstruct.py --dataset pq --graph filtered --seed 42

# Only PQL, using both filtered and source graphs:
python scripts/reconstruct.py --dataset pql --graph both --seed 42

# Encode node/relation IDs in navigation-facing fields:
python scripts/reconstruct.py --dataset all --graph both --seed 42 --encode-ids
```

Use `python scripts/reconstruct.py --help` for `--split-ratio` and `--allow-source-coverage-gaps` options.

**Source graph preparation:** PQ uses `Freebase13.txt` in the original submodule. The upstream PQL source is distributed as `FB2M_TRI_NEW.txt.zip`; the wrapper extracts only the expected text member into the ignored `generated/source/` directory when the PQL source graph is requested. No upstream files are modified.

### Output organization

- `generated/link_prediction/PQ/` and `generated/link_prediction/PQL/`: filtered/source triplets and optional shared ID mappings.
- `generated/qa/PQ/` and `generated/qa/PQL/`: navigation-ready QA splits, families, memberships, reference paths, analyses, and conversion metadata.
- `generated/manifests/`: execution provenance and SHA-256 checksums for local reproducibility checks.
- `generated/source/`: locally extracted PQL source data.

The converter creates training, development, and test splits **by question family**, stratified within hop counts, to avoid family leakage. The **same split assignment** is used under filtered and source graph conditions. It preserves the original released answer annotations while separately recording graph-consistent answers and paths, rather than silently replacing released supervision.

### Reproducibility and caveats

The complete reconstruction implementation is included unchanged as `scripts/pathqa_2_kgqa.py`. Its default seed is 42, and `--graph both` constructs both graph regimes with shared encoding when `--encode-ids` is enabled. A seed is **not** sufficient by itself: use the pinned upstream submodule commit, Python/package versions, options, and recorded hashes to reproduce a run.

The converter checks that released annotated paths exist in the filtered graph, and by default rejects source-graph coverage gaps. Only use `--allow-source-coverage-gaps` after examining the logged discrepancies and their consequences. Source-graph reconstructions can be memory- and time-intensive.

Run synthetic unit tests (which do not require the upstream data):

```bash
python -m unittest discover -s tests -v
```

## Source, citation, and rights

Original benchmark: Mantong Zhou, Minlie Huang, and Xiaoyan Zhu. **An Interpretable Reasoning Network for Multi-Relation Question Answering.** COLING 2018. [Paper](https://aclanthology.org/C18-1171/) · [Original release](https://github.com/zmtkeke/IRN).

The original IRN release did not declare an explicit PathQuestion dataset license when this repository was prepared. **Our MIT license covers only code and documentation that we have the right to license.** It does not license the original benchmark, source knowledge graphs, or generated derivatives containing upstream material. See [THIRD_PARTY.md](THIRD_PARTY.md).

A submodule points users to the original distribution; it does **not** grant additional rights to redistribute the original dataset or its derivatives.
