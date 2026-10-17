# Third-party datasets and licensing

## PathQuestion / PathQuestion-Large

The original PathQuestion and PathQuestion-Large (PQ/PQL) benchmark is distributed in [zmtkeke/IRN](https://github.com/zmtkeke/IRN/tree/master/PathQuestion), accompanying:

> Zhou, Mantong; Huang, Minlie; Zhu, Xiaoyan (2018). *An Interpretable Reasoning Network for Multi-Relation Question Answering*. COLING 2018.

The upstream repository does not declare an explicit license for the PathQuestion benchmark as checked in October 2026. The original data are referenced using a pinned Git submodule; they are not copied into this repository's own tracked blobs. This approach does **not** grant redistribution or other rights not already held by users. Running reconstruction locally and distributing derived material may involve separate copyright and other rights considerations.

The upstream source graphs are described as derived from Freebase. Google's Freebase data is available under Creative Commons Attribution terms, but that does not automatically license the original benchmark authors' question text, benchmark construction, or other third-party contributions.

## Scope of this repository's license

The MIT license covers the independently supplied processing scripts, tests, and project documentation to the extent their authors have the rights to license them. It **does not** cover the upstream IRN submodule, its benchmark data, or generated datasets containing material from upstream.

Questions regarding redistribution have been raised with the original maintainers; absent a clear grant, do not treat the upstream benchmark as MIT licensed.
