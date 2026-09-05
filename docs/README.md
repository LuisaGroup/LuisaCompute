# Documentation maintenance

`index.rst` is the single Sphinx entry point. Keep the existing guides in
`source/` and place subsystem detail under the topic that owns it:

```text
source/
  getting_started.md, tutorials.md, gallery.md   learning
  dsl.md, resources.md                           established programming guides
  tile/                                         Tile programming guide
    design.md, execution.md, layouts.md          overview and formal model
    values.md, memory.md, pipeline.md            operations and effects
    staging.md, kernels.md                      capture/JIT surface and examples
  internals/                                    compiler and Runtime implementation
    tile/                                       TileIR, export, Runtime and target plans
  performance/                                  measured results and validation
    tile/                                       current status, route results, checkpoints
```

- Introduce Tile from the existing DSL and architecture guides. Do not add each
  new Tile report or experiment as a peer of those guides in the root toctree.
- Keep a current status page short and answer-first. Put chronological updates
  in `performance/tile/checkpoints.md`, detailed measurements in the route or
  reduction reports, and raw runs in `scripts/benchmark/tile_torch/results/`.
- Distinguish proposed language contracts, implemented lowering, executed
  correctness tests and performance claims. Retain negative results and the
  original device/dtype/shape/timing/allocation/math-policy qualifications.
- Keep historical schedule tables and test-run counts with performance and
  validation evidence. Implementation references describe invariants and link
  to those records; they should not accumulate a second experiment diary.
- Give a long reference a single responsibility before adding another section.
  Tile syntax belongs in `source/tile/`; IR mutation/capture algorithms and
  bridge realization belong in `source/internals/tile/`. The generic planner,
  backend policy API, Metal matrix family, Metal reductions and TIRx CPU plans
  have separate owners. Cross-link them instead of extending a monolithic
  language or planner document.
- Label pseudocode and proposed extensions at their reading entry point.
  Keep the historical design checklist in the architecture decisions; it is
  not the current implementation plan or evidence of completion.
- Give long references a local contents list. After splitting a document,
  replace inherited cross-page chapter numbers with descriptive links, while
  preserving the published section anchors.
- Put diagrams in `_static/<topic>/`. Use relative MyST `figure` and
  `literalinclude` directives; use `download` for repository evidence outside
  the documentation tree.
- The canonical Tile example is `source/tile/tile_programming_poc.cpp`. Both
  CMake and XMake compile it through the capture tests; do not fork a prose-only
  copy when changing its syntax.
- Every reader page belongs in one owning toctree. Cross-link other reading
  paths without giving the same page multiple parents. `custom_agility_sdk.md`
  belongs to getting started; coroutine extensions belong to implementation.
- Preserve published URLs when moving pages. `_ext/legacy_urls.py` and the
  redirect template generate legacy HTML routes without duplicate Markdown
  pages. `SPLIT_FRAGMENTS` preserves old section links when a landing page
  remains in use; the shared layout redirects only those moved fragments.
  Every old section must still exist or resolve to its new owner.

Build from the repository root:

```sh
doxygen docs/Doxyfile
uv run --no-project --python 3.13 --with sphinx --with sphinx-rtd-theme \
  --with myst-parser --with breathe sphinx-build -b html -W --keep-going \
  docs docs/_build/html
uv run --no-project --python 3.13 --with sphinx \
  python scripts/check_docs.py docs/_build/html
```

Doxygen is required for the generated API reference (`docs/output/xml`). A
machine without it can preview the handwritten pages, but that preview does
not pass the complete documentation build: do not suppress missing-XML warnings
and report it as full API-reference validation.

Use a fresh output directory for publication checks; an incremental build can
retain obsolete HTML from deleted source files. Check rendered navigation,
diagrams, tables, example inclusions and old-URL redirects, not just the build
exit code. Generated `_build/` content is not committed.

The Sphinx extension rejects missing or multiple owning toctrees.
`scripts/check_docs.py` checks local links, assets and published compatibility
anchors (including split references) in generated HTML. It does not check
remote links or substitute for visual review and the Doxygen/API build. Run
it on the same fresh output tree you intend to publish.
