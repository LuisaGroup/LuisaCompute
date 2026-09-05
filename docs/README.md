# Documentation maintenance

`index.rst` is the single Sphinx entry point. Keep the existing guides in
`source/` and place subsystem detail under the topic that owns it:

```text
source/
  getting_started.md, tutorials.md, gallery.md   learning
  dsl.md, resources.md                           established programming guides
  tile/                                         Tile programming guide and syntax
  internals/                                    compiler and Runtime implementation
    tile/                                       Tile planning and lowering references
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
  pages. Split-page section links must also resolve to their new owner.

Build from the repository root:

```sh
doxygen docs/Doxyfile
uv run --no-project --python 3.13 --with sphinx --with sphinx-rtd-theme \
  --with myst-parser --with breathe sphinx-build -b html -W --keep-going \
  docs docs/_build/html
```

Doxygen is required for the generated API reference (`docs/output/xml`). A
machine without it can preview the handwritten pages, but that preview does
not pass the complete documentation build: do not suppress missing-XML warnings
and report it as full API-reference validation.

Use a fresh output directory for publication checks; an incremental build can
retain obsolete HTML from deleted source files. Check rendered navigation,
diagrams, tables, example inclusions and old-URL redirects, not just the build
exit code. Generated `_build/` content is not committed.
