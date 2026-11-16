# Examples have moved

The examples that used to live in this directory are maintained in the
**[LuisaComputeGallery](https://github.com/LuisaGroup/LuisaComputeGallery)** repository:

* sources: `src/<group>/` (rendering, simulation, gui, compute, coro, ml, extension,
  interop, tensor, ntc, ocr, embedding, embedding_mm, dx, ios)
* build: both xmake (`xmake build <target>`) and CMake
  (`cmake -S . -B build -G Ninja && cmake --build build --target <target>`) are
  configured for every target; `src/<group>/xmake.lua` and
  `src/<group>/CMakeLists.txt` are generated from `scripts/gallery_targets.json`.
* every example has a headless mode:

  ```
  <example> <backend> --headless [--frames N] [--output-dir DIR] [example options]
  ```

  and `python scripts/headless_smoke.py` runs the whole set without a desktop.

The unit/integration tests that exercised these examples moved as well, to
`tests/` in the same repository. The LuisaCompute tree keeps the library, the
tutorials and its own test suite (`src/tests`).
