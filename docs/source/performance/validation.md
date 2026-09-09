# Compiler and backend validation archive

These records describe specific regression investigations and the checks
performed at those checkpoints. They are historical evidence, not a claim
that every backend or full application has been retested at the current
revision. Keep each report's source state, device, limitations and local
artifact references when interpreting its results.

For current Tile status and route comparisons, see the [Tile report](tile/index.md).
The original report paths are retained so existing links remain valid.

## Execution storage and coroutine runtime

```{toctree}
:maxdepth: 1

../../validation/2026-09-06/auxiliary-pipeline/README
../../validation/2026-09-07/auxiliary-admission/README
../../validation/2026-09-07/local-and-coro-lifetimes/README
../../validation/2026-09-07/frame-index-test/README
../../validation/2026-09-07/resume-annotations/README
../../validation/2026-09-07/fallback-coro-arena/README
../../validation/2026-09-07/fallback-abi-temporaries/README
../../validation/2026-09-08/fallback-queue-wakeup/README
```

## Control-flow reconstruction

```{toctree}
:maxdepth: 1

../../validation/2026-09-08/owned-conditional-restructure/README
../../validation/2026-09-08/loop-scope-restructure/README
../../validation/2026-09-09/shared-switch-cases/README
```

## Backend code generation

```{toctree}
:maxdepth: 1

../../validation/hip_late_inline_ssa
../../validation/2026-09-09/metal-world-shader-codegen/README
```
