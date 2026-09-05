# Tile compiler and Runtime

Tile adds a staged frontend and mutable TileIR to LuisaCompute. It reuses the
existing backend factory, shader handle, buffer and Stream model. The public
language is described in the [Tile programming guide](../../tile/index.md);
this section owns how that program is represented, planned and lowered.

```{toctree}
:maxdepth: 1

decisions
planner
runtime
reductions
xir
```

## Compilation routes

```text
staged C++ Tile kernel
        |
     TileIR -- analyses, verification and transforms
        |
   backend factory
        +-- TIRx bridge ------ CPU / Metal realization
        +-- native Metal ---- bounded MPP realization
        +-- XIR bridge ------ SIMD CPU backend
        |
   Runtime shader handle -> Stream launch
```

The bridges live in `tile/bridge/{tirx,xir}`. Native MPP lowering belongs to
the Metal backend. TIRx export uses TVMx's C++ API, not generated Python;
there is no MLIR dependency. One structural export does not imply identical
CPU and GPU thread bindings, instruction atoms or memory plans.

## Representation, legality and optimization

Execution structure, value distribution and memory addressing compose through
typed layouts without collapsing into the same concept. TileIR is mutable SSA
with regions and managed intrusive ownership/use lists, not a serialization
format. It is intentionally thinner than a general Machine TileIR.

The planner first checks mapping and resource legality, then ranks the legal
candidates. A backend-overridable cost policy separates hardware information
and scoring from the solver. Bounded enumeration and Pareto dynamic programming
are implemented; general scheduling, calibrated spill/traffic costs and a
cross-backend Machine TileIR are not complete.

[Implementation coverage](../../performance/tile/implementation.md) records the
current boundary of each route. [Performance reports](../../performance/tile/index.md)
separately preserve benchmark baselines, regressions and timing limitations.
