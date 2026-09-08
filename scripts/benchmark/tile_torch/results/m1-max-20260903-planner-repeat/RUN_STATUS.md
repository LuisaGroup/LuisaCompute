# Aborted baseline-dependency run

This run was interrupted because the frozen pre-planner executable/library
bundle omitted `libglslang.16.dylib`. Its reference invocations failed in the
dynamic loader before reaching capture, compilation, or kernel execution.

The raw partial records are retained for audit. Do not use this directory for
performance conclusions. The unchanged third-party library was added to the
frozen bundle and the complete run was restarted in
`m1-max-20260903-planner-repeat-verified`.
