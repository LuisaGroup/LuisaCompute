# Separate 8192-cubed MPS diagnostic capture

The user authorized MPS profiling/capture to investigate large-matrix gaps.
The capture ran after the first CPU replay finished, outside all acceptance
timings. Command:

```sh
uv run --offline --no-project --python 3.13 --with numpy --with torch python \
  scripts/benchmark/tile_torch/profile_torch.py --backend metal \
  --shape 8192,8192,8192 --seconds 1 --batch 1 --threads 8 \
  --capture-dir /tmp/luisa-mps-scale-20260906-8192
```

The helper reported PyTorch `2.14.0` at
`08187d9e0fba026dc8217405802ab5381dc88d90`, default MPS path, no signposts,
seven post-capture workload iterations and 1146.601208 ms profiled wall time.
That interval is **not an uninstrumented benchmark or an isolated kernel
time**. Eight warmup GEMMs and full-output FP64 checks surround the capture;
the final maximum absolute error was zero (`atol=rtol=1e-4`, unchanged dyadic
input family). This does not establish arbitrary-input accuracy.

Local artifact:
`/private/tmp/luisa-mps-scale-20260906-8192/0000-gemm.gputrace`.
Its plist records one captured frame and UUID
`B6EB8FC3-3837-4D70-9C60-384707466DB7`. The directory includes a 1 GiB MTLHeap
snapshot, retained locally rather than committed. Small identity files:

| File | SHA-256 |
|---|---|
| capture | fe75b21c34c60fb746e8e49b08a14b01e0956ae0f2398b2d59693584cbcb0a73 |
| metadata | 0bce38a01494fda6e3b7144cee8211759fb5ae492116d8af05063a4231a5637c |
| index | a365bb40a82f57010131d45a6e703cc6eeb63539c07ba71bd4c44ed16461d5d8 |

Xcode accepted the open request, but both subsequent System Events and Xcode
window queries timed out with AppleEvent `-1712`. No application was killed,
no settings were changed and no timed GPU replay/counter result was recovered.
Thus shader identity, launch geometry, working-set reuse and stall/occupancy
attribution for this large shape remain **unresolved**. The September 3
1024-cubed capture must not be relabeled as evidence for 8192-cubed behavior.

The next Metal diagnostic should compare this actual MPS launch with the
whole-group MPP atom and the independent-subgroup cohort. Existing native
code consumes the complete K dimension in one dynamic-length MPP operation;
TIRx's bounded large input views have separate K-tail admission limits. These
are source-level facts and candidate explanations, not recovered MPS internals.
