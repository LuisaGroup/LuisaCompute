# iOS Metal4 Device Tests

This directory owns reusable device-conformance sources and the opt-in signed
test application. The interactive example in `examples/ios` mirrors the same
path-tracing/conformance body, while this directory keeps test-only ownership
and target naming.

Enable `LUISA_COMPUTE_BUILD_IOS_TESTS=ON` to create:

~~~text
luisa-metal4-ios-device-air-path-tracer
luisa-ios-device-tests
~~~

The first target is the signed application; the second is its aggregate build
target. It is intentionally separate from ordinary `LUISA_COMPUTE_BUILD_TESTS`
and CTest because a physical iPhone launch, visible frame, retrieved JSON/PNG,
and device console are part of the result. Cross-compiling or installing the
bundle alone is not a pass.

For the supported configure/build flags, use:

~~~sh
scripts/build_ios_llvm.sh \
  --host-llvm-prefix "$(brew --prefix llvm@22)"
scripts/build_ios_metal4.sh \
  --llvm-dir cmake-build-llvm22-ios/lib/cmake/llvm \
  --team <apple-development-team> --mode tests
~~~

Use `--unsigned` for CI-only link closure. The GitHub Actions job builds both
opt-in groups and runs `scripts/audit_ios_bundles.sh`; signed launch and
artifact retrieval remain physical-device requirements.

On 2026-08-28 the shared runtime-linked workload passed on an iPhone 17 Pro Max
with an Apple A19 Pro GPU (Apple10). It used native MTL4 address-driven AS
builds, exercised matrix and component/SRT motion, both D24S8 and D32S8A24
stencil paths, and retrieved nondegenerate conformance and repository
path-tracing PNG/JSON artifacts. The corresponding host preflight passed on an
M1 Max through the guarded pre-Apple9 AS-build bridge.

On a macOS host, the same option builds
`luisa-metal4-ios-path-tracer-aot`, a small AIR/container oracle. That oracle is
useful for format and App Store validation work, but it does not replace the
runtime-linked phone test.

See `examples/ios/metal4_path_tracing/README.md` for configuration, signing,
installation, evidence fields, and the Metal4/Apple-family feature guards.
