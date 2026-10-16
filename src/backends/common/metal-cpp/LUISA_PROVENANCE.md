# metal-cpp provenance

This directory contains Apple's unmodified `metal-cpp` release headers.

- Upstream: <https://github.com/apple/metal-cpp>
- Release tag: `release/metal-cpp_macOS27_iOS27`
- Archive: <https://github.com/apple/metal-cpp/archive/refs/tags/release/metal-cpp_macOS27_iOS27.tar.gz>
- Archive SHA-256: `12c5dc033b49e8541216605cd08f4706ffc94b659cb1e1a9ea98cca9c784e037`
- Header version: `381.0.0`

The headers are kept under `src/backends/common` because both the legacy Metal
backend and the independent Metal 4 backend consume the same Objective-C bridge.
