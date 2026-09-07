# LuisaCompute C++ Remote Backend Design

## Status and scope

This document describes protocol version 1.1 of the experimental `remote` backend.
The implementation is entirely C++ and deliberately does not reuse the legacy
Rust remote code, `reproc`, `CallableLibrary`, native object images, or
process-local pointers. The client is a normal LuisaCompute backend plugin; the server is a
long-lived service that creates a normal native LuisaCompute device for each
authenticated session and executes work on it.

Version 1 is intended to establish a safe, testable semantic baseline:

- resource handles have connection-local, strongly tagged remote identities;
- command lists retain LuisaCompute stream ordering and asynchronous completion;
- kernels and transitive callables cross the network as a versioned AST document;
- every untrusted frame and AST document is decoded with explicit bounds;
- no C++ object layout, enum width, pointer, or `size_t` representation is put on
  the wire;
- the server never accepts a client-provided native backend handle.

The current wire format is specified only for little-endian 64-bit peers. The
handshake rejects incompatible protocol versions and machine representations.

## Trust and deployment boundary

The protocol parser treats network bytes as structurally untrusted. Framing,
message kinds, resource kinds, sizes, offsets, counts, AST node kinds, types,
bindings, and command regions are checked before native calls. Session resource
tables and configurable quotas prevent one connection from addressing another
connection's objects or growing without a bound. Protocol version 1 still treats
an authenticated client as trusted to originate semantically valid LuisaCompute
builtin operations; backend-specific semantic validation remains the selected
native backend's responsibility. The service is not a hostile multi-tenant
sandbox.

The transport is currently plain TCP. The shared token authenticates a session
but does not encrypt traffic, provide server identity, or prevent replay by an
observer. The frame checksum detects accidental corruption; it is not a MAC.
Version 1 should therefore be bound to loopback or used inside a trusted private
network, VPN, or SSH tunnel. Exposing the listener directly to an untrusted
network requires a future TLS transport with certificate verification.

## Architecture

```text
Luisa application
      |
      | DeviceInterface API
      v
remote backend plugin
  - local proxy handles
  - AST JSON encoder
  - command encoder
  - request/completion dispatcher
  - optional local presentation bridge
      |
      | framed TCP, persistent connection
      v
remote server session
  - authentication and capability negotiation
  - bounded protocol/AST decoders
  - typed per-session resource tables
  - command/binding remapper
      |
      | normal DeviceInterface calls
      v
server-selected native backend and device
```

The server configures a default non-remote backend and an allowlist of
client-selectable backends. A session advertises the protocol feature families
implemented by the service. Native resource or shader creation can still reject
a feature that the selected backend does not support; protocol version 1 does
not yet expose exhaustive per-device capability reflection.

## Build and operation

Enable the backend with:

```text
-DLUISA_COMPUTE_ENABLE_REMOTE=ON
```

The build produces the `remote` backend module and `luisa-remote-server`.
Standalone Asio 1.38.2 is fetched by an archive URL with a pinned SHA-256 hash.
It is used header-only with `ASIO_STANDALONE` and no Boost dependency.

Example server invocation:

```text
luisa-remote-server --backend metal --allow-backend metal \
    --listen 127.0.0.1 --port 18080 \
    --token <shared-token>
```

A native cross-process smoke test is deliberately two independent commands:

```text
luisa-remote-server --backend metal --listen 127.0.0.1 --port 18080 \
    --token test-token
test_remote_backend_native metal 127.0.0.1 18080 test-token
```

The token can instead be supplied through `LUISA_REMOTE_TOKEN`. Client endpoint,
token, request timeouts, in-flight-byte limits, and the upload-cache threshold are
configured with `RemoteDeviceConfigExt`. The same extension can explicitly select
the client-side presentation backend; otherwise the client chooses the native
platform backend (`metal` on macOS, `dx` on Windows, and `vk` on Linux) when a
swapchain is first requested. It can also request a server backend, device index,
and validation mode. The server exposes
`--blob-cache-bytes`, `--blob-cache-entry-bytes`, and
`--blob-cache-min-bytes`; a zero cache capacity disables the feature. Repeated
`--allow-backend` arguments form the exact backend allowlist,
`--allow-client-validation` controls whether a client may enable validation, and
`--max-sessions` bounds concurrent clients. Process
supervisors may pass `--print-ready` to receive a flushed, machine-readable
`LCRP_READY_V1 <bound-port>` line after the service listener and policy are ready.
This also permits race-free discovery when `--port 0` requests an ephemeral port.
The service creates each requested full native compute device rather than asking
for backend "headless" mode: protocol version 1 sends AST and therefore requires
runtime JIT, which headless mode disables on some native backends.
`SIGINT` and `SIGTERM` are consumed by an Asio signal loop and translated into
`Server::stop()`, allowing the accept loop and active sessions to shut down and
join before the process exits.

## Service lifecycle and device selection

The protocol 1.1 `HELLO` extension carries a backend name, device index, and
validation request. Empty backend and sentinel device index select the server
defaults. Protocol 1.0 clients omit these fields and continue to select the
defaults. Backend names are syntax-checked before they reach a factory and the
standalone service additionally requires an exact match in its configured
allowlist. Device indices are checked against the native backend's advertised
device list. Client-requested validation is rejected unless the service operator
explicitly enables it.

Every accepted session owns a separately constructed `DeviceInterface` and its
typed resource tables. An EOF, reset, malformed handshake, rejected device
request, or client process abort stops only that session: its streams are drained,
callbacks are detached, resources are destroyed in dependency order, and the
native device is released. The accept loop remains live, reaps completed workers,
and admits a fresh reconnect. The session limit rejects excess connections before
allocating a device. A service shutdown closes the listener and all current
sessions, then joins their workers.

Reconnect means creating a fresh client `Device` and session. Transparent resume
of the same client object is intentionally not attempted: connection-local
resource IDs cannot remain valid after their native resources have been reclaimed.
`remote.connected` is a client-local query that lets an application detect loss
and rebuild that device. Resource destruction after a lost connection is
best-effort and never sends stale handles into a new session. The lower-level TCP
`Connection` object itself may be reconnected after its reader observes EOF, but
the new connection has no relationship to the previous resource namespace.

## Local client presentation

Windows and native display handles are process-local and never cross the wire.
`RemoteDevice::create_swapchain` instead creates a graphics stream, swapchain,
and mirror image on a lazily created local presentation device. The application
continues to use the ordinary `Window -> Device::create_swapchain -> present`
API; only kernels and application resources belong to the remote server.

At each present boundary, the client drains the remote graphics stream, downloads
the base mip of the presented remote image into bounded client staging storage,
uploads it to the local mirror image, and presents that image on the local native
swapchain. A subsequent present or stream synchronization waits for the local
submission before reusing the staging bytes. The baseline path is intentionally
synchronous across the network; later protocol versions can add compressed or
multi-buffered frame transport without exposing native handles or weakening
stream ordering.

The presented image must match the swapchain extent and backend storage. Raster
shaders remain outside protocol version 1, but compute-, ray-tracing-, and
simulation-rendered images can be displayed locally. The selected backend is
reported by the `remote.local_present_backend` device query.

## Framing and request correlation

Every message begins with a fixed 40-byte little-endian header. Fields are
encoded individually:

| Field | Width | Meaning |
| --- | ---: | --- |
| magic | 4 | `LCRP` protocol discriminator |
| major/minor | 2 + 2 | negotiated protocol version |
| kind | 2 | closed message-kind enum |
| flags | 2 | versioned flags, zero unless specified |
| reserved | 4 | must be zero in version 1 |
| request id | 8 | nonzero request/response correlation id |
| payload size | 8 | checked before allocation or read |
| payload checksum | 8 | corruption detection for the payload |

The client keeps one persistent connection, allocates monotonically increasing
request IDs, and has a reader loop that resolves pending requests independently
of submission order. Responses carry status and an error message. Unsolicited
notifications are reserved for dispatch completion and stream logging.

Decoders reject unknown enums, malformed booleans, invalid UTF-8-independent
length prefixes, integer overflow, trailing bytes, oversized frames, strings, or
arrays. Transport close atomically fails all pending requests. Request and
connect timeouts cancel the corresponding socket operation.

Servers advertise `Feature::LIMIT_NEGOTIATION`. After `HELLO`, the client reads
the server's frame, string, and array ceilings through `PROTOCOL_INFO` and uses
the intersection with its own limits for later bounded encoding and decoding. The
negotiated values are exposed as `remote.protocol.max_frame_payload`,
`remote.protocol.max_string_size`, and `remote.protocol.max_array_size`.
`Feature::DEVICE_SELECTION` and the `remote.device_selection` query report that
the session was created by a service device factory.

## Remote handles and lifetime

A remote resource ID consists of an 8-bit resource-kind tag and a 56-bit session
index. Buffer, texture, bindless array, stream, shader, event, mesh, curve,
procedural primitive, motion instance, and acceleration structure IDs occupy
separate namespaces. Every lookup verifies the tag as well as existence.

IDs are meaningful only in the session that created them and are never reused
after index exhaustion. Destroy requests remove the server-side entry. Session
shutdown releases all remaining entries and closes its streams before the native
device is released. A client cannot inject a process-local handle into a shader
or command: all such fields pass through the session resolver.

## Shader transport: versioned AST JSON

`ast2json` now defines schema version 1 and has a matching `from_json` decoder.
It serializes the complete reachable function graph, including custom callables,
types, constants, expressions, statements, arguments, resource bindings, block
size, required curve bases, and variable usage. It does not serialize executable
code, C++ object representations, function pointers, or `CallableLibrary` data.

The server shader path is:

```text
client Function -> ast2json schema v1 -> bounded JSON bytes
                -> yyjson parse and structural preflight
                -> FunctionBuilder reconstruction with remote binding resolver
                -> native DeviceInterface::create_shader
```

The decoder uses yyjson with a quota allocator and enforces document, parse
memory, function, type, node, recursion-depth, string, array, and deduplicated
constant-byte limits. It requires exact object keys where ambiguity would be
dangerous, validates Base64 strictly, rejects duplicate identifiers and unknown
node kinds, checks expression/statement ownership and type compatibility, and
rejects unsafe CPU/custom/external operations. Builtin argument counts and the
reconstruction invariants that could otherwise trigger `FunctionBuilder` or type
registry assertions are checked through non-terminating decoder errors first.
The only custom type identifiers accepted at this boundary are the framework's
`LC_IndirectDispatchBuffer`, `LC_RayQueryAll`, and `LC_RayQueryAny`; arbitrary
client-defined custom types remain rejected. Scalar assignment conversions match
the DSL's implicit scalar rules, while composite assignments require identical
types.

Serialized buffer, texture, bindless-array, and acceleration-structure bindings
are remapped through `ASTJsonBindingResolver`. The server resolver verifies the
remote resource kind, session ownership, buffer slice, and texture mip before it
returns a native binding. Native shader include blobs and AOT shader loading are
not part of protocol version 1.

The original `to_json(Type)` and `to_json(Function)` entry points remain for
source compatibility. New network-facing code uses `try_to_json` and checks its
structured error result. `from_json` is the only supported reverse path.

## Command lists and data movement

Command payloads contain a command tag followed by explicitly encoded fields.
Version 1 supports:

- buffer upload, download, and buffer-to-buffer copy;
- texture upload, download, texture-to-texture copy, buffer-to-texture copy, and
  texture-to-buffer copy;
- content-addressed buffer and texture uploads negotiated through the blob cache;
- direct, multiple-dispatch, and indirect shader dispatch;
- bindless-array updates;
- mesh, curve, procedural primitive, motion instance, and top-level acceleration
  structure build commands;
- stream signal/wait, event query/synchronization, resource naming, and stream
  logging.

The server validates all buffer ranges with overflow-safe arithmetic. Texture
commands are checked against the recorded storage format, base extent, mip
count, selected mip extent, copy region, and required byte footprint. Uniform
arguments are padded to the alignment expected by `CommandEncoder`. Transforms
and other floating-point structural fields that must be finite are checked before
native dispatch.

Downloads use server-owned staging memory whose lifetime extends through native
completion. Data is returned in the completion notification, not in the initial
dispatch acknowledgement. The client copies it into the application destination
before invoking the command-list completion callbacks.

## Stream ordering and asynchronous completion

Submitting a command list performs only bounded encoding, in-flight admission,
and a dispatch request. The acknowledgement means that the server accepted the
submission; it does not mean the native device finished it. The server appends a
native command-list callback and sends `DISPATCH_COMPLETE` only from that callback.

The client associates completion state, readback destinations, callbacks, and
the encoded byte footprint with a monotonically increasing submission ID. Its
reader/completion path applies readbacks in command order, invokes callbacks,
and releases in-flight capacity. `synchronize_stream` first synchronizes the
server stream and then drains all client-side completion work for that stream.
Event operations map directly to native timeline events.

An in-flight byte budget supplies client-side backpressure. The budget covers
encoded command bytes plus retained readback state and prevents an unbounded
queue when the network or native device is slower than the producer. Missing
blob bodies are transferred synchronously before dispatch admission completes;
the server pins their immutable cache storage through native completion.

## Failure model

- A malformed request receives a typed error when a response can still be sent.
- Authentication or version failure closes the session after the handshake.
- A dispatch decode or admission failure is reported against its request ID;
  the current `DeviceInterface::dispatch` API has no native asynchronous error
  return channel after admission.
- Socket failure wakes pending synchronous requests and stream-drain waiters.
- Client disconnect, abort, or rejected device selection tears down only that
  session; the accept loop remains available for a fresh connection.
- A completion callback is never reported before its readback data is installed.
- Resource limits fail creation without allocating a native resource.
- Server stop closes the acceptor and active sessions so their worker threads can
  join without relying on another incoming connection.

## Version 1 boundaries

The following are intentionally not represented yet:

- general GUI/native interop and imported process-local resources beyond the
  isolated local swapchain presentation bridge;
- raster shaders and raster command lists;
- sparse resources, DirectStorage, and native extension handles;
- AOT shader names or backend-native shader binaries;
- transport encryption, certificate-based identity, or transparent session resume;
- heterogeneous endian/word-size peers;
- automatic load balancing or migration between devices after session creation.

Unsupported methods fail explicitly or return an invalid creation result. They
must not silently fall back to a local device because that would break resource
ownership and ordering.

## Content-addressed upload cache

`Feature::BLOB_CACHE` adds a compatible cached path while preserving the inline
version-1 commands for older clients and small uploads. For each submission the
client deduplicates eligible buffer and texture upload bodies, computes a SHA-256
digest, and sends ordered `(size, digest)` descriptors with `PREPARE_BLOBS`. The
server returns the missing descriptor indices. One or more bounded
`UPLOAD_BLOBS` requests then carry only those bodies, repeating the index, size,
and digest so each record is self-validating. Buffer and texture commands refer
to the prepared slot instead of embedding the bytes.

Preparation is a submission-scoped lease, not merely a speculative cache query.
It pins hits across the prepare/upload/dispatch interval, eliminating an eviction
race between a reported hit and command decoding. The decoded command list owns
immutable shared blob references through its native completion callback. Every
prepared descriptor must be filled and referenced by the matching submission;
duplicate descriptors, missing bodies, wrong sizes, digest mismatches, stale
submission IDs, and unused slots are rejected.

The server cache is shared across its authenticated sessions and uses a
byte-budgeted LRU. Only entries without active leases or submissions may be
evicted; inability to make space produces `RESOURCE_LIMIT` rather than silently
falling back to the wrong bytes. SHA-256 is computed before an uploaded body is
published, and an existing key is byte-compared when concurrent publication
races occur. Small or one-shot uploads stay inline because an extra negotiation
round trip would cost more than the saved bandwidth.

All sessions accepted by one server currently share the same configured token
and cache trust domain. A future multi-tenant authentication scheme must
partition cache keys and accounting by authenticated identity; otherwise a
client that knows another tenant's digest could reuse its bytes. TLS remains
required on an untrusted network regardless of the digest algorithm.

Operational counters use the `remote.blob_cache.` prefix with the properties
`hits`, `misses`, `stores`, `evictions`, `uploaded_bytes`, `resident_bytes`, and
`resident_entries`, plus `capacity_bytes` and `enabled`.

## Validation strategy

The implementation has four focused layers of tests:

- AST serde tests cover kernels, custom callables, control flow, binding remap,
  malformed documents, duplicate keys, invalid Base64, builtin arity,
  reconstruction-safety guards, and representative decoder quotas;
- protocol and transport tests cover framing, checksums, protocol 1.0/1.1 minor
  compatibility, limits, request correlation, timeout/cancellation, connection
  failure, same-object reconnect, and asynchronous notifications;
- blob-cache tests cover SHA-256 vectors, digest rejection, LRU pinning and
  eviction, duplicate-content planning, cached command encoding, hit reuse, and
  inline fallback when the server does not advertise the feature;
- an end-to-end test dynamically loads the `remote` backend against a localhost
  server and a mock native device, then exercises buffers, every texture-copy
  direction, AST shader creation with callable/bound-resource remapping, bindless
  updates, timeline events, stream callbacks/logs, and all ray-tracing resource
  build/remap paths. It also force-closes a client without `GOODBYE`, verifies
  resource reclamation and reconnect, rejects out-of-range shader metadata
  queries without losing the session, permits client resources to destruct after
  service loss, isolates a throwing device factory, rejects one device request
  without stopping the service, and creates two simultaneous sessions using
  distinct requested backend names and device indices;
- a two-command cross-process native test runs `luisa-remote-server` and the
  native remote client test as independent executables. The client drives a real
  Metal device through TCP and checks AST JIT and callable reconstruction, bound
  resources, GPU buffer/texture execution and readback, asynchronous completion,
  timeline events, and cold/hot upload-cache accounting;
- a separate local-presentation test executes a remote image kernel, downloads
  each present boundary, and displays three frames through a client-owned Metal
  swapchain; and
- the GUI-enabled rendering sweep runs 17 finite offline examples through a
  persistent Metal service, including path tracing variants, AST/XIR-to-AST,
  ray masks, ray-query cutout, spectrum, photon mapping, SDF, procedural,
  blackhole, voxel, and shader-toy workloads. Gallery-backed examples are also
  checked at their reference sample counts using PSNR and structural metrics.

The mock test verifies the boundary and ordering without requiring a particular
GPU. Other native backends and performance tests remain deployment-specific and
should be run in addition before enabling the backend in production.
