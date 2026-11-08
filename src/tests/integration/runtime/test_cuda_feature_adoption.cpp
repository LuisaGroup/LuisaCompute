// Tests for the CUDA-13.x feature adoption in the CUDA backend.
//
// Feature under test:
//   B1  CudaGraphExt::capabilities().programmatic_edges is a *device* capability.
//       Programmatic (PDL) edges are a sm_90+ feature, so the flag can never be
//       reported on a device on which the other Hopper-gated flags
//       (`device_launch`, `device_updatable_nodes`) are false, even when the
//       toolkit/driver are new enough for the typed-edge API itself. The absolute
//       form (all three false) holds only on pre-Hopper hardware and is not
//       assertable from a test; see TODO(hardware) in test_capability_gate.
//   B2  The stream-form PDL launch attribute is keyed on the kernel's AST
//       containing `grid_dependency_synchronize()`; that precondition is asserted
//       here (plain kernel -> false, waiting kernel -> true, user callable -> true)
//       because the launch path itself cannot re-derive it (a shader-cache hit
//       never runs code generation, so CUDAShaderMetadata records it once).
//   B3  The host-callback ticket protocol of CUDAStream: exactly one callback run
//       per submission, callbacks of one submission run in enqueue order, a
//       callback observes the work enqueued before it, and work enqueued after a
//       host function does not overtake it.
//
// TODO(hardware): the stream-form PDL launch itself (`cuLaunchKernelEx` +
// CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, the per-shader latch and
// the attribute-refusal fallback) cannot be exercised on this platform: the device
// is pre-Hopper, so the gate is inert by construction, and a kernel containing
// `griddepcontrol.*` device code cannot even be loaded for sm_89 (it fails PTX JIT
// with CUDA_ERROR_INVALID_PTX, which is the reason the gate exists). Cover it on an
// sm_90+ machine by running a two-kernel producer/consumer chain on one stream with
// the wait in the consumer and comparing against the sequential result.
// TODO(hardware): the positive direction of B1 (all three Hopper-gated flags true on
// sm_90+ with runtime >= 12.4) is likewise unobservable here; only the version-free
// implications are assertable on a pre-Hopper device.
// TODO(hardware): the PDL *graph* edge execution (a device-launch graph whose
// consumer node waits on a programmatic edge) is skipped by test_cuda_graph_pdl on
// this device for the same reason; it needs sm_90+ as well.
// TODO(driver): the spin-wait mode of the host callbacks (cuLaunchHostFunc_v2 with
// CU_HOST_TASK_SPINWAIT, resolved through cuGetProcAddress) is a driver scheduling
// choice. The backend logs the selected mode once ("CUDA host callbacks: ..."), and
// the ticket protocol below is identical for both modes, so nothing observable
// distinguishes spin from blocking here. A negative control for the version gate
// needs a pre-13.2 driver.
// TODO(backend): the backend builds an explicit CUDA graph (no stream capture), so
// the "no launch attribute while capturing" case cannot occur; if a capture path is
// ever added, the attribute must be suppressed there.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <thread>

#include <luisa/core/stl/filesystem.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace test_cuda_feature_adoption_detail {

using FuncBuilder = luisa::compute::detail::FunctionBuilder;

/// The exact predicate `CUDADevice::create_shader` stores in the shader metadata
/// and that the launch path then consults (B2).
[[nodiscard]] bool waits_for_programmatic_dependency(
    const luisa::shared_ptr<const FuncBuilder> &builder) noexcept {
    return Function{builder.get()}.propagated_builtin_callables().test(
        CallOp::GRID_DEPENDENCY_SYNCHRONIZE);
}

[[nodiscard]] std::string read_text_file(const luisa::filesystem::path &path) {
    std::ifstream file{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{file},
            std::istreambuf_iterator<char>{}};
}

void write_text_file(const luisa::filesystem::path &path,
                     const std::string &text) {
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
}

/// The sidecar path is derived from the shader name, so a missing file must
/// report what the backend actually wrote instead of only "not found".
[[nodiscard]] std::string describe_directory(const luisa::filesystem::path &directory) {
    std::error_code error;
    std::string listing;
    for (auto &&entry : luisa::filesystem::directory_iterator{directory, error}) {
        listing.append(entry.path().filename().string()).append(" ");
    }
    if (error) {
        return std::string{"<cannot list "} + directory.string() + ": " +
               error.message() + ">";
    }
    return listing.empty() ? std::string{"<empty>"} : listing;
}

/// Evidence that a compile-only shader was really produced: `Device::compile_to`
/// returns `ShaderCreationInfo::compile_ok()`, which is *always* true for a
/// compile-only CUDA shader - the backend returns
/// `ShaderCreationInfo::make_invalid()` for `compile_only` (whose `compile_ok` keeps
/// its default `true`) and a failed NVRTC compile aborts inside the compiler instead
/// of returning false - so the written artefacts are the only assertions that can
/// fail when a compile produced nothing.
struct Artefact {
    std::string text;
    luisa::filesystem::file_time_type written{};
    bool has_timestamp{false};
};

[[nodiscard]] Artefact read_artefact(const luisa::filesystem::path &path) {
    std::error_code error;
    Artefact artefact;
    artefact.written = luisa::filesystem::last_write_time(path, error);
    artefact.has_timestamp = !error;
    artefact.text = read_text_file(path);
    return artefact;
}

/// True when the file was written again: either the bytes changed or the timestamp
/// advanced (a deterministic recompile can produce identical bytes, while a cached
/// entry is not rewritten at all).
[[nodiscard]] bool was_rewritten(const Artefact &before, const Artefact &after) noexcept {
    // A file that disappeared is a change as well: two failed reads both look like
    // "no text and no timestamp", so the comparisons below would call a deleted
    // artefact unchanged.
    if (before.has_timestamp != after.has_timestamp) { return true; }
    return before.text != after.text ||
           (before.has_timestamp && after.has_timestamp &&
            before.written != after.written);
}

/// `compile_to` cannot report a CUDA compile failure (see `Artefact`), so its return
/// value is only quoted by the failure messages of the artefact assertions.
template<typename Kernel>
[[nodiscard]] bool compile_to_file(Device &device, const Kernel &kernel,
                                   const luisa::filesystem::path &ptx) noexcept {
    return device.compile_to(kernel, ptx.string());
}

[[nodiscard]] const char *describe_compile_result(bool compile_ok) noexcept {
    return compile_ok ? "success" : "failure";
}

/// Separate two writes so that a coarse filesystem clock cannot report the same
/// `last_write_time` for both; the byte comparison in `was_rewritten` alone would
/// read a deterministic recompile as "not rewritten".
void pause_before_rewrite() noexcept {
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
}

/// Precondition of the cache-hit control: case (c) must have left a readable `.ptx`
/// and a sidecar that records the wait.
[[nodiscard]] bool hit_precondition_ok(const luisa::filesystem::path &ptx,
                                       const luisa::filesystem::path &sidecar) {
    auto artefact = read_artefact(ptx);
    return artefact.has_timestamp && !artefact.text.empty() &&
           read_text_file(sidecar).find("GRID_DEPENDENCY_SYNC TRUE") != std::string::npos;
}

struct ScopedDirectoryCleanup {
    luisa::filesystem::path path;
    ~ScopedDirectoryCleanup() noexcept {
        std::error_code error;
        luisa::filesystem::remove_all(path, error);
    }
};

/// B1: the programmatic-edge capability must be a subset of the Hopper-gated
/// capabilities. `programmatic_edges == true` on a device that reports neither
/// `device_launch` nor `device_updatable_nodes` is the bug this guards: the flag then
/// promises an edge annotation the kernels cannot even be compiled for
/// (`griddepcontrol.*` needs sm_90+). The absolute form - all three false - only
/// holds on pre-Hopper hardware, so it is logged rather than asserted here.
void test_capability_gate(Device &device) noexcept {
    auto *ext = device.extension<CudaGraphExt>();
    if (ext == nullptr) {
        LUISA_INFO("CudaGraphExt is not available on this backend; skipping the capability test.");
        return;
    }
    auto caps = ext->capabilities();
    LUISA_INFO("CUDA graph capabilities: cuda_version = {}, programmatic_edges = {}, "
               "device_launch = {}, device_updatable_nodes = {}, conditional_nodes = {}, "
               "graph_memory = {}, node_priority = {}, switch_conditional = {}.",
               caps.cuda_version, caps.programmatic_edges,
               caps.device_launch, caps.device_updatable_nodes,
               caps.conditional_nodes, caps.graph_memory,
               caps.node_priority, caps.switch_conditional);

    // `caps.cuda_version` is the *compile-time* CUDA_VERSION (cuda_graph_ext.cpp),
    // while every flag below is gated on the *effective* runtime version
    // min(toolkit, driver). The compile-time value is therefore an upper bound for
    // the runtime version, which makes the "flag implies at least this version"
    // direction sound on every toolkit/driver pair and the opposite direction
    // vacuous. Asserting the seven implications individually (instead of counting
    // them) is what pins each flag to its own documented floor. The enforced build
    // floor is CUDA 12.0 (scripts/validate_options.cmake: find_package(CUDAToolkit
    // 12)), not 12.1.
    expect(caps.cuda_version >= 12000u)
        << "the CUDA backend only builds against CUDA >= 12.0";
    expect(!caps.programmatic_edges || caps.cuda_version >= 12040u)
        << "programmatic (PDL) edges need CUDA >= 12.4";
    expect(!caps.device_launch || caps.cuda_version >= 12000u)
        << "device graph launch needs CUDA >= 12.0";
    expect(!caps.device_updatable_nodes || caps.cuda_version >= 12040u)
        << "device-updatable kernel nodes need CUDA >= 12.4";
    expect(!caps.conditional_nodes || caps.cuda_version >= 12030u)
        << "conditional graph nodes need CUDA >= 12.3";
    expect(!caps.graph_memory || caps.cuda_version >= 11040u)
        << "graph device memory nodes need CUDA >= 11.4";
    expect(!caps.node_priority || caps.cuda_version >= 12080u)
        << "node priorities need CUDA >= 12.8";
    expect(!caps.switch_conditional || caps.cuda_version >= 12080u)
        << "conditional switch nodes need CUDA >= 12.8";
    // The PDL flag is a subset of the two other Hopper-gated capabilities: a
    // programmatic edge annotation on a device that can neither launch nor update
    // nodes programmatically is precisely the bug this file guards.
    expect(!caps.programmatic_edges || caps.device_launch)
        << "programmatic_edges must not be reported when the device cannot do "
           "programmatic dependent launch at all (device_launch is the sm_90+ gate)";
    expect(!caps.programmatic_edges || caps.device_updatable_nodes)
        << "the same Hopper gate must apply to every programmatic capability";
    // The four runtime-gated flags form a monotone chain: a 12.8 flag implies the
    // 12.3 one, which implies the 11.4 one (no device capability is involved).
    expect(!caps.node_priority || caps.conditional_nodes)
        << "node priorities imply conditional nodes (12.8 >= 12.3)";
    expect(!caps.conditional_nodes || caps.graph_memory)
        << "conditional nodes imply graph memory (12.3 >= 11.4)";
    expect(caps.node_priority == caps.switch_conditional)
        << "both 12.8 flags are gated on the same runtime version";

    // The two implications above are the device-independent half of the Hopper gate.
    // The absolute form - "all three are false on this box" - is *not* assertable
    // from a test: the implementation gates the three flags on
    // hopper_or_newer (compute capability >= 90) on top of the runtime version, so on
    // sm_90+ the absolute form is a guaranteed red. The observed values are logged.
    LUISA_INFO("Hopper-gated CUDA graph capabilities: device_launch = {}, "
               "device_updatable_nodes = {}, programmatic_edges = {}.",
               caps.device_launch, caps.device_updatable_nodes,
               caps.programmatic_edges);
    // TODO(hardware): the positive direction (device_launch &&
    // device_updatable_nodes && programmatic_edges all true on sm_90+ with runtime
    // >= 12.4) needs a Hopper/Blackwell box. It cannot be written conditionally
    // either, because CudaGraphExt::Capabilities does not expose the compute
    // capability that decides it (cuda_graph_ext.cpp computes it locally as
    // hopper_or_newer); adding that field is what would make the gate assertable.
    // The four flags that only need the CUDA runtime are *reported*, not required:
    // they are gated on the effective runtime version min(CUDA_VERSION, driver), so a
    // toolkit newer than the installed driver - a supported pairing - legitimately
    // makes them false.
    if (caps.cuda_version >= 12080u) {
        LUISA_INFO("CUDA runtime-gated graph capabilities (toolkit >= 12.8): "
                   "conditional_nodes = {}, graph_memory = {}, node_priority = {}, "
                   "switch_conditional = {}.",
                   caps.conditional_nodes, caps.graph_memory,
                   caps.node_priority, caps.switch_conditional);
    }
    // TODO(driver): a *positive* assertion for those four flags ("must be reported
    // with runtime >= 12.x") is only correct when its guard is the effective runtime
    // version, i.e. min(CUDA_VERSION, cuDriverGetVersion());
    // CudaGraphExt::Capabilities exposes only the compile-time cuda_version today. A
    // pre-12.8 driver satisfies any compile-time guard while all four flags are
    // legitimately false, so they are logged instead of asserted.
}

/// B2 precondition: the AST records the wait for the kernel and for anything that
/// calls a waiting callable.
void test_stream_pdl_precondition() noexcept {
    Kernel1D waiting = [] { grid_dependency_synchronize(); };
    expect(waits_for_programmatic_dependency(waiting.function()))
        << "a kernel that waits for its programmatic dependency must report it";
    Kernel1D plain = [] {};
    expect(!waits_for_programmatic_dependency(plain.function()))
        << "a kernel without the wait must keep the serialized launch";
    Callable<void()> wait = [] { grid_dependency_synchronize(); };
    expect(waits_for_programmatic_dependency(wait.function_builder()))
        << "the callable itself must record the wait";
    Kernel1D caller = [&wait] { wait(); };
    expect(waits_for_programmatic_dependency(caller.function()))
        << "the wait must be propagated into every caller of the callable";
    LUISA_INFO("stream-form PDL precondition (AST bitset) passed.");
}

/// B2 end to end: the AST bit must reach the serialized shader metadata that the
/// launch path consumes, and that metadata must take part in the shader cache
/// identity. Both are one line in CUDADevice::create_shader, so no other test in
/// this file - and no other test in the tree - would notice if either changed.
void test_shader_metadata_records_the_wait(Device &device) noexcept {
    auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    auto directory = luisa::filesystem::temp_directory_path() /
                     luisa::format("luisa_cuda_pdl_sidecar_{}", nonce);
    luisa::filesystem::create_directories(directory);
    ScopedDirectoryCleanup cleanup{directory};

    Kernel1D waiting = [] { grid_dependency_synchronize(); };
    Kernel1D plain = [] {};

    // The sidecar path is "<shader name>.metadata" (cuda_device.cpp appends the
    // extension to the user name), and a compile-only shader writes it without
    // loading the module - which is what makes this observable on a pre-Hopper
    // device at all (`griddepcontrol.wait` cannot be JIT-loaded for sm_89).
    auto sidecar_of = [](const luisa::filesystem::path &ptx) {
        return luisa::filesystem::path{
            luisa::format("{}.metadata", ptx.string())};
    };

    // (a) the flag is recorded for both values. `compile_to` itself cannot fail on
    // CUDA (see compile_to_file), so each compile is checked through the artefact it
    // must have written.
    auto waiting_ptx = directory / "wait.ptx";
    auto waiting_compiled = compile_to_file(device, waiting, waiting_ptx);
    auto waiting_ptx_first = read_artefact(waiting_ptx);
    expect(!waiting_ptx_first.text.empty())
        << "compiling the waiting kernel must write " << waiting_ptx.string()
        << " (compile_to reported " << describe_compile_result(waiting_compiled)
        << "); directory content: " << describe_directory(directory);
    auto waiting_sidecar_path = sidecar_of(waiting_ptx);
    auto waiting_sidecar = read_text_file(waiting_sidecar_path);
    expect(!waiting_sidecar.empty())
        << "the CUDA backend must write the metadata sidecar "
        << waiting_sidecar_path.string() << "; directory content: "
        << describe_directory(directory);
    expect(waiting_sidecar.find("GRID_DEPENDENCY_SYNC TRUE") != std::string::npos)
        << "the metadata sidecar must record the programmatic-dependency wait; "
           "sidecar:\n"
        << waiting_sidecar;

    auto plain_ptx = directory / "plain.ptx";
    auto plain_compiled = compile_to_file(device, plain, plain_ptx);
    auto plain_ptx_first = read_artefact(plain_ptx);
    expect(!plain_ptx_first.text.empty())
        << "compiling the plain kernel must write " << plain_ptx.string()
        << " (compile_to reported " << describe_compile_result(plain_compiled)
        << "); directory content: " << describe_directory(directory);
    auto plain_sidecar = read_text_file(sidecar_of(plain_ptx));
    expect(!plain_sidecar.empty())
        << "the CUDA backend must write the metadata sidecar for the plain kernel; "
           "directory content: "
        << describe_directory(directory);
    expect(plain_sidecar.find("GRID_DEPENDENCY_SYNC FALSE") != std::string::npos)
        << "a kernel without the wait must be recorded as FALSE; sidecar:\n"
        << plain_sidecar;

    // (b) cache identity, case 1: recompiling *the same shader name* with the other
    // property must not silently reuse the cached sidecar.
    auto shared_ptx = directory / "shared.ptx";
    auto shared_sidecar_path = sidecar_of(shared_ptx);
    auto shared_before_first = read_artefact(shared_ptx);
    auto shared_compiled_first = compile_to_file(device, plain, shared_ptx);
    auto shared_after_first = read_artefact(shared_ptx);
    expect(!shared_after_first.text.empty() &&
           was_rewritten(shared_before_first, shared_after_first))
        << "the first compile under the shared name must write "
        << shared_ptx.string() << " (compile_to reported "
        << describe_compile_result(shared_compiled_first)
        << "); directory content: " << describe_directory(directory);
    auto shared_first = read_text_file(shared_sidecar_path);
    expect(shared_first.find("GRID_DEPENDENCY_SYNC FALSE") != std::string::npos)
        << "the first compile under the shared name must record FALSE; sidecar:\n"
        << shared_first;
    pause_before_rewrite();
    auto shared_compiled_second = compile_to_file(device, waiting, shared_ptx);
    auto shared_after_second = read_artefact(shared_ptx);
    expect(was_rewritten(shared_after_first, shared_after_second))
        << "recompiling the same shader name with a waiting kernel must rewrite "
        << shared_ptx.string() << " (nothing was written, so the stale entry was "
                                  "reused); compile_to reported "
        << describe_compile_result(shared_compiled_second);
    auto shared_second = read_text_file(shared_sidecar_path);
    expect(shared_second.find("GRID_DEPENDENCY_SYNC TRUE") != std::string::npos)
        << "recompiling the same shader name with a waiting kernel must recompile, "
           "not reuse the stale sidecar; sidecar:\n"
        << shared_second;

    // (c) cache identity, case 2: a sidecar that disagrees *only* in this flag must
    // be rejected. Every other term (checksum, block size, argument list, ...) is
    // left untouched, so this is the only case that isolates the flag's participation
    // in CUDAShaderMetadata::operator== (which would otherwise be masked by the
    // differing checksum of a genuinely different kernel).
    auto stale = waiting_sidecar;
    constexpr std::string_view wait_token{"GRID_DEPENDENCY_SYNC TRUE"};
    auto token_position = stale.find(wait_token);
    expect(token_position != std::string::npos);
    if (token_position == std::string::npos) { return; }
    stale.replace(token_position, wait_token.size(), "GRID_DEPENDENCY_SYNC FALSE");
    write_text_file(waiting_sidecar_path, stale);
    expect(read_text_file(waiting_sidecar_path) == stale)
        << "the stale sidecar fixture must have been written";
    pause_before_rewrite();
    auto rejected_compiled = compile_to_file(device, waiting, waiting_ptx);
    auto refreshed = read_text_file(waiting_sidecar_path);
    expect(!refreshed.empty() && refreshed != stale)
        << "the waiting kernel must still compile when its cached entry is rejected, "
           "which rewrites "
        << waiting_sidecar_path.string() << " (compile_to reported "
        << describe_compile_result(rejected_compiled) << "); sidecar:\n"
        << refreshed;
    expect(refreshed.find("GRID_DEPENDENCY_SYNC TRUE") != std::string::npos)
        << "a cached metadata entry that disagrees about the wait must be recompiled "
           "instead of reused; sidecar:\n"
        << refreshed;

    // (d) positive control for (c): the sidecar is pristine again (the rejected entry
    // was recompiled above, so it says TRUE and matches this kernel), and the very
    // same kernel is compiled under the very same name once more - which must now be
    // a clean *cache hit*: neither file is written. Without this, (c) would also pass
    // on a backend that recompiles unconditionally, i.e. if the user-path cache were
    // never consulted at all, and then the FALSE -> TRUE rewrite would prove nothing
    // about the flag's participation in the cache identity.
    expect(hit_precondition_ok(waiting_ptx, waiting_sidecar_path))
        << "the (c) recompile must leave a complete cached entry to hit ("
        << waiting_ptx.string() << "); directory content: "
        << describe_directory(directory);
    pause_before_rewrite();
    auto hit_before = read_artefact(waiting_ptx);
    auto hit_sidecar_before = read_artefact(waiting_sidecar_path);
    auto hit_compiled = compile_to_file(device, waiting, waiting_ptx);
    auto hit_after = read_artefact(waiting_ptx);
    auto hit_sidecar_after = read_artefact(waiting_sidecar_path);
    expect(!was_rewritten(hit_before, hit_after))
        << "an unmodified cache entry must be a cache hit: " << waiting_ptx.string()
        << " was rewritten, so the user-path cache was not consulted (compile_to "
           "reported "
        << describe_compile_result(hit_compiled) << ")";
    expect(!was_rewritten(hit_sidecar_before, hit_sidecar_after))
        << "a cache hit must not rewrite the metadata sidecar "
        << waiting_sidecar_path.string();

    // The launch itself is not covered here - it needs sm_90+; see the
    // TODO(hardware) block at the top of this file (cuLaunchKernelEx, the
    // per-shader latch and the attribute-refusal fallback).
    LUISA_INFO("shader metadata: the programmatic-dependency wait is recorded and "
               "part of the cache identity.");
}

/// B3: the ticket protocol behind CUDAStream::callback().
void test_host_callback_protocol(Device &device) noexcept {
    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint32_t>(1u);
    luisa::vector<uint32_t> host(1u, 0u);
    Kernel1D write_value = [](BufferUInt b, UInt v) noexcept { b.write(0u, v); };
    auto shader = device.compile(write_value);

    // One submission per iteration: the callback must run after the copy of that
    // submission (so it observes the freshly written value) and exactly once.
    constexpr auto kSubmissions = 16u;
    std::atomic_uint callback_count{0u};
    for (auto i = 0u; i < kSubmissions; i++) {
        std::atomic_uint observed{0u};
        host[0] = 0u;
        CommandList list;
        list << shader(buffer, i + 1u).dispatch(1u);
        list << buffer.copy_to(luisa::span{host});
        list.add_callback([&observed, &host] {
            observed.store(host[0], std::memory_order_relaxed);
        });
        list.add_callback([&callback_count] {
            callback_count.fetch_add(1u, std::memory_order_relaxed);
        });
        stream << list.commit() << synchronize();
        expect(observed.load() == i + 1u)
            << "the callback must observe the copy enqueued before it (iteration "
            << i << ": got " << observed.load() << ", expected " << i + 1u << ")";
    }
    expect(callback_count.load() == kSubmissions)
        << "exactly one callback run per submission (got " << callback_count.load()
        << ", expected " << kSubmissions << ")";
    LUISA_INFO("host callbacks: ticket protocol passed ({} submissions).", kSubmissions);

    // FIFO order inside one package - the shader printer relies on it.
    constexpr auto kPackage = 16u;
    std::atomic_uint next{0u};
    std::atomic_bool fifo_ok{true};
    CommandList package;
    package << shader(buffer, 0u).dispatch(1u);
    for (auto i = 0u; i < kPackage; i++) {
        package.add_callback([&next, &fifo_ok, i] {
            if (next.fetch_add(1u, std::memory_order_relaxed) != i) {
                fifo_ok.store(false, std::memory_order_relaxed);
            }
        });
    }
    stream << package.commit() << synchronize();
    expect(next.load() == kPackage)
        << "every callback of the package must run (got " << next.load() << ")";
    expect(fifo_ok.load()) << "the callbacks of one package must run in enqueue order";
    LUISA_INFO("host callbacks: package FIFO order passed ({} callbacks).", kPackage);

    // A host function blocks work added *after* it: the copy submitted in the next
    // command list must not overtake the callback, so the callback still sees the
    // sentinel and not the value the copy writes.
    constexpr auto kSentinel = 0xdeadbeefu;
    host[0] = kSentinel;
    std::atomic_uint seen{0u};
    CommandList callback_only;
    callback_only << shader(buffer, 7u).dispatch(1u);
    callback_only.add_callback([&seen, &host] {
        seen.store(host[0], std::memory_order_relaxed);
    });
    stream << callback_only.commit();
    CommandList copy_after;
    copy_after << buffer.copy_to(luisa::span{host});
    stream << copy_after.commit() << synchronize();
    expect(seen.load() == kSentinel)
        << "a copy submitted after a host callback must not overtake it (got " << seen.load()
        << ", expected 0x" << std::hex << kSentinel << std::dec << ")";
    LUISA_INFO("host callbacks: post-callback work ordering passed.");

    // TODO(driver): the CUDA_ERROR_NOT_SUPPORTED / CUDA_ERROR_INVALID_VALUE
    // re-issue branches of the two resolvers - the host-callback path re-issuing the
    // unversioned cuLaunchHostFunc, and the stream-form PDL launch falling back to
    // cuLaunchKernel after the driver refuses the launch attribute - are only
    // reachable when a driver actually refuses. On driver 13.2 both entry points are
    // accepted, so neither branch runs here; the "spin vs blocking" mode itself is
    // covered by the TODO(driver) note in the file header.
    // TODO(driver): StreamCallbackSemaphoreUpdate::recycle()'s idempotency is only
    // exercised when a driver reports an error *after* the host function was
    // enqueued, which cannot be forced portably; it is reviewed logically only.
}

}// namespace test_cuda_feature_adoption_detail

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    if (luisa::string_view{device.backend_name()} != "cuda") {
        LUISA_INFO("CUDA-only feature test; skipping backend '{}'.", device.backend_name());
        return 0;
    }
    using namespace test_cuda_feature_adoption_detail;
    "cuda_graph_programmatic_capability_gate"_test = [&] { test_capability_gate(device); };
    "cuda_stream_pdl_precondition"_test = [] { test_stream_pdl_precondition(); };
    "cuda_shader_metadata_records_the_grid_dependency_wait"_test = [&] {
        test_shader_metadata_records_the_wait(device);
    };
    "cuda_host_callback_protocol"_test = [&] { test_host_callback_protocol(device); };
    return 0;
}
