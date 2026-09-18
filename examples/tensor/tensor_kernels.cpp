// =============================================================================
// tensor_kernels.cpp — Shared result accounting + tile compilation helpers
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

namespace {
int &pass_counter() noexcept {
    static int count = 0;
    return count;
}
int &fail_counter() noexcept {
    static int count = 0;
    return count;
}
}// namespace

void record(string_view name, bool ok, string_view detail) noexcept {
    if (ok) {
        pass_counter()++;
        LUISA_INFO("[tensor] PASS  {}{}", name, detail.empty() ? string_view{} : string_view{" — "});
        if (!detail.empty()) { LUISA_INFO("[tensor]       {}", detail); }
    } else {
        fail_counter()++;
        LUISA_WARNING("[tensor] FAIL  {}{}", name, detail.empty() ? string_view{} : string_view{" — "});
        if (!detail.empty()) { LUISA_WARNING("[tensor]       {}", detail); }
    }
}

int failure_count() noexcept { return fail_counter(); }
int pass_count() noexcept { return pass_counter(); }

void check(string_view name, double err, double tol) noexcept {
    auto ok = std::abs(err) <= tol && std::isfinite(err);
    record(name, ok, luisa::format("max err = {:.6g} (tol {:.6g})", err, tol));
}

void skip(string_view name, string_view detail) noexcept {
    LUISA_INFO("[tensor] SKIP  {} — {}", name, detail);
}

namespace {
string_view backend_name_storage;
}

void set_active_backend(string_view name) noexcept { backend_name_storage = name; }
string_view active_backend() noexcept { return backend_name_storage; }

tile::Shader compile_tile(lc::Device &device, const tile::Kernel &kernel,
                          string_view name, tile::Lowering lowering) noexcept {
    if (!kernel.valid()) {
        tile::KernelMetadata metadata;
        metadata.error = "captured TileIR is invalid";
        for (auto &&d : kernel.diagnostics()) {
            metadata.error += "\n  ";
            metadata.error += d;
        }
        return tile::Shader{device.impl(), lc::ShaderCreationInfo::make_invalid(), std::move(metadata)};
    }
    auto verified = tile::verify(kernel.module());
    if (!verified.ok()) {
        tile::KernelMetadata metadata;
        metadata.error = luisa::format("TileIR verification failed ({} diagnostic(s))", verified.diagnostics().size());
        for (auto &&d : verified.diagnostics()) {
            metadata.error += "\n  ";
            metadata.error += d.message;
        }
        return tile::Shader{device.impl(), lc::ShaderCreationInfo::make_invalid(), std::move(metadata)};
    }
    auto shader = tile::compile(device, kernel, {.lowering = lowering});
    if (!shader && lowering == tile::Lowering::NATIVE &&
        shader.metadata().error.find("TIRX") != luisa::string::npos) {
        // CUDA ships its Tile realization on the TVM TIRx bridge; retry there.
        LUISA_INFO("[tensor] {}: NATIVE lowering unavailable ({}); retrying with TIRX.",
                   name, shader.metadata().error);
        shader = tile::compile(device, kernel, {.lowering = tile::Lowering::TIRX});
        if (shader) { LUISA_INFO("[tensor] {}: compiled with TIRX lowering.", name); }
    }
    if (!shader) {
        LUISA_WARNING("[tensor] {}: tile::compile failed: {}", name, shader.metadata().error);
    } else {
        auto &&m = shader.metadata();
        LUISA_INFO("[tensor] {}: dispatch=({},{},{}), {} argument(s), realization: {}",
                   name, m.dispatch_size.x, m.dispatch_size.y, m.dispatch_size.z,
                   m.arguments.size(), m.realization);
    }
    return shader;
}

}// namespace tensor_example
