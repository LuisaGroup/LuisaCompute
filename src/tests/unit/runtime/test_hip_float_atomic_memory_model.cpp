#include "ut/ut.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace boost::ut;
using namespace luisa;
using namespace luisa::compute;

namespace {

[[nodiscard]] bool contains_agent_atomic(
    const std::string &module, std::string_view operation) noexcept {
    auto position = module.find(operation);
    while (position != std::string::npos) {
        const auto line_end = module.find('\n', position);
        const auto line = std::string_view{module}.substr(
            position, line_end == std::string::npos ?
                          std::string::npos : line_end - position);
        if (line.find("syncscope(\"agent\")") != std::string_view::npos &&
            line.find("monotonic") != std::string_view::npos) {
            return true;
        }
        position = module.find(operation, position + operation.size());
    }
    return false;
}

}// namespace

int main(int argc, char *argv[]) {
    Context context{argc > 0 && argv != nullptr ? argv[0] : ""};
    auto device = context.create_device("hip");

    std::error_code filesystem_error;
    const auto original_directory =
        std::filesystem::current_path(filesystem_error);
    const auto dump_directory =
        std::filesystem::temp_directory_path(filesystem_error) /
        ("luisa_hip_float_atomic_memory_model_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dump_directory, filesystem_error);
    std::filesystem::current_path(dump_directory, filesystem_error);
    expect(!filesystem_error)
        << "failed to prepare isolated HIP LLVM dump directory";
#if defined(_WIN32)
    _putenv_s("LUISA_DUMP_LLVM_IR", "1");
#else
    setenv("LUISA_DUMP_LLVM_IR", "1", 1);
#endif

    Kernel1D kernel = [](BufferFloat film,
                         BufferFloat returned,
                         BufferUInt queue_count) noexcept {
        const auto index = dispatch_x();
        queue_count.atomic(0u).fetch_sub(1u);
        returned.write(index * 2u,
                       film.atomic(index).fetch_add(1.25f));
        returned.write(index * 2u + 1u,
                       film.atomic(index).fetch_sub(0.5f));
    };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = false});

    auto film = device.create_buffer<float>(1u);
    auto returned = device.create_buffer<float>(2u);
    auto queue_count = device.create_buffer<uint>(1u);
    auto film_value = 4.0f;
    std::array returned_values{0.0f, 0.0f};
    auto count = 1u;
    auto stream = device.create_stream();
    stream << film.copy_from(luisa::span{&film_value, 1u})
           << queue_count.copy_from(luisa::span{&count, 1u})
           << shader(film, returned, queue_count).dispatch(1u)
           << film.copy_to(luisa::span{&film_value, 1u})
           << returned.copy_to(luisa::span{returned_values})
           << queue_count.copy_to(luisa::span{&count, 1u})
           << synchronize();
    expect(returned_values[0] == 4.0f);
    expect(returned_values[1] == 5.25f);
    expect(film_value == 4.75f);
    expect(count == 0u);

    // A source basic block may be refined into a CAS-loop region during HIP
    // lowering. Its outgoing SSA edge originates at the region's exit, not at
    // the source block's LLVM entry. Exercise a value returned by the atomic
    // on one arm of a merge so LLVM verification and execution both guard the
    // PHI predecessor mapping.
    Kernel1D atomic_phi_kernel = [](BufferFloat value,
                                    BufferFloat result,
                                    UInt take_atomic) noexcept {
        Float selected = 0.0f;
        $if (take_atomic != 0u) {
            selected = value.atomic(0u).fetch_add(2.0f);
        }
        $else {
            selected = value.read(0u);
        };
        result.write(0u, selected);
    };
    auto atomic_phi_shader = device.compile(
        atomic_phi_kernel, ShaderOption{.enable_cache = false});
    film_value = 3.0f;
    returned_values[0] = 0.0f;
    stream << film.copy_from(luisa::span{&film_value, 1u})
           << atomic_phi_shader(film, returned, 1u).dispatch(1u)
           << film.copy_to(luisa::span{&film_value, 1u})
           << returned.copy_to(
                  luisa::span{returned_values}.subspan(0u, 1u))
           << synchronize();
    expect(returned_values[0] == 3.0f);
    expect(film_value == 5.0f);

    auto matched_lowering = false;
    auto retained_raw_float_load = false;
    auto retained_raw_cmpxchg = false;
    for (const auto &entry :
         std::filesystem::directory_iterator(dump_directory)) {
        const auto filename = entry.path().filename().string();
        if (!filename.starts_with("hip_kernel_before_opt_") ||
            entry.path().extension() != ".ll") {
            continue;
        }
        std::ifstream input{entry.path()};
        const std::string module{
            std::istreambuf_iterator<char>{input},
            std::istreambuf_iterator<char>{}};
        if (module.find("float.cas.initial") == std::string::npos) {
            continue;
        }
        matched_lowering |=
            contains_agent_atomic(module, "load atomic i32") &&
            contains_agent_atomic(module, "cmpxchg");
        retained_raw_float_load |=
            module.find("llvm.amdgcn.raw.buffer.load.f32") !=
            std::string::npos;
        retained_raw_cmpxchg |=
            module.find("llvm.amdgcn.raw.buffer.atomic.cmpswap") !=
            std::string::npos;
    }
    expect(matched_lowering)
        << "HIP float CAS RMW must use agent-scope atomic load and cmpxchg";
    expect(!retained_raw_float_load)
        << "HIP float CAS RMW retained a non-atomic raw-buffer initial load";
    expect(!retained_raw_cmpxchg)
        << "HIP float CAS RMW escaped LLVM's atomic memory model";

    std::filesystem::current_path(original_directory, filesystem_error);
    std::filesystem::remove_all(dump_directory, filesystem_error);
}
