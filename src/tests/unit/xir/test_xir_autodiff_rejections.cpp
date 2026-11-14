// Host-only fatal-rejection test for the XIR autodiff pass.
//
// Feature under test: the elementary special functions added to the XIR
// ArithmeticOp enum (ERF, ERFC, EXPM1, LDEXP) and the mixed-precision DOT2ADD
// have no derivative rule, so both
// autodiff directions must reject them with a named, fatal diagnostic instead of
// silently producing a wrong gradient:
//   reverse mode: "Unsupported reverse-mode autodiff arithmetic operation <op>."
//   forward mode: "Unsupported forward-mode autodiff arithmetic operation <op>."
// where <op> is the lower-case XIR wire name from src/xir/op.cpp.
//
// LuisaCompute reports this through LUISA_ERROR (log + abort), so - as the test
// skill's "Fatal checks without exceptions" rule requires - every case runs in a
// separate child process: this executable re-executes itself with the
// `--autodiff-probe <op> <mode>` marker, the parent redirects the child's output
// into a temporary file and asserts both the non-zero status and the exact
// message. The XIR pass is target-independent, so no device is created at all.
//
// The `sin` control run in `autodiff_probe_harness_control` proves the harness
// (and the hand-built autodiff scope) is sound: an op *with* a derivative rule
// transforms and exits 0, so the eight aborts below are op-specific and not an
// artifact of a malformed probe kernel.
//
// TODO(backend): the device-side path of these ops (a kernel under $autodiff
// compiled through a real backend) is not covered here: the pass itself is
// front-end code, so a device would only add its own diagnostics. Cover it on
// cuda via test_cuda_feature_adoption-style device compilation once the backend
// routes autodiff through the LLVM codegen path.

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/autodiff.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>

using namespace boost::ut;
using namespace boost::ut::literals;

namespace test_xir_autodiff_rejections_detail {

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;

constexpr std::string_view child_probe = "--autodiff-probe";

[[nodiscard]] ArithmeticOp arithmetic_op_from_name(std::string_view name) noexcept {
    if (name == "erf") { return ArithmeticOp::ERF; }
    if (name == "erfc") { return ArithmeticOp::ERFC; }
    if (name == "expm1") { return ArithmeticOp::EXPM1; }
    if (name == "ldexp") { return ArithmeticOp::LDEXP; }
    if (name == "dot2add") { return ArithmeticOp::DOT2ADD; }
    if (name == "sin") { return ArithmeticOp::SIN; }
    LUISA_ERROR("Unknown autodiff probe op '{}'.", name);
}

/// ERF/ERFC/EXPM1 are unary (float -> float); LDEXP additionally takes the
/// integer exponent, i.e. the same operand shapes the XIR verifier requires.
[[nodiscard]] Value *emit_probe_op(Module &module, XIRBuilder &builder,
                                   ArithmeticOp op, Value *x) noexcept {
    switch (op) {
        case ArithmeticOp::ERF:
        case ArithmeticOp::ERFC:
        case ArithmeticOp::EXPM1:
        case ArithmeticOp::SIN: return builder.call(Type::of<float>(), op, {x});
        case ArithmeticOp::LDEXP: return builder.call(
            Type::of<float>(), op,
            {x, module.create_constant_one(Type::of<int32_t>())});
        case ArithmeticOp::DOT2ADD: {
            // The exact (half2, half2, float) -> float signature; the float
            // probe value is widened and packed into the half2 lanes.
            auto half_type = Type::of<half>();
            auto lane = builder.static_cast_(half_type, x);
            auto packed = builder.call(
                Type::of<half2>(), ArithmeticOp::AGGREGATE, {lane, lane});
            return builder.call(Type::of<float>(), op, {packed, packed, x});
        }
        default: break;
    }
    LUISA_ERROR("Not a probe op.");
}

/// Reverse mode: an autodiff scope with requires_gradient/gradient_marker/
/// backward, plus one epilogue gradient read so the pass really lowers the
/// epilogue (the shape of test_xir_pass_autodiff_cfg.cpp).
void build_reverse_mode_probe(Module &module, ArithmeticOp op,
                              KernelFunction *&kernel) noexcept {
    kernel = module.create_kernel();
    auto *body = kernel->create_body_block();
    auto *x = kernel->create_value_argument(Type::of<float>());
    auto *condition = kernel->create_value_argument(Type::of<bool>());
    auto *one = module.create_constant_one(Type::of<float>());
    XIRBuilder builder;
    builder.set_insertion_point(body);
    auto *gradient_out = builder.alloca_local(Type::of<float>());
    auto *scope = builder.autodiff_scope();
    auto *scope_entry = scope->create_entry_block();
    auto *scope_merge = scope->create_merge_block();

    builder.set_insertion_point(scope_entry);
    builder.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
    auto *y = emit_probe_op(module, builder, op, x);
    builder.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {y, one});
    builder.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
    auto *epilogue_if = builder.if_(condition);
    auto *true_block = epilogue_if->create_true_block();
    auto *false_block = epilogue_if->create_false_block();

    builder.set_insertion_point(true_block);
    auto *true_gradient = builder.call(
        Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
    builder.store(gradient_out, true_gradient);
    builder.br(scope_merge);
    builder.set_insertion_point(false_block);
    auto *false_gradient = builder.call(
        Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
    builder.store(gradient_out, false_gradient);
    builder.br(scope_merge);
    builder.set_insertion_point(scope_merge);
    builder.return_void();
}

/// Forward mode: a one-slot forward autodiff scope with a propagated seed and
/// the probe instruction inside the scope.
void build_forward_mode_probe(Module &module, ArithmeticOp op,
                              KernelFunction *&kernel) noexcept {
    kernel = module.create_kernel();
    auto *body = kernel->create_body_block();
    auto *x = kernel->create_value_argument(Type::of<float>());
    auto *one = module.create_constant_one(Type::of<float>());
    XIRBuilder builder;
    builder.set_insertion_point(body);
    auto *scope = builder.forward_autodiff_scope(1u);
    auto *scope_entry = scope->create_entry_block();
    auto *scope_merge = scope->create_merge_block();

    builder.set_insertion_point(scope_entry);
    builder.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT,
                 {x, one});
    static_cast<void>(emit_probe_op(module, builder, op, x));
    builder.br(scope_merge);
    builder.set_insertion_point(scope_merge);
    builder.return_void();
}

[[nodiscard]] int run_autodiff_probe(int argc, char *argv[]) noexcept {
    if (argc < 4 || argv[2] == nullptr || argv[3] == nullptr) {
        LUISA_ERROR("Usage: {} {} <erf|erfc|expm1|ldexp|dot2add|sin> <reverse|forward>.",
                    argc > 0 ? argv[0] : "test_xir_autodiff_rejections",
                    child_probe);
    }
    auto op_name = std::string_view{argv[2]};
    auto mode = std::string_view{argv[3]};
    auto op = arithmetic_op_from_name(op_name);
    Module module;
    KernelFunction *kernel = nullptr;
    if (mode == "reverse") {
        build_reverse_mode_probe(module, op, kernel);
    } else if (mode == "forward") {
        build_forward_mode_probe(module, op, kernel);
    } else {
        LUISA_ERROR("Unknown autodiff probe mode '{}'.", mode);
    }
    auto info = autodiff_pass_run_on_function(kernel);
    LUISA_INFO("autodiff probe: op '{}' mode '{}' unexpectedly returned and "
               "transformed {} scope(s) / removed {} instruction(s).",
               op_name, mode, info.transformed_scope_count,
               info.removed_instruction_count);
    return 0;
}

struct ChildResult {
    int status{0};
    std::string text;
};

[[nodiscard]] std::string read_text_file(const luisa::filesystem::path &path) {
    std::ifstream file{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

/// Runs the probe in a child process and returns its exit status and its
/// redirected output (the in-tree pattern of test_vk_native_route_guard.cpp).
[[nodiscard]] ChildResult run_autodiff_child(
    const std::string &executable_path, std::string_view op,
    std::string_view mode, const luisa::filesystem::path &log_path) {
    auto command = luisa::format(
        "\"{}\" {} {} {} > \"{}\" 2>&1",
        executable_path, child_probe, op, mode, log_path.string());
#ifdef _WIN32
    // system() goes through cmd /C, which strips the outer quote pair; wrap the
    // whole line so the executable and log paths stay quoted.
    command = luisa::format("\"{}\"", command);
#endif
    auto status = std::system(command.c_str());
    return {status, read_text_file(log_path)};
}

struct ScopedDirectoryCleanup {
    luisa::filesystem::path path;
    ~ScopedDirectoryCleanup() noexcept {
        std::error_code error;
        luisa::filesystem::remove_all(path, error);
    }
};

}// namespace test_xir_autodiff_rejections_detail

int main(int argc, char *argv[]) {

    // The child dispatch must happen before any Boost.UT registration, so that
    // the probe marker never reaches the framework's argument parser.
    if (argc >= 2 && argv[1] != nullptr &&
        std::strcmp(argv[1], test_xir_autodiff_rejections_detail::child_probe.data()) == 0) {
        return test_xir_autodiff_rejections_detail::run_autodiff_probe(argc, argv);
    }

    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    using namespace test_xir_autodiff_rejections_detail;

    auto executable_path = luisa::filesystem::absolute(argv[0]).string();
    auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    auto log_directory = luisa::filesystem::temp_directory_path() /
                         ("luisa_xir_autodiff_rejections_" + std::to_string(nonce));
    luisa::filesystem::create_directories(log_directory);
    ScopedDirectoryCleanup cleanup{log_directory};

    "autodiff_probe_harness_control"_test = [&] {
        auto log_path = log_directory / "control_sin_reverse.log";
        auto result = run_autodiff_child(executable_path, "sin", "reverse", log_path);
        expect(result.status == 0)
            << "the control op (sin, which has a derivative rule) must not abort; "
               "child output:\n"
            << result.text;
        expect(result.text.find("transformed 1 scope(s)") != std::string::npos)
            << "the control run must actually run the pass over the probe scope; "
               "child output:\n"
            << result.text;
    };

    "autodiff_reverse_mode_rejects_the_derivative_free_ops"_test = [&] {
        for (auto op : {"erf", "erfc", "expm1", "ldexp", "dot2add"}) {
            auto log_path = log_directory / luisa::format("reverse_{}.log", op);
            auto result = run_autodiff_child(executable_path, op, "reverse", log_path);
            expect(result.status != 0)
                << "reverse-mode autodiff must reject " << op
                << " instead of returning a wrong gradient; child output:\n"
                << result.text;
            auto expected = luisa::format(
                "Unsupported reverse-mode autodiff arithmetic operation {}.", op);
            expect(result.text.find(expected) != std::string::npos)
                << "the rejection must name the operation (" << expected
                << "); child output:\n"
                << result.text;
        }
    };

    "autodiff_forward_mode_rejects_the_derivative_free_ops"_test = [&] {
        for (auto op : {"erf", "erfc", "expm1", "ldexp", "dot2add"}) {
            auto log_path = log_directory / luisa::format("forward_{}.log", op);
            auto result = run_autodiff_child(executable_path, op, "forward", log_path);
            expect(result.status != 0)
                << "forward-mode autodiff must reject " << op
                << " instead of returning a wrong gradient; child output:\n"
                << result.text;
            auto expected = luisa::format(
                "Unsupported forward-mode autodiff arithmetic operation {}.", op);
            expect(result.text.find(expected) != std::string::npos)
                << "the rejection must name the operation (" << expected
                << "); child output:\n"
                << result.text;
        }
    };
}
