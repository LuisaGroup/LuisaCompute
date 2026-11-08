// Host-only regression for the AST bit the CUDA backend uses to detect a
// kernel that waits for its programmatic dependency
// (grid_dependency_synchronize()). The launch path cannot re-derive it - a
// shader-cache hit never runs codegen - so CUDAShaderMetadata records
// Function::propagated_builtin_callables() at creation time. That set must
// therefore contain GRID_DEPENDENCY_SYNCHRONIZE both for a kernel that calls the
// wait directly and for a kernel that calls a user callable which waits
// (src/ast/function_builder.cpp propagates it through every callable it uses).
//
// The bit is a *specific* builtin, not "some builtin call": an ordinary
// operation such as erf() must leave it clear, otherwise the CUDA launch
// path would relax the stream serialization for a kernel that never waits.
// The wire names of the appended CallOps are asserted here as well, because
// the HLSL/SPIR-V diagnostics and the AST serializers key on them.

#include "ut/ut.hpp"

#include <array>

#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>
#include <luisa/core/magic_enum.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace test_builtin_call_propagation_detail {

using FuncBuilder = luisa::compute::detail::FunctionBuilder;

// The exact predicate CUDADevice::create_shader stores in the shader metadata.
[[nodiscard]] bool waits_for_programmatic_dependency(
    const luisa::shared_ptr<const FuncBuilder> &builder) noexcept {
    return Function{builder.get()}.propagated_builtin_callables().test(
        CallOp::GRID_DEPENDENCY_SYNCHRONIZE);
}

}// namespace test_builtin_call_propagation_detail

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    using test_builtin_call_propagation_detail::waits_for_programmatic_dependency;

    "grid_dependency_synchronize_is_recorded_on_the_kernel"_test = [] {
        Kernel1D waiting = [] {
            grid_dependency_synchronize();
        };
        expect(waits_for_programmatic_dependency(waiting.function()))
            << "a kernel that waits for its programmatic dependency must report it";
        Kernel1D plain = [] {};
        expect(!waits_for_programmatic_dependency(plain.function()))
            << "a kernel without the wait must not report it";
    };

    "grid_dependency_synchronize_is_propagated_from_a_callable"_test = [] {
        Callable<void()> wait = [] {
            grid_dependency_synchronize();
        };
        expect(waits_for_programmatic_dependency(wait.function_builder()))
            << "the callable itself must record the wait";
        Kernel1D caller = [&wait] {
            wait();
        };
        expect(waits_for_programmatic_dependency(caller.function()))
            << "the wait must be propagated into every caller of the callable";
    };

    "a_plain_callable_does_not_mark_its_caller"_test = [] {
        Callable<void()> noop = [] {};
        Kernel1D caller = [&noop] {
            noop();
        };
        expect(!waits_for_programmatic_dependency(caller.function()))
            << "using a callable without the wait must not set the flag";
    };

    "an_ordinary_builtin_does_not_set_the_wait_bit"_test = [] {
        Callable<float(float)> ordinary = [](Float x) noexcept {
            return erf(x);
        };
        expect(!waits_for_programmatic_dependency(ordinary.function_builder()))
            << "erf must not be confused with the programmatic-dependency wait";
        Kernel1D direct = [](BufferFloat buffer) noexcept {
            buffer.write(0u, erf(buffer.read(0u)));
        };
        expect(!waits_for_programmatic_dependency(direct.function()))
            << "a kernel that only calls erf must keep the serialized launch";
        Kernel1D via_callable = [&ordinary](BufferFloat buffer) noexcept {
            buffer.write(0u, ordinary(buffer.read(0u)));
        };
        expect(!waits_for_programmatic_dependency(via_callable.function()))
            << "calling an erf callable must not set the wait bit";
        // Positive control inside the same test: the identical kernel *with* the
        // wait does set the bit, so the three assertions above cannot pass just
        // because the bit is never propagated at all.
        Kernel1D waiting_with_erf = [](BufferFloat buffer) noexcept {
            grid_dependency_synchronize();
            buffer.write(0u, erf(buffer.read(0u)));
        };
        expect(waits_for_programmatic_dependency(waiting_with_erf.function()))
            << "the wait must still be recorded next to an ordinary builtin call";
    };

    "special_function_call_op_wire_names_round_trip"_test = [] {
        constexpr std::array appended{CallOp::ERF, CallOp::ERFC,
                                      CallOp::EXPM1, CallOp::LDEXP};
        for (auto op : appended) {
            auto name = luisa::to_string(op);
            // The contract is that the four appended members are inside
            // LUISA_MAGIC_ENUM_RANGE (otherwise the name is empty) plus the four
            // spellings pinned below. A magic_enum::enum_cast(name) == op round trip
            // would only re-test magic_enum itself - luisa::to_string is literally
            // magic_enum::enum_name - so it is deliberately not asserted.
            expect(!name.empty())
                << "every appended CallOp needs a stable wire name";
        }
        // The names are part of the user-visible diagnostics, so pin them.
        expect(luisa::to_string(CallOp::ERF) == "ERF");
        expect(luisa::to_string(CallOp::ERFC) == "ERFC");
        expect(luisa::to_string(CallOp::EXPM1) == "EXPM1");
        expect(luisa::to_string(CallOp::LDEXP) == "LDEXP");
    };
}
