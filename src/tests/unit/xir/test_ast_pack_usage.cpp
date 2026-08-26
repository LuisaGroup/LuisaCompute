#include "ut/ut.hpp"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/translators/ast2xir.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool usage_contains(Usage usage, Usage expected) noexcept {
    return (to_underlying(usage) & to_underlying(expected)) ==
           to_underlying(expected);
}

struct XIRPackCounts {
    size_t buffer_reads{0u};
    size_t buffer_writes{0u};
    size_t bitwise_casts{0u};
};

[[nodiscard]] XIRPackCounts count_pack_instructions(const Module &module) noexcept {
    XIRPackCounts counts;
    for (auto function : module.function_list()) {
        auto definition = function->definition();
        if (definition == nullptr) { continue; }
        definition->traverse_instructions([&](const Instruction *inst) noexcept {
            if (inst->isa<ResourceReadInst>() &&
                static_cast<const ResourceReadInst *>(inst)->op() == ResourceReadOp::BUFFER_READ) {
                counts.buffer_reads++;
            } else if (inst->isa<ResourceWriteInst>() &&
                       static_cast<const ResourceWriteInst *>(inst)->op() == ResourceWriteOp::BUFFER_WRITE) {
                counts.buffer_writes++;
            } else if (inst->isa<CastInst>() &&
                       static_cast<const CastInst *>(inst)->op() == xir::CastOp::BITWISE_CAST) {
                counts.bitwise_casts++;
            }
        });
    }
    return counts;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "pack_marks_destination_write_and_lowers_float3_to_four_words"_test = [] {
        Kernel1D pack = [](Float3 value, BufferUInt words) noexcept {
            pack_to(value, words, 0u);
        };
        compute::Function ast_function{pack.function().get()};
        auto arguments = ast_function.arguments();
        expect(arguments.size() == 2u);
        expect(usage_contains(ast_function.variable_usage(arguments[0].uid()), Usage::READ));
        expect(usage_contains(ast_function.variable_usage(arguments[1].uid()), Usage::WRITE));

        auto module = ast_to_xir_translate(ast_function, {});
        auto counts = count_pack_instructions(*module);
        expect(counts.buffer_reads == 0u);
        expect(counts.buffer_writes == 4u);
        expect(counts.bitwise_casts == 1u);
    };

    "unpack_marks_source_read_and_lowers_float3_from_four_words"_test = [] {
        Kernel1D unpack = [](BufferUInt words, BufferFloat3 output) noexcept {
            output.write(0u, unpack_from<float3>(words, 0u));
        };
        compute::Function ast_function{unpack.function().get()};
        auto arguments = ast_function.arguments();
        expect(arguments.size() == 2u);
        expect(usage_contains(ast_function.variable_usage(arguments[0].uid()), Usage::READ));
        expect(usage_contains(ast_function.variable_usage(arguments[1].uid()), Usage::WRITE));

        auto module = ast_to_xir_translate(ast_function, {});
        auto counts = count_pack_instructions(*module);
        expect(counts.buffer_reads == 4u);
        expect(counts.buffer_writes == 1u);
        expect(counts.bitwise_casts == 1u);
    };

    "custom_opaque_argument_propagates_callable_write_usage"_test = [] {
        Callable set_count = [](Var<IndirectDispatchBuffer> buffer) noexcept {
            buffer.set_dispatch_count(1u);
        };
        Kernel1D kernel = [&set_count](Var<IndirectDispatchBuffer> buffer) noexcept {
            set_count(buffer);
        };
        compute::Function ast_function{kernel.function().get()};
        auto arguments = ast_function.arguments();
        expect(arguments.size() == 1u);
        expect(usage_contains(ast_function.variable_usage(arguments[0].uid()), Usage::WRITE));
    };

    return 0;
}
