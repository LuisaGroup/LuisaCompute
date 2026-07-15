#include "ut/ut.hpp"
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/passes/fuse_consecutive_buffer_reads.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_kernel(Module &m, BasicBlock *&body,
                                          ResourceArgument *&buffer,
                                          ValueArgument *&base) noexcept {
    auto *k = m.create_kernel();
    buffer = k->create_resource_argument(Type::buffer(Type::of<float>()));
    base = k->create_value_argument(Type::of<uint>());
    body = k->create_body_block();
    return k;
}

[[nodiscard]] Value *offset_index(XIRBuilder &b, Module &m, Value *base,
                                  uint32_t offset) noexcept {
    if (offset == 0u) { return base; }
    auto *c = m.create_constant(Type::of<uint>(), &offset);
    return b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD, {base, c});
}

ResourceReadInst *read_at(XIRBuilder &b, Module &m, Value *buffer,
                          Value *base, uint32_t offset) noexcept {
    auto *index = offset_index(b, m, base, offset);
    return b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ, {buffer, index});
}

ResourceWriteInst *write_at(XIRBuilder &b, Module &m, Value *buffer,
                            Value *base, uint32_t offset,
                            Value *value) noexcept {
    auto *index = offset_index(b, m, base, offset);
    return b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, index, value});
}

[[nodiscard]] size_t count_reads(FunctionDefinition *def, const Type *type) noexcept {
    size_t count = 0u;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ResourceReadInst>() && inst->type() == type) { ++count; }
    });
    return count;
}

[[nodiscard]] size_t count_writes(FunctionDefinition *def, const Type *value_type) noexcept {
    size_t count = 0u;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ResourceWriteInst>() && inst->operand(2)->type() == value_type) {
            ++count;
        }
    });
    return count;
}

[[nodiscard]] ResourceWriteInst *find_write(FunctionDefinition *def,
                                            const Type *value_type) noexcept {
    ResourceWriteInst *result = nullptr;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ResourceWriteInst>() && inst->operand(2)->type() == value_type) {
            result = static_cast<ResourceWriteInst *>(inst);
        }
    });
    return result;
}

[[nodiscard]] bool appears_before(BasicBlock *block, Instruction *a, Instruction *b) noexcept {
    bool saw_a = false;
    for (auto *inst : block->instructions()) {
        if (inst == a) { saw_a = true; }
        if (inst == b) { return saw_a; }
    }
    return false;
}

}// namespace

void reg_fuse_consecutive_buffer_reads() {

    "fuse_four_contiguous_reads"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        for (uint32_t i = 0u; i < 4u; ++i) { read_at(b, m, buffer, base, i); }
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);
        expect(info.fused_group_count == 1u);
        expect(info.fused_read_count == 4u);
        expect(count_reads(k, Type::of<float>()) == 0u);
        expect(count_reads(k, Type::of<float4>()) == 1u);
    };

    "sparse_offsets_are_not_mistaken_for_dense_vector"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        read_at(b, m, buffer, base, 0u);
        read_at(b, m, buffer, base, 2u);
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);
        expect(info.fused_group_count == 0u);
        expect(info.fused_read_count == 0u);
        expect(count_reads(k, Type::of<float>()) == 2u);
        expect(count_reads(k, Type::of<float2>()) == 0u);
    };

    "dense_prefix_fuses_without_consuming_sparse_tail"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        read_at(b, m, buffer, base, 0u);
        read_at(b, m, buffer, base, 1u);
        read_at(b, m, buffer, base, 3u);
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);
        expect(info.fused_group_count == 1u);
        expect(info.fused_read_count == 2u);
        expect(count_reads(k, Type::of<float>()) == 1u);
        expect(count_reads(k, Type::of<float2>()) == 1u);
    };

    "reverse_program_order_is_left_unchanged"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        read_at(b, m, buffer, base, 1u);
        read_at(b, m, buffer, base, 0u);
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);
        expect(info.fused_group_count == 0u);
        expect(count_reads(k, Type::of<float>()) == 2u);
    };

    "buffer_write_splits_read_fusion_epoch"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        read_at(b, m, buffer, base, 0u);
        write_at(b, m, buffer, base, 1u, m.create_constant_one(Type::of<float>()));
        read_at(b, m, buffer, base, 1u);
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);
        expect(info.fused_group_count == 0u);
        expect(count_reads(k, Type::of<float>()) == 2u);
        expect(count_reads(k, Type::of<float2>()) == 0u);
    };

    "fused_write_is_inserted_after_all_scalar_values"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<float>());
        write_at(b, m, buffer, base, 0u, one);
        auto *later_value = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {one, one});
        write_at(b, m, buffer, base, 1u, later_value);
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);
        expect(info.fused_group_count == 1u);
        expect(info.fused_read_count == 2u);
        expect(count_writes(k, Type::of<float>()) == 0u);
        expect(count_writes(k, Type::of<float2>()) == 1u);
        auto *fused = find_write(k, Type::of<float2>());
        expect(fused != nullptr);
        expect(appears_before(body, later_value, fused));
    };

    "buffer_read_splits_write_fusion_epoch"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<float>());
        write_at(b, m, buffer, base, 0u, one);
        read_at(b, m, buffer, base, 0u);
        write_at(b, m, buffer, base, 1u, one);
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);
        expect(info.fused_group_count == 0u);
        expect(count_writes(k, Type::of<float>()) == 2u);
        expect(count_writes(k, Type::of<float2>()) == 0u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_fuse_consecutive_buffer_reads();
    return 0;
}
