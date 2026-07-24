#include "ut/ut.hpp"

#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
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

[[nodiscard]] ResourceReadInst *read_at(XIRBuilder &b, Module &m, Value *buffer,
                                        Value *base, uint32_t offset) noexcept {
    auto *index = offset_index(b, m, base, offset);
    return b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ, {buffer, index});
}

[[nodiscard]] ResourceWriteInst *write_at(XIRBuilder &b, Module &m, Value *buffer,
                                          Value *base, uint32_t offset,
                                          Value *value) noexcept {
    auto *index = offset_index(b, m, base, offset);
    return b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, index, value});
}

[[nodiscard]] luisa::vector<Instruction *> instruction_sequence(BasicBlock *block) noexcept {
    luisa::vector<Instruction *> result;
    for (auto *inst : block->instructions()) { result.emplace_back(inst); }
    return result;
}

void expect_typed_buffer_accesses_valid(FunctionDefinition *def) noexcept {
    def->traverse_instructions([](Instruction *inst) noexcept {
        if (inst->isa<ResourceReadInst>()) {
            auto *read = static_cast<ResourceReadInst *>(inst);
            if (read->op() == ResourceReadOp::BUFFER_READ) {
                auto *buffer_type = read->operand(0u)->type();
                expect(buffer_type->is_buffer());
                expect(read->type() == buffer_type->element())
                    << "typed buffer read type must match the buffer element type";
            }
        } else if (inst->isa<ResourceWriteInst>()) {
            auto *write = static_cast<ResourceWriteInst *>(inst);
            if (write->op() == ResourceWriteOp::BUFFER_WRITE) {
                auto *buffer_type = write->operand(0u)->type();
                expect(buffer_type->is_buffer());
                expect(write->operand(2u)->type() == buffer_type->element())
                    << "typed buffer write type must match the buffer element type";
            }
        }
    });
}

[[nodiscard]] KernelFunction *make_byte_kernel(Module &m, BasicBlock *&body,
                                               ResourceArgument *&buffer,
                                               ValueArgument *&base) noexcept {
    auto *k = m.create_kernel();
    buffer = k->create_resource_argument(Type::buffer(Type::of<uint>()));
    base = k->create_value_argument(Type::of<uint>());
    body = k->create_body_block();
    return k;
}

[[nodiscard]] ResourceReadInst *byte_read_at(XIRBuilder &b, Module &m, Value *buffer,
                                             Value *base, uint32_t byte_offset) noexcept {
    auto *index = offset_index(b, m, base, byte_offset);
    return b.call(Type::of<float>(), ResourceReadOp::BYTE_BUFFER_READ, {buffer, index});
}

[[nodiscard]] ResourceWriteInst *byte_write_at(XIRBuilder &b, Module &m, Value *buffer,
                                               Value *base, uint32_t byte_offset,
                                               Value *value) noexcept {
    auto *index = offset_index(b, m, base, byte_offset);
    return b.call(ResourceWriteOp::BYTE_BUFFER_WRITE, {buffer, index, value});
}

}// namespace

void reg_fuse_consecutive_buffer_reads() {

    "typed_scalar_reads_are_quarantined_unchanged"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        luisa::vector<ResourceReadInst *> reads;
        for (uint32_t i = 0u; i < 4u; ++i) {
            reads.emplace_back(read_at(b, m, buffer, base, i));
        }
        b.return_void();
        auto before = instruction_sequence(body);

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);

        expect(info.fused_group_count == 0u);
        expect(info.fused_read_count == 0u);
        expect(instruction_sequence(body) == before);
        for (auto *read : reads) {
            expect(read->is_linked());
            expect(read->type() == Type::of<float>());
            expect(read->operand(0u) == buffer);
        }
        expect_typed_buffer_accesses_valid(k);
    };

    "typed_scalar_writes_are_quarantined_unchanged"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<float>());
        auto *write0 = write_at(b, m, buffer, base, 0u, one);
        auto *write1 = write_at(b, m, buffer, base, 1u, one);
        auto *write2 = write_at(b, m, buffer, base, 2u, one);
        b.return_void();
        auto before = instruction_sequence(body);

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);

        expect(info.fused_group_count == 0u);
        expect(info.fused_read_count == 0u);
        expect(instruction_sequence(body) == before);
        expect(write0->is_linked());
        expect(write1->is_linked());
        expect(write2->is_linked());
        expect(write0->operand(2u) == one);
        expect(write1->operand(2u) == one);
        expect(write2->operand(2u) == one);
        expect_typed_buffer_accesses_valid(k);
    };

    "intervening_may_alias_write_order_is_preserved"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_kernel(m, body, buffer, base);
        auto *dynamic_index = k->create_value_argument(Type::of<uint>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<float>());
        auto *first = write_at(b, m, buffer, base, 0u, one);
        auto *middle = b.call(ResourceWriteOp::BUFFER_WRITE,
                              {buffer, dynamic_index, one});
        auto *last = write_at(b, m, buffer, base, 1u, one);
        b.return_void();
        auto before = instruction_sequence(body);

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);

        expect(info.fused_group_count == 0u);
        expect(instruction_sequence(body) == before);
        expect(first->is_linked());
        expect(middle->is_linked());
        expect(last->is_linked());
        expect_typed_buffer_accesses_valid(k);
    };

    "module_entry_point_reports_no_illegal_fusion"_test = [] {
        Module m;
        for (uint32_t f = 0u; f < 2u; ++f) {
            BasicBlock *body;
            ResourceArgument *buffer;
            ValueArgument *base;
            (void)make_kernel(m, body, buffer, base);
            XIRBuilder b;
            b.set_insertion_point(body);
            (void)read_at(b, m, buffer, base, 0u);
            (void)read_at(b, m, buffer, base, 1u);
            b.return_void();
        }

        auto info = fuse_consecutive_buffer_reads_pass_run_on_module(&m);

        expect(info.fused_group_count == 0u);
        expect(info.fused_read_count == 0u);
        for (auto *f : m.function_list()) {
            if (auto *def = f->definition()) {
                expect_typed_buffer_accesses_valid(def);
            }
        }
    };
}

void reg_fuse_consecutive_byte_buffer_accesses() {

    "two_adjacent_byte_reads_fuse_into_vector_read"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_byte_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *r0 = byte_read_at(b, m, buffer, base, 0u);
        auto *r1 = byte_read_at(b, m, buffer, base, 4u);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {r0, r1});
        static_cast<void>(sum);
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);

        expect(info.fused_group_count == 1u);
        expect(info.fused_read_count == 2u);
        // one vector read of float2 + two extracts + the add + terminator
        auto vector_read_count = 0u;
        auto extract_count = 0u;
        for (auto *inst : body->instructions()) {
            if (inst->isa<ResourceReadInst>()) {
                auto *read = static_cast<ResourceReadInst *>(inst);
                expect(read->op() == ResourceReadOp::BYTE_BUFFER_READ);
                expect(read->type() == Type::of<float2>());
                vector_read_count++;
            }
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::EXTRACT) {
                extract_count++;
            }
        }
        expect(vector_read_count == 1u);
        expect(extract_count == 2u);
    };

    "four_adjacent_byte_reads_fuse_into_float4_read"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_byte_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        luisa::vector<ResourceReadInst *> reads;
        for (uint32_t i = 0u; i < 4u; ++i) {
            reads.emplace_back(byte_read_at(b, m, buffer, base, i * 4u));
        }
        Value *acc = reads[0];
        for (auto i = 1u; i < 4u; ++i) {
            acc = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {acc, reads[i]});
        }
        static_cast<void>(acc);
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);

        expect(info.fused_group_count == 1u);
        expect(info.fused_read_count == 4u);
        auto vector_read_count = 0u;
        for (auto *inst : body->instructions()) {
            if (inst->isa<ResourceReadInst>()) {
                expect(static_cast<ResourceReadInst *>(inst)->type() == Type::of<float4>());
                vector_read_count++;
            }
        }
        expect(vector_read_count == 1u);
    };

    "non_consecutive_byte_offsets_do_not_fuse"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_byte_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *r0 = byte_read_at(b, m, buffer, base, 0u);
        auto *r1 = byte_read_at(b, m, buffer, base, 8u);
        auto *r2 = byte_read_at(b, m, buffer, base, 16u);
        static_cast<void>(r0);
        static_cast<void>(r1);
        static_cast<void>(r2);
        b.return_void();
        auto before = instruction_sequence(body);

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);

        expect(info.fused_group_count == 0u);
        expect(instruction_sequence(body) == before);
    };

    "intervening_write_to_same_buffer_blocks_fusion"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_byte_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<float>());
        auto *r0 = byte_read_at(b, m, buffer, base, 0u);
        auto *w = byte_write_at(b, m, buffer, base, 64u, one);
        auto *r1 = byte_read_at(b, m, buffer, base, 4u);
        static_cast<void>(r0);
        static_cast<void>(w);
        static_cast<void>(r1);
        b.return_void();
        auto before = instruction_sequence(body);

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);

        expect(info.fused_group_count == 0u);
        expect(instruction_sequence(body) == before);
    };

    "reads_from_different_buffers_do_not_fuse"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_byte_kernel(m, body, buffer, base);
        auto *other = k->create_resource_argument(Type::buffer(Type::of<uint>()));
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *r0 = byte_read_at(b, m, buffer, base, 0u);
        auto *r1 = byte_read_at(b, m, other, base, 4u);
        static_cast<void>(r0);
        static_cast<void>(r1);
        b.return_void();
        auto before = instruction_sequence(body);

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);

        expect(info.fused_group_count == 0u);
        expect(instruction_sequence(body) == before);
    };

    "three_adjacent_byte_writes_fuse_into_vector_write"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_byte_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<float>());
        auto two_value = 2.0f;
        auto three_value = 3.0f;
        auto *two = m.create_constant(Type::of<float>(), &two_value);
        auto *three = m.create_constant(Type::of<float>(), &three_value);
        auto *w0 = byte_write_at(b, m, buffer, base, 0u, one);
        auto *w1 = byte_write_at(b, m, buffer, base, 4u, two);
        auto *w2 = byte_write_at(b, m, buffer, base, 8u, three);
        static_cast<void>(w0);
        static_cast<void>(w1);
        static_cast<void>(w2);
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);

        expect(info.fused_group_count == 1u);
        expect(info.fused_write_count == 3u);
        auto vector_write_count = 0u;
        auto aggregate_count = 0u;
        for (auto *inst : body->instructions()) {
            if (inst->isa<ResourceWriteInst>()) {
                auto *write = static_cast<ResourceWriteInst *>(inst);
                expect(write->op() == ResourceWriteOp::BYTE_BUFFER_WRITE);
                expect(write->operand(2u)->type() == Type::of<float3>());
                vector_write_count++;
            }
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::AGGREGATE) {
                aggregate_count++;
            }
        }
        expect(vector_write_count == 1u);
        expect(aggregate_count == 1u);
    };

    "two_separate_adjacent_groups_fuse_independently"_test = [] {
        Module m;
        BasicBlock *body;
        ResourceArgument *buffer;
        ValueArgument *base;
        auto *k = make_byte_kernel(m, body, buffer, base);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *r0 = byte_read_at(b, m, buffer, base, 0u);
        auto *r1 = byte_read_at(b, m, buffer, base, 4u);
        auto *r2 = byte_read_at(b, m, buffer, base, 16u);
        auto *r3 = byte_read_at(b, m, buffer, base, 20u);
        static_cast<void>(r0);
        static_cast<void>(r1);
        static_cast<void>(r2);
        static_cast<void>(r3);
        b.return_void();

        auto info = fuse_consecutive_buffer_reads_pass_run_on_function(k);

        expect(info.fused_group_count == 2u);
        expect(info.fused_read_count == 4u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_fuse_consecutive_buffer_reads();
    reg_fuse_consecutive_byte_buffer_accesses();
    return 0;
}
