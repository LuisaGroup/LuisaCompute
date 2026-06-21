#include <luisa/xir/passes/fuse_consecutive_buffer_reads.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/arithmetic.h>

namespace luisa::compute::xir {

namespace {

// Check if two indices differ by a constant offset.
// Returns the offset if they do, otherwise returns INT64_MAX.
[[nodiscard]] int64_t constant_offset_between(Value *idx_a, Value *idx_b) noexcept {
    if (idx_a == idx_b) return 0;
    // Case 1: idx_b = idx_a + constant
    if (idx_b->isa<ArithmeticInst>()) {
        auto *arith = static_cast<ArithmeticInst *>(idx_b);
        if (arith->op() == ArithmeticOp::BINARY_ADD) {
            if (arith->operand(0) == idx_a && arith->operand(1)->isa<Constant>()) {
                auto *c = static_cast<Constant *>(arith->operand(1));
                if (c->type()->is_uint32()) return static_cast<int64_t>(c->as<uint32_t>());
                if (c->type()->is_int32()) return static_cast<int64_t>(c->as<int32_t>());
            }
            if (arith->operand(1) == idx_a && arith->operand(0)->isa<Constant>()) {
                auto *c = static_cast<Constant *>(arith->operand(0));
                if (c->type()->is_uint32()) return static_cast<int64_t>(c->as<uint32_t>());
                if (c->type()->is_int32()) return static_cast<int64_t>(c->as<int32_t>());
            }
        }
    }
    // Case 2: idx_a = idx_b + constant (negative offset)
    if (idx_a->isa<ArithmeticInst>()) {
        auto *arith = static_cast<ArithmeticInst *>(idx_a);
        if (arith->op() == ArithmeticOp::BINARY_ADD) {
            if (arith->operand(0) == idx_b && arith->operand(1)->isa<Constant>()) {
                auto *c = static_cast<Constant *>(arith->operand(1));
                if (c->type()->is_uint32()) return -static_cast<int64_t>(c->as<uint32_t>());
                if (c->type()->is_int32()) return -static_cast<int64_t>(c->as<int32_t>());
            }
            if (arith->operand(1) == idx_b && arith->operand(0)->isa<Constant>()) {
                auto *c = static_cast<Constant *>(arith->operand(0));
                if (c->type()->is_uint32()) return -static_cast<int64_t>(c->as<uint32_t>());
                if (c->type()->is_int32()) return -static_cast<int64_t>(c->as<int32_t>());
            }
        }
    }
    return INT64_MAX;
}

// Check if a scalar type is eligible for fusion (float, int, uint).
[[nodiscard]] bool is_fusable_scalar(const Type *type) noexcept {
    return type->is_scalar() &&
           (type->is_float() || type->is_int32() || type->is_uint32());
}

// Fuse consecutive buffer reads into a single vector read + extracts.
template<int MaxGroupSize>
void fuse_buffer_reads_in_block(BasicBlock *bb, FunctionDefinition *def,
                                FuseConsecutiveBufferReadsInfo &info) noexcept {
    XIRBuilder builder;
    // Collect all buffer reads in this block, grouped by buffer
    luisa::unordered_map<Value *, luisa::vector<ResourceReadInst *>> read_groups;
    luisa::unordered_map<Value *, luisa::vector<Value *>> index_groups;
    for (auto *inst : bb->instructions()) {
        if (inst->derived_instruction_tag() != DerivedInstructionTag::RESOURCE_READ) continue;
        auto *read = static_cast<ResourceReadInst *>(inst);
        if (read->op() != ResourceReadOp::BUFFER_READ) continue;
        if (!is_fusable_scalar(read->type())) continue;
        auto *buffer = read->operand(0);
        auto *index = read->operand(1);
        read_groups[buffer].push_back(read);
        index_groups[buffer].push_back(index);
    }

    for (auto &[buffer, reads] : read_groups) {
        auto &indices = index_groups[buffer];
        auto n = reads.size();
        if (n < 2u) continue;

        for (size_t i = 0; i < n; ++i) {
            if (reads[i] == nullptr) continue;
            // Try to find reads at offsets +1, +2, ..., +(MaxGroupSize-1)
            int match[4] = {-1, -1, -1, -1};
            match[0] = static_cast<int>(i);
            int found = 1;
            for (size_t j = 0; j < n && found < MaxGroupSize; ++j) {
                if (j == i || reads[j] == nullptr) continue;
                auto offset = constant_offset_between(indices[i], indices[j]);
                if (offset >= 1 && offset < MaxGroupSize) {
                    if (match[offset] == -1) {
                        match[offset] = static_cast<int>(j);
                        found++;
                    }
                }
            }

            if (found >= 2) {
                // We have 'found' consecutive reads
                auto *base_idx = indices[match[0]];
                auto *elem_type = reads[match[0]]->type();
                auto *vec_type = Type::vector(elem_type, found);

                // Collect valid reads
                ResourceReadInst *fused_reads[4] = {};
                bool all_valid = true;
                for (int k = 0; k < found; ++k) {
                    fused_reads[k] = reads[match[k]];
                    if (!fused_reads[k]->parent_block()) { all_valid = false; break; }
                }
                if (!all_valid) continue;

                // Create vector read
                builder.set_insertion_point(fused_reads[0]);
                auto *vec_read = builder.call(vec_type, ResourceReadOp::BUFFER_READ,
                                              {buffer, base_idx});

                // Create extracts and replace
                auto *m = def->parent_module();
                for (int k = 0; k < found; ++k) {
                    builder.set_insertion_point(fused_reads[k]);
                    auto idx_val = static_cast<uint32_t>(k);
                    auto *idx_const = m->create_constant(Type::of<uint32_t>(), &idx_val);
                    auto *extract = builder.call(elem_type, ArithmeticOp::EXTRACT, {vec_read, idx_const});
                    fused_reads[k]->replace_all_uses_with(extract);
                    fused_reads[k]->remove_self();
                    reads[match[k]] = nullptr;
                }

                info.fused_group_count++;
                info.fused_read_count += found;
            }
        }
    }
}

// Fuse consecutive buffer writes into a single vector write.
template<int MaxGroupSize>
void fuse_buffer_writes_in_block(BasicBlock *bb, FuseConsecutiveBufferReadsInfo &info) noexcept {
    XIRBuilder builder;
    // Collect all buffer writes in this block, grouped by buffer
    luisa::unordered_map<Value *, luisa::vector<ResourceWriteInst *>> write_groups;
    luisa::unordered_map<Value *, luisa::vector<Value *>> index_groups;
    luisa::unordered_map<Value *, luisa::vector<Value *>> value_groups;
    for (auto *inst : bb->instructions()) {
        if (inst->derived_instruction_tag() != DerivedInstructionTag::RESOURCE_WRITE) continue;
        auto *write = static_cast<ResourceWriteInst *>(inst);
        if (write->op() != ResourceWriteOp::BUFFER_WRITE) continue;
        auto *value = write->operand(2);  // (buffer, index, value)
        if (!is_fusable_scalar(value->type())) continue;
        auto *buffer = write->operand(0);
        auto *index = write->operand(1);
        write_groups[buffer].push_back(write);
        index_groups[buffer].push_back(index);
        value_groups[buffer].push_back(value);
    }

    for (auto &[buffer, writes] : write_groups) {
        auto &indices = index_groups[buffer];
        auto &values = value_groups[buffer];
        auto n = writes.size();
        if (n < 2u) continue;

        for (size_t i = 0; i < n; ++i) {
            if (writes[i] == nullptr) continue;
            int match[4] = {-1, -1, -1, -1};
            match[0] = static_cast<int>(i);
            int found = 1;
            for (size_t j = 0; j < n && found < MaxGroupSize; ++j) {
                if (j == i || writes[j] == nullptr) continue;
                auto offset = constant_offset_between(indices[i], indices[j]);
                if (offset >= 1 && offset < MaxGroupSize) {
                    if (match[offset] == -1) {
                        match[offset] = static_cast<int>(j);
                        found++;
                    }
                }
            }

            if (found >= 2) {
                ResourceWriteInst *fused_writes[4] = {};
                bool all_valid = true;
                for (int k = 0; k < found; ++k) {
                    fused_writes[k] = writes[match[k]];
                    if (!fused_writes[k]->parent_block()) { all_valid = false; break; }
                }
                if (!all_valid) continue;

                // Construct the vector value
                auto *elem_type = values[match[0]]->type();
                auto *vec_type = Type::vector(elem_type, found);
                auto *m = bb->parent_function()->parent_module();

                builder.set_insertion_point(fused_writes[0]);
                auto *vec_undef = m->create_undefined(vec_type);
                Value *vec_val = vec_undef;
                for (int k = 0; k < found; ++k) {
                    auto idx_val = static_cast<uint32_t>(k);
                    auto *idx_const = m->create_constant(Type::of<uint32_t>(), &idx_val);
                    vec_val = builder.call(vec_type, ArithmeticOp::INSERT,
                                           {vec_val, values[match[k]], idx_const});
                }

                // Create vector write
                auto *base_idx = indices[match[0]];
                builder.call(ResourceWriteOp::BUFFER_WRITE,
                            {buffer, base_idx, vec_val});

                // Remove original writes
                for (int k = 0; k < found; ++k) {
                    fused_writes[k]->remove_self();
                    writes[match[k]] = nullptr;
                }

                info.fused_group_count++;
                info.fused_read_count += found;  // reused counter for fused operations
            }
        }
    }
}

void fuse_consecutive_buffer_reads_on_function(FunctionDefinition *def,
                                                FuseConsecutiveBufferReadsInfo &info) noexcept {
    for (auto *bb : def->basic_blocks()) {
        // Try groups of 4, 3, then 2 (larger groups first for better throughput)
        fuse_buffer_reads_in_block<4>(bb, def, info);
        fuse_buffer_reads_in_block<3>(bb, def, info);
        fuse_buffer_reads_in_block<2>(bb, def, info);
        // Also fuse writes
        fuse_buffer_writes_in_block<4>(bb, info);
        fuse_buffer_writes_in_block<3>(bb, info);
        fuse_buffer_writes_in_block<2>(bb, info);
    }
}

}// namespace

FuseConsecutiveBufferReadsInfo fuse_consecutive_buffer_reads_pass_run_on_function(
    Function *function) noexcept {
    FuseConsecutiveBufferReadsInfo info;
    if (auto def = function->definition()) {
        fuse_consecutive_buffer_reads_on_function(def, info);
    }
    return info;
}

FuseConsecutiveBufferReadsInfo fuse_consecutive_buffer_reads_pass_run_on_module(
    Module *module, PassReport *report) noexcept {
    FuseConsecutiveBufferReadsInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            fuse_consecutive_buffer_reads_on_function(def, info);
        }
    }
    if (report != nullptr) {
        report->set("fused_group_count", info.fused_group_count);
        report->set("fused_read_count", info.fused_read_count);
    }
    return info;
}

}// namespace luisa::compute::xir
