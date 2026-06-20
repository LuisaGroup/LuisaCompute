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

struct BufferReadGroup {
    Value *buffer;
    luisa::vector<ResourceReadInst *> reads;
    luisa::vector<Value *> indices;
};

void fuse_consecutive_buffer_reads_on_function(FunctionDefinition *def,
                                                FuseConsecutiveBufferReadsInfo &info) noexcept {
    XIRBuilder builder;
    
    for (auto *bb : def->basic_blocks()) {
        // Collect all buffer reads in this block, grouped by buffer
        luisa::unordered_map<Value *, BufferReadGroup> groups;
        for (auto *inst : bb->instructions()) {
            if (inst->derived_instruction_tag() != DerivedInstructionTag::RESOURCE_READ) continue;
            auto *read = static_cast<ResourceReadInst *>(inst);
            if (read->op() != ResourceReadOp::BUFFER_READ) continue;
            // Only fuse scalar float reads (most common for dot products)
            if (!read->type()->is_scalar() || !read->type()->is_float()) continue;
            auto *buffer = read->operand(0);
            auto *index = read->operand(1);
            auto &group = groups[buffer];
            group.buffer = buffer;
            group.reads.push_back(read);
            group.indices.push_back(index);
        }
        
        // For each group, try to find runs of 4 consecutive reads
        for (auto &[buffer, group] : groups) {
            if (group.reads.size() < 4) continue;
            
            // Build offset map: for each pair, compute constant offset
            // We want to find a base index and 4 reads at offsets 0,1,2,3
            size_t n = group.reads.size();
            for (size_t i = 0; i < n; ++i) {
                // Try to find reads at offsets +1, +2, +3 from this base
                int match[4] = {-1, -1, -1, -1};
                match[0] = static_cast<int>(i);
                int found = 1;
                for (size_t j = 0; j < n && found < 4; ++j) {
                    if (j == i) continue;
                    auto offset = constant_offset_between(group.indices[i], group.indices[j]);
                    if (offset >= 1 && offset <= 3) {
                        if (match[offset] == -1) {
                            match[offset] = static_cast<int>(j);
                            found++;
                        }
                    }
                }
                
                if (found == 4) {
                    // We have 4 consecutive reads at offsets 0,1,2,3
                    auto *base_idx = group.indices[match[0]];
                    auto *read0 = group.reads[match[0]];
                    auto *read1 = group.reads[match[1]];
                    auto *read2 = group.reads[match[2]];
                    auto *read3 = group.reads[match[3]];
                    
                    // Ensure all reads are still valid (not already replaced)
                    if (!read0->parent_block() || !read1->parent_block() ||
                        !read2->parent_block() || !read3->parent_block()) continue;
                    
                    // Create a vector read: vec4 = resource_read(buffer, base_idx)
                    auto *elem_type = read0->type();
                    auto *vec_type = Type::vector(elem_type, 4);
                    builder.set_insertion_point(read0);
                    auto *vec_read = builder.call(vec_type, ResourceReadOp::BUFFER_READ,
                                                   {buffer, base_idx});
                    
                    // Create extract instructions for each element
                    auto *m = def->parent_module();
                    auto make_extract = [&](Instruction *before, int idx) -> Value * {
                        builder.set_insertion_point(before);
                        auto idx_val = static_cast<uint32_t>(idx);
                        auto *idx_const = m->create_constant(Type::of<uint32_t>(), &idx_val);
                        return builder.call(elem_type, ArithmeticOp::EXTRACT, {vec_read, idx_const});
                    };
                    
                    // Replace each scalar read with an extract from the vector
                    Value *extracts[4];
                    extracts[0] = make_extract(read0, 0);
                    extracts[1] = make_extract(read1, 1);
                    extracts[2] = make_extract(read2, 2);
                    extracts[3] = make_extract(read3, 3);
                    
                    read0->replace_all_uses_with(extracts[0]);
                    read0->remove_self();
                    read1->replace_all_uses_with(extracts[1]);
                    read1->remove_self();
                    read2->replace_all_uses_with(extracts[2]);
                    read2->remove_self();
                    read3->replace_all_uses_with(extracts[3]);
                    read3->remove_self();
                    
                    info.fused_group_count++;
                    info.fused_read_count += 4;
                    
                    // Mark these reads as processed so we don't try to fuse them again
                    group.reads[match[0]] = nullptr;
                    group.reads[match[1]] = nullptr;
                    group.reads[match[2]] = nullptr;
                    group.reads[match[3]] = nullptr;
                }
            }
        }
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
