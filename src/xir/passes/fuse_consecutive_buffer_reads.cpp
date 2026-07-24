#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/passes/fuse_consecutive_buffer_reads.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include <limits>

namespace luisa::compute::xir {

namespace detail {

// Typed BUFFER_READ/BUFFER_WRITE operations must read/write exactly the
// element type declared by the buffer handle: the SPIR-V codegen (and the
// unit contract test) requires the access type to equal the buffer element
// type, so a scalar-to-vector rewrite of typed accesses is not legal without
// a byte-addressed lowering. Byte-addressed buffers, on the other hand,
// already accept an arbitrary plain data type at an arbitrary byte offset,
// so consecutive BYTE_BUFFER_READ/BYTE_BUFFER_WRITE operations can be
// coalesced into a single wider vector transaction within the existing ABI.
namespace {

struct ByteIndex {
    Value *base{nullptr};// the variable part of the index (nullptr for pure constants)
    int64_t offset{0};   // the constant byte offset from the base
};

[[nodiscard]] bool decode_constant_int(const Constant *constant, int64_t &value) noexcept {
    if (constant == nullptr || constant->type() == nullptr) { return false; }
    auto type = constant->type();
    if (type->is_int8()) {
        value = constant->as<int8_t>();
    } else if (type->is_uint8()) {
        value = constant->as<uint8_t>();
    } else if (type->is_int16()) {
        value = constant->as<int16_t>();
    } else if (type->is_uint16()) {
        value = constant->as<uint16_t>();
    } else if (type->is_int32()) {
        value = constant->as<int32_t>();
    } else if (type->is_uint32()) {
        value = constant->as<uint32_t>();
    } else if (type->is_int64()) {
        value = constant->as<int64_t>();
    } else if (type->is_uint64()) {
        auto unsigned_value = constant->as<uint64_t>();
        if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return false; }
        value = static_cast<int64_t>(unsigned_value);
    } else {
        return false;
    }
    return true;
}

// Decompose an index value into (base, constant byte offset). Returns false
// when the index shape is not understood.
[[nodiscard]] bool decompose_byte_index(Value *index, ByteIndex &out) noexcept {
    if (index == nullptr) { return false; }
    if (index->isa<Constant>()) {
        int64_t value = 0;
        if (!decode_constant_int(static_cast<Constant *>(index), value)) { return false; }
        out.base = nullptr;
        out.offset = value;
        return true;
    }
    if (index->isa<ArithmeticInst>()) {
        auto arith = static_cast<ArithmeticInst *>(index);
        if (arith->op() == ArithmeticOp::BINARY_ADD && arith->operand_count() == 2u) {
            for (auto i = 0u; i < 2u; ++i) {
                if (arith->operand(i)->isa<Constant>()) {
                    int64_t value = 0;
                    if (!decode_constant_int(static_cast<Constant *>(arith->operand(i)), value)) { return false; }
                    out.base = arith->operand(1u - i);
                    out.offset = value;
                    return true;
                }
            }
        }
    }
    // Any other shape becomes its own base with a zero constant offset, so
    // e.g. read(base * 16), read(base * 16 + 4) still shares a common base.
    out.base = index;
    out.offset = 0;
    return true;
}

[[nodiscard]] bool is_fusable_access_type(const Type *type) noexcept {
    return type != nullptr && type->is_scalar() && type->is_arithmetic() &&
           !type->is_bool() && type->size() >= 4u;
}

[[nodiscard]] uint32_t max_lane_count(const Type *type) noexcept {
    return type->size() >= 8u ? 2u : 4u;
}

// Whether an instruction may observe or mutate the given byte buffer in a way
// that forbids moving memory operations across it.
[[nodiscard]] bool is_barrier_for(const Instruction *inst, const Value *buffer,
                                  bool reads_are_barriers) noexcept {
    if (inst->isa<ResourceWriteInst>()) {
        auto write = static_cast<const ResourceWriteInst *>(inst);
        if (write->op() == ResourceWriteOp::BUFFER_VOLATILE_WRITE ||
            write->op() == ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE) {
            return true;
        }
        if ((write->op() == ResourceWriteOp::BYTE_BUFFER_WRITE ||
             write->op() == ResourceWriteOp::BUFFER_WRITE) &&
            write->operand(0u) == buffer) {
            return true;
        }
    }
    if (reads_are_barriers && inst->isa<ResourceReadInst>()) {
        auto read = static_cast<const ResourceReadInst *>(inst);
        if (read->op() == ResourceReadOp::BUFFER_VOLATILE_READ ||
            read->op() == ResourceReadOp::BYTE_BUFFER_VOLATILE_READ) {
            return true;
        }
        if ((read->op() == ResourceReadOp::BYTE_BUFFER_READ ||
             read->op() == ResourceReadOp::BUFFER_READ) &&
            read->operand(0u) == buffer) {
            return true;
        }
    }
    if (inst->isa<ResourceReadInst>()) {
        auto read = static_cast<const ResourceReadInst *>(inst);
        if (read->op() == ResourceReadOp::BUFFER_VOLATILE_READ ||
            read->op() == ResourceReadOp::BYTE_BUFFER_VOLATILE_READ) {
            return true;
        }
    }
    if (inst->isa<AtomicInst>()) {
        auto atomic = static_cast<const AtomicInst *>(inst);
        if (atomic->operand_count() != 0u && atomic->operand(0u) == buffer) {
            return true;
        }
    }
    // Calls may capture buffers; stay conservative.
    return inst->isa<CallInst>();
}

struct ReadRun {
    luisa::vector<ResourceReadInst *> reads;
    ByteIndex first_index;
};

[[nodiscard]] bool try_fuse_read_run(Module *module, const ReadRun &run,
                                     FuseConsecutiveBufferReadsInfo &info) noexcept {
    auto lane_count = run.reads.size();
    LUISA_DEBUG_ASSERT(lane_count >= 2u, "Read run too short to fuse.");
    auto *first = run.reads.front();
    auto *elem_type = first->type();
    auto *vector_type = Type::vector(elem_type, lane_count);
    if (vector_type == nullptr) { return false; }
    XIRBuilder builder;
    // Insert the vector read right after the first scalar read, followed by
    // the per-lane extracts. The original scalar reads are erased afterwards.
    builder.set_insertion_point(first);
    auto *buffer = first->operand(0u);
    auto *vector_read = builder.call(
        vector_type, ResourceReadOp::BYTE_BUFFER_READ,
        {buffer, first->operand(1u)});
    for (auto lane = 0u; lane < lane_count; ++lane) {
        auto lane_index = static_cast<uint32_t>(lane);
        auto *index = module->create_constant(Type::of<uint32_t>(), &lane_index);
        auto *extract = builder.call(elem_type, ArithmeticOp::EXTRACT,
                                     {vector_read, index});
        run.reads[lane]->replace_all_uses_with(extract);
    }
    for (auto *read : run.reads) {
        static_cast<void>(read->remove_self());
    }
    info.fused_group_count++;
    info.fused_read_count += lane_count;
    return true;
}

struct WriteRun {
    luisa::vector<ResourceWriteInst *> writes;
    ByteIndex first_index;
};

[[nodiscard]] bool try_fuse_write_run(Module *module, const WriteRun &run,
                                      FuseConsecutiveBufferReadsInfo &info) noexcept {
    auto lane_count = run.writes.size();
    LUISA_DEBUG_ASSERT(lane_count >= 2u, "Write run too short to fuse.");
    auto *first = run.writes.front();
    auto *last = run.writes.back();
    auto *elem_type = first->operand(2u)->type();
    auto *vector_type = Type::vector(elem_type, lane_count);
    if (vector_type == nullptr) { return false; }
    XIRBuilder builder;
    // Insert the aggregate and the vector write immediately before the last
    // scalar write, so every aggregated value is already defined. The
    // original scalar writes are erased afterwards.
    builder.set_insertion_point(last->prev());
    luisa::vector<Value *> values;
    values.reserve(lane_count);
    for (auto *write : run.writes) { values.emplace_back(write->operand(2u)); }
    auto *aggregate = builder.call(vector_type, ArithmeticOp::AGGREGATE, values);
    builder.call(ResourceWriteOp::BYTE_BUFFER_WRITE,
                 {first->operand(0u), first->operand(1u), aggregate});
    for (auto *write : run.writes) {
        static_cast<void>(write->remove_self());
    }
    info.fused_group_count++;
    info.fused_write_count += lane_count;
    return true;
}

void collect_and_fuse(Module *module, BasicBlock *block, FuseConsecutiveBufferReadsInfo &info) noexcept {
    ReadRun read_run;
    WriteRun write_run;
    auto flush_reads = [&]() noexcept {
        if (read_run.reads.size() >= 2u) {
            try_fuse_read_run(module, read_run, info);
        }
        read_run.reads.clear();
    };
    auto flush_writes = [&]() noexcept {
        if (write_run.writes.size() >= 2u) {
            try_fuse_write_run(module, write_run, info);
        }
        write_run.writes.clear();
    };
    // Instructions are collected before mutation: both runs only reference
    // instructions that precede the current scan position, and flushing
    // rewrites exactly those instructions.
    luisa::vector<Instruction *> instructions;
    for (auto *inst : block->instructions()) { instructions.emplace_back(inst); }
    for (auto *inst : instructions) {
        auto handled = false;
        if (inst->isa<ResourceReadInst>()) {
            auto read = static_cast<ResourceReadInst *>(inst);
            if (read->op() == ResourceReadOp::BYTE_BUFFER_READ &&
                is_fusable_access_type(read->type())) {
                handled = true;
                ByteIndex index;
                if (!decompose_byte_index(read->operand(1u), index)) {
                    flush_reads();
                } else if (!read_run.reads.empty() &&
                           (read->operand(0u) != read_run.reads.front()->operand(0u) ||
                            read->type() != read_run.reads.front()->type() ||
                            index.base != read_run.first_index.base ||
                            index.offset != read_run.first_index.offset +
                                                static_cast<int64_t>(read->type()->size() * read_run.reads.size()) ||
                            read_run.reads.size() >= max_lane_count(read->type()))) {
                    flush_reads();
                    read_run.reads.emplace_back(read);
                    read_run.first_index = index;
                } else {
                    if (read_run.reads.empty()) { read_run.first_index = index; }
                    read_run.reads.emplace_back(read);
                }
            }
        } else if (inst->isa<ResourceWriteInst>()) {
            auto write = static_cast<ResourceWriteInst *>(inst);
            if (write->op() == ResourceWriteOp::BYTE_BUFFER_WRITE &&
                is_fusable_access_type(write->operand(2u)->type())) {
                handled = true;
                auto *value_type = write->operand(2u)->type();
                ByteIndex index;
                if (!decompose_byte_index(write->operand(1u), index)) {
                    flush_writes();
                } else if (!write_run.writes.empty() &&
                           (write->operand(0u) != write_run.writes.front()->operand(0u) ||
                            value_type != write_run.writes.front()->operand(2u)->type() ||
                            index.base != write_run.first_index.base ||
                            index.offset != write_run.first_index.offset +
                                                static_cast<int64_t>(value_type->size() * write_run.writes.size()) ||
                            write_run.writes.size() >= max_lane_count(value_type))) {
                    flush_writes();
                    write_run.writes.emplace_back(write);
                    write_run.first_index = index;
                } else {
                    if (write_run.writes.empty()) { write_run.first_index = index; }
                    write_run.writes.emplace_back(write);
                }
            }
        }
        // A read run may span intervening independent reads but not writes to
        // the same buffer; a write run may not span any access to the same
        // buffer. An access that joins one run may still be a barrier for the
        // other, so both runs are checked against every instruction except
        // the one that was just appended to that run.
        auto joined_read_run = handled && !read_run.reads.empty() &&
                               read_run.reads.back() == inst;
        auto joined_write_run = handled && !write_run.writes.empty() &&
                                write_run.writes.back() == inst;
        if (!joined_read_run && !read_run.reads.empty() &&
            is_barrier_for(inst, read_run.reads.front()->operand(0u), false)) {
            flush_reads();
        }
        if (!joined_write_run && !write_run.writes.empty() &&
            is_barrier_for(inst, write_run.writes.front()->operand(0u), true)) {
            flush_writes();
        }
    }
    flush_reads();
    flush_writes();
}

void run_on_function(FunctionDefinition *def, FuseConsecutiveBufferReadsInfo &info) noexcept {
    auto *module = def->parent_module();
    luisa::vector<BasicBlock *> blocks;
    for (auto *block : def->basic_blocks()) { blocks.emplace_back(block); }
    for (auto *block : blocks) { collect_and_fuse(module, block, info); }
}

}// namespace

}// namespace detail

FuseConsecutiveBufferReadsInfo fuse_consecutive_buffer_reads_pass_run_on_function(
    Function *function) noexcept {
    FuseConsecutiveBufferReadsInfo info;
    if (auto def = function->definition()) {
        detail::run_on_function(def, info);
    }
    return info;
}

FuseConsecutiveBufferReadsInfo fuse_consecutive_buffer_reads_pass_run_on_module(
    Module *module, PassReport *report) noexcept {
    FuseConsecutiveBufferReadsInfo info;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            detail::run_on_function(def, info);
        }
    }
    if (report != nullptr) {
        report->set("fused_group_count", info.fused_group_count);
        report->set("fused_read_count", info.fused_read_count);
        report->set("fused_write_count", info.fused_write_count);
    }
    return info;
}

}// namespace luisa::compute::xir
