#include "llvm_schedule_emitter.h"

#include "../schedule/strided_mma_types.h"
#include "../schedule/contiguous_copy_types.h"

namespace luisa::compute::simd::detail {

namespace {

// A private, typed leaf in the same native module, not a runtime callback.
// Its independent builder deliberately does not inherit packet fast-math
// flags. Output vectorization changes no output's scalar reduction order.
class StridedMmaEmitter {
private:
    ::llvm::Module &_module;
    const schedule::StridedMmaMetadata &_descriptor;
    ::llvm::IRBuilder<> _builder;
    ::llvm::Function *_function;
    ::llvm::FixedVectorType *_vector_type;

    template<typename Update>
    [[nodiscard]] ::llvm::Value *_fold(uint64_t count, ::llvm::Value *seed, Update &&update) {
        if (count == 0u) { return seed; }
        auto *entry = _builder.GetInsertBlock();
        auto *header = ::llvm::BasicBlock::Create(_module.getContext(), "mma.fold", _function);
        auto *body = ::llvm::BasicBlock::Create(_module.getContext(), "mma.update", _function);
        auto *done = ::llvm::BasicBlock::Create(_module.getContext(), "mma.fold.done", _function);
        _builder.CreateBr(header);
        _builder.SetInsertPoint(header);
        auto *index = _builder.CreatePHI(_builder.getInt64Ty(), 2u, "mma.k");
        auto *acc = _builder.CreatePHI(seed->getType(), 2u, "mma.acc");
        index->addIncoming(_builder.getInt64(0u), entry);
        acc->addIncoming(seed, entry);
        _builder.CreateCondBr(_builder.CreateICmpULT(index, _builder.getInt64(count)), body, done);
        _builder.SetInsertPoint(body);
        auto *next = update(index, acc);
        auto *next_index = _builder.CreateAdd(index, _builder.getInt64(1u));
        auto *latch = _builder.GetInsertBlock();
        _builder.CreateBr(header);
        index->addIncoming(next_index, latch);
        acc->addIncoming(next, latch);
        _builder.SetInsertPoint(done);
        return acc;
    }

    template<typename Body>
    void _loop(uint64_t count, Body &&body) {
        // The integer token is not a numerical reduction and is eliminated
        // by the normal native-module optimizer.
        static_cast<void>(_fold(count, _builder.getInt64(0u), [&](auto *index, auto *token) {
            body(index);
            return token;
        }));
    }

    [[nodiscard]] ::llvm::Value *_address(uint32_t argument, ::llvm::Value *offset) {
        return _builder.CreateGEP(_builder.getFloatTy(), _function->getArg(argument), offset);
    }

    [[nodiscard]] ::llvm::Value *_load(uint32_t argument, ::llvm::Value *offset, bool vector) {
        auto *type = vector ? static_cast<::llvm::Type *>(_vector_type) : _builder.getFloatTy();
        auto *load = _builder.CreateLoad(type, _address(argument, offset));
        load->setAlignment(::llvm::Align{alignof(float)});
        return load;
    }

    [[nodiscard]] ::llvm::Value *_offset(::llvm::Value *flat, const std::vector<uint64_t> &strides) {
        auto *offset = static_cast<::llvm::Value *>(_builder.getInt64(0u));
        for (auto i = _descriptor.output_extents.size(); i-- != 0u;) {
            auto extent = _descriptor.output_extents[i];
            if (extent > 1u) {
                auto *coordinate = _builder.CreateURem(flat, _builder.getInt64(extent));
                if (strides[i] != 0u) {
                    offset = _builder.CreateAdd(offset, _builder.CreateMul(coordinate, _builder.getInt64(strides[i])));
                }
                flat = _builder.CreateUDiv(flat, _builder.getInt64(extent));
            }
        }
        return offset;
    }

    [[nodiscard]] ::llvm::Value *_contribution_offset(::llvm::Value *offset, ::llvm::Value *k, uint64_t stride) {
        return stride == 0u ? offset : _builder.CreateAdd(offset, _builder.CreateMul(k, _builder.getInt64(stride)));
    }

    void _output(::llvm::Value *flat, bool vector, size_t output_axis) {
        auto &&d = _descriptor;
        auto *lhs_offset = _offset(flat, d.lhs_output_strides);
        auto *rhs_offset = _offset(flat, d.rhs_output_strides);
        auto *seed = _load(2u, flat, vector);
        auto *result = _fold(d.contraction_extent, seed, [&](auto *k, auto *acc) {
            auto load_operand = [&](uint32_t argument, auto *offset, uint64_t stride, bool broadcast) {
                auto *value = _load(argument, _contribution_offset(offset, k, stride), vector && !broadcast);
                return vector && broadcast ? _builder.CreateVectorSplat(d.vector_width, value) : value;
            };
            auto *lhs = load_operand(0u, lhs_offset, d.lhs_contraction_stride, d.lhs_output_strides[output_axis] == 0u);
            auto *rhs = load_operand(1u, rhs_offset, d.rhs_contraction_stride, d.rhs_output_strides[output_axis] == 0u);
            return _builder.CreateFAdd(acc, _builder.CreateFMul(lhs, rhs));
        });
        _builder.CreateStore(result, _address(3u, flat))->setAlignment(::llvm::Align{alignof(float)});
    }

    void _contraction(::llvm::Value *flat) {
        auto &&d = _descriptor;
        auto *lhs_offset = _offset(flat, d.lhs_output_strides);
        auto *rhs_offset = _offset(flat, d.rhs_output_strides);
        auto *acc = _load(2u, flat, false);
        auto chunks = d.contraction_extent / d.vector_width;
        if (chunks != 0u) {
            // Seed each lane with its first actual product, not an inserted
            // +0 term: reassociation alone does not grant no-signed-zeros.
            auto *first = _builder.CreateFMul(_load(0u, lhs_offset, true), _load(1u, rhs_offset, true));
            auto *partial = _fold(chunks - 1u, first, [&](auto *chunk, auto *sum) {
                auto *k = _builder.CreateMul(_builder.CreateAdd(chunk, _builder.getInt64(1u)), _builder.getInt64(d.vector_width));
                auto *lhs = _load(0u, _builder.CreateAdd(lhs_offset, k), true);
                auto *rhs = _load(1u, _builder.CreateAdd(rhs_offset, k), true);
                return _builder.CreateFAdd(sum, _builder.CreateFMul(lhs, rhs));
            });
            // Explicit tree only in the reassociation-permitted mode. FMA,
            // approximate functions, NaN and signed-zero shortcuts remain off.
            std::vector<::llvm::Value *> sums;
            for (auto lane = uint32_t{0u}; lane < d.vector_width; lane++) {
                sums.emplace_back(_builder.CreateExtractElement(partial, lane));
            }
            while (sums.size() > 1u) {
                for (auto i = size_t{0u}; i < sums.size() / 2u; i++) {
                    sums[i] = _builder.CreateFAdd(sums[2u * i], sums[2u * i + 1u]);
                }
                sums.resize(sums.size() / 2u);
            }
            acc = _builder.CreateFAdd(acc, sums.front());
        }
        acc = _fold(d.contraction_extent % d.vector_width, acc, [&](auto *tail, auto *sum) {
            auto *k = _builder.CreateAdd(tail, _builder.getInt64(chunks * d.vector_width));
            auto *lhs = _load(0u, _builder.CreateAdd(lhs_offset, k), false);
            auto *rhs = _load(1u, _builder.CreateAdd(rhs_offset, k), false);
            return _builder.CreateFAdd(sum, _builder.CreateFMul(lhs, rhs));
        });
        _builder.CreateStore(acc, _address(3u, flat))->setAlignment(::llvm::Align{alignof(float)});
    }

public:
    StridedMmaEmitter(::llvm::Module &module, const schedule::StridedMmaMetadata &descriptor, std::string name)
        : _module{module}, _descriptor{descriptor}, _builder{module.getContext()},
          _function{::llvm::Function::Create(
              ::llvm::FunctionType::get(_builder.getVoidTy(),
                                        {_builder.getPtrTy(), _builder.getPtrTy(), _builder.getPtrTy(), _builder.getPtrTy()}, false),
              ::llvm::GlobalValue::PrivateLinkage, name, module)},
          _vector_type{::llvm::FixedVectorType::get(_builder.getFloatTy(), descriptor.vector_width)} {
        _builder.clearFastMathFlags();
        _function->addFnAttr(::llvm::Attribute::NoUnwind);
        _function->addFnAttr(::llvm::Attribute::WillReturn);
        _function->addFnAttr("fp-contract", "off");
        _builder.SetInsertPoint(::llvm::BasicBlock::Create(module.getContext(), "entry", _function));
    }

    [[nodiscard]] ::llvm::Function *emit() {
        auto &&d = _descriptor;
        auto volume = uint64_t{1u};
        auto axis = size_t{0u};
        for (auto i = size_t{0u}; i < d.output_extents.size(); i++) {
            volume *= d.output_extents[i];
            if (d.output_extents[i] > 1u) { axis = i; }
        }
        if (d.vectorization == schedule::StridedMmaVectorization::contraction) {
            _loop(volume, [&](auto *flat) { _contraction(flat); });
        } else {
            auto extent = d.output_extents[axis];
            _loop(volume / extent, [&](auto *row) {
                auto *base = _builder.CreateMul(row, _builder.getInt64(extent));
                _loop(extent / d.vector_width, [&](auto *block) {
                    auto *flat = _builder.CreateAdd(base, _builder.CreateMul(block, _builder.getInt64(d.vector_width)));
                    _output(flat, true, axis);
                });
                _loop(extent % d.vector_width, [&](auto *tail) {
                    auto *flat = _builder.CreateAdd(base, _builder.CreateAdd(tail, _builder.getInt64(extent / d.vector_width * d.vector_width)));
                    _output(flat, false, axis);
                });
            });
        }
        _builder.CreateRetVoid();
        return _function;
    }
};

// This helper copies representation bits only. Integer-vector loads/stores
// also preserve signaling NaNs, payloads, infinities and signed zero under
// the caller's numerical policy. No resource alias or read-only annotation
// is exported: the destination is independently proven compiler-local.
[[nodiscard]] ::llvm::Function *emit_contiguous_copy(
    ::llvm::Module &module, const schedule::ContiguousCopyMetadata &descriptor,
    const std::string &name) {
    ::llvm::IRBuilder<> builder{module.getContext()};
    auto *function = ::llvm::Function::Create(
        ::llvm::FunctionType::get(builder.getVoidTy(),
                                  {builder.getPtrTy(), builder.getInt64Ty(), builder.getPtrTy()}, false),
        ::llvm::GlobalValue::PrivateLinkage, name, module);
    function->addFnAttr(::llvm::Attribute::NoUnwind);
    function->addFnAttr(::llvm::Attribute::WillReturn);
    auto *entry = ::llvm::BasicBlock::Create(module.getContext(), "entry", function);
    builder.SetInsertPoint(entry);
    // Like the existing non-volatile BUFFER_READ, the live source interval
    // must be valid. Logical view bounds are guarded by the Tile producer;
    // this helper must not invent zero-fill or silently leave a snapshot
    // uninitialized on an invalid resource binding.
    auto *offset = function->getArg(1u);
    auto *source = builder.CreateGEP(builder.getInt32Ty(), function->getArg(0u), offset);
    auto *destination = function->getArg(2u);
    auto emit_loop = [&](uint64_t count, uint64_t first, uint32_t step, ::llvm::Type *type) {
        if (count == 0u) { return; }
        auto *preheader = builder.GetInsertBlock();
        auto *header = ::llvm::BasicBlock::Create(module.getContext(), "copy.loop", function);
        auto *body = ::llvm::BasicBlock::Create(module.getContext(), "copy.chunk", function);
        auto *next = ::llvm::BasicBlock::Create(module.getContext(), "copy.next", function);
        builder.CreateBr(header);
        builder.SetInsertPoint(header);
        auto *index = builder.CreatePHI(builder.getInt64Ty(), 2u, "copy.index");
        index->addIncoming(builder.getInt64(0u), preheader);
        builder.CreateCondBr(builder.CreateICmpULT(index, builder.getInt64(count)), body, next);
        builder.SetInsertPoint(body);
        auto *element = builder.CreateAdd(builder.getInt64(first), builder.CreateMul(index, builder.getInt64(step)));
        auto *source_address = builder.CreateGEP(builder.getInt32Ty(), source, element);
        auto *destination_address = builder.CreateGEP(builder.getInt32Ty(), destination, element);
        auto *value = builder.CreateLoad(type, source_address);
        value->setAlignment(::llvm::Align{alignof(float)});
        builder.CreateStore(value, destination_address)->setAlignment(::llvm::Align{alignof(float)});
        auto *increment = builder.CreateAdd(index, builder.getInt64(1u));
        builder.CreateBr(header);
        index->addIncoming(increment, body);
        builder.SetInsertPoint(next);
    };
    auto width = descriptor.vector_width;
    auto chunks = descriptor.element_count / width;
    emit_loop(chunks, 0u, width, ::llvm::FixedVectorType::get(builder.getInt32Ty(), width));
    emit_loop(descriptor.element_count % width, chunks * width, 1u, builder.getInt32Ty());
    builder.CreateRetVoid();
    return function;
}

}// namespace

void ScheduleEmitter::_preflight_typed_calls() {
    std::vector<uint8_t> root_local(_source.values().size(), uint8_t{0u});
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode == schedule::Opcode::alloca && instruction.result &&
                instruction.source_op == static_cast<uint32_t>(xir::AllocaOp::LOCAL)) {
                root_local[instruction.result->value] = 1u;
            }
        }
    }
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode != schedule::Opcode::call) {
                if (instruction.strided_mma || instruction.contiguous_copy) {
                    _fail("typed call metadata requires a call instruction");
                    return;
                }
                continue;
            }
            if (instruction.strided_mma && instruction.contiguous_copy) {
                _fail("typed call requires exactly one strided MMA or contiguous copy descriptor");
                return;
            }
            if (instruction.contiguous_copy) {
                if (instruction.result || instruction.operands.size() != 3u) {
                    _fail("contiguous copy requires a void descriptor and three operands");
                    return;
                }
                std::array<const Type *, 3u> types{};
                for (auto i = size_t{0u}; i < types.size(); i++) {
                    auto *value = _source.value(instruction.operands[i]);
                    if (value == nullptr) {
                        _fail("contiguous copy has an invalid operand");
                        return;
                    }
                    types[i] = value->type;
                }
                auto *resource = _source.value(instruction.operands[0u]);
                auto *offset = _source.value(instruction.operands[1u]);
                auto *parameter = std::get_if<schedule::ParameterValueMetadata>(&resource->metadata);
                auto destination = instruction.operands[2u];
                if (resource->origin != schedule::ValueOrigin::parameter || parameter == nullptr ||
                    parameter->argument_tag != static_cast<uint32_t>(xir::DerivedArgumentTag::RESOURCE) ||
                    resource->value_class != schedule::ValueClass::warp_uniform ||
                    (!schedule::is_uniform(offset->value_class) && offset->value_class != schedule::ValueClass::varying) ||
                    _is_local_lvalue(instruction.operands[0u]) || _is_local_lvalue(instruction.operands[1u]) ||
                    !root_local[destination.value] || !_is_local_lvalue(destination) || _is_shared_lvalue(destination)) {
                    _fail("contiguous copy requires a resource parameter, value offset and complete root thread-local destination");
                    return;
                }
                if (auto error = schedule::validate_contiguous_copy(*instruction.contiguous_copy, types); !error.empty()) {
                    _fail(std::string{error});
                    return;
                }
                continue;
            }
            if (!instruction.strided_mma || instruction.result || instruction.operands.size() != 4u) {
                _fail("SIMD call requires a void strided MMA descriptor and four references");
                return;
            }
            std::array<const Type *, 4u> types{};
            for (auto i = size_t{0u}; i < types.size(); i++) {
                auto id = instruction.operands[i];
                auto *value = _source.value(id);
                if (value == nullptr || !root_local[id.value] || !_is_local_lvalue(id) || _is_shared_lvalue(id)) {
                    _fail("strided MMA requires complete root thread-local allocations");
                    return;
                }
                if (i < 3u && id == instruction.operands[3u]) {
                    _fail("strided MMA output must not alias an input snapshot");
                    return;
                }
                types[i] = value->type;
            }
            if (auto error = schedule::validate_strided_mma(*instruction.strided_mma, types); !error.empty()) {
                _fail(std::string{error});
                return;
            }
        }
    }
}

void ScheduleEmitter::_strided_mma(const schedule::Instruction &instruction) {
    std::array<::llvm::Value *, 4u> handles{};
    for (auto i = size_t{0u}; i < handles.size(); i++) {
        auto id = instruction.operands[i];
        if (_interleaved_local_values[id.value] != 0u) {
            _fail("strided MMA references cannot use packet-interleaved private arrays");
            return;
        }
        handles[i] = _load_value(id);
        if (handles[i] == nullptr) { return; }
    }
    auto *helper = StridedMmaEmitter{_module, *instruction.strided_mma, _entry_name + ".strided_mma"}.emit();
    // One call for each active program lane. The helper's vector dimension is
    // independent of the packet width, with no cross-program communication.
    for (auto lane = uint32_t{0u}; lane < _width; lane++) {
        auto *call_block = ::llvm::BasicBlock::Create(_module.getContext(), "mma.active", _entry);
        auto *next = ::llvm::BasicBlock::Create(_module.getContext(), "mma.continue", _entry);
        _builder.CreateCondBr(_builder.CreateExtractElement(_active_mask, lane), call_block, next);
        _builder.SetInsertPoint(call_block);
        std::array<::llvm::Value *, 4u> pointers{};
        for (auto i = size_t{0u}; i < pointers.size(); i++) {
            auto *base = _builder.CreateExtractElement(_local_base(_builder, handles[i]), lane);
            auto *offset = _builder.CreateExtractElement(_local_offsets(_builder, handles[i]), lane);
            pointers[i] = _builder.CreateGEP(_builder.getInt8Ty(), base, offset);
        }
        _builder.CreateCall(helper, pointers);
        _builder.CreateBr(next);
        _builder.SetInsertPoint(next);
    }
}

void ScheduleEmitter::_contiguous_copy(const schedule::Instruction &instruction) {
    auto destination = instruction.operands[2u];
    if (_interleaved_local_values[destination.value] != 0u) {
        _fail("contiguous copy destination cannot use packet-interleaved private arrays");
        return;
    }
    auto *buffer = _load_value(instruction.operands[0u]);
    auto *offsets = _load_value(instruction.operands[1u]);
    auto *handle = _load_value(destination);
    if (buffer == nullptr || offsets == nullptr || handle == nullptr) { return; }
    auto *helper = emit_contiguous_copy(_module, *instruction.contiguous_copy, _entry_name + ".contiguous_copy");
    auto *base = _builder.CreateExtractValue(buffer, {0u});
    // The contiguous dimension belongs to one program's snapshot, not the
    // packet's execution lanes. No inactive program calls or reads the helper.
    for (auto lane = uint32_t{0u}; lane < _width; lane++) {
        auto *call_block = ::llvm::BasicBlock::Create(_module.getContext(), "copy.active", _entry);
        auto *next = ::llvm::BasicBlock::Create(_module.getContext(), "copy.continue", _entry);
        _builder.CreateCondBr(_builder.CreateExtractElement(_active_mask, lane), call_block, next);
        _builder.SetInsertPoint(call_block);
        auto *local_base = _builder.CreateExtractElement(_local_base(_builder, handle), lane);
        auto *local_offset = _builder.CreateExtractElement(_local_offsets(_builder, handle), lane);
        auto *pointer = _builder.CreateGEP(_builder.getInt8Ty(), local_base, local_offset);
        auto *offset = offsets->getType()->isVectorTy() ? _builder.CreateExtractElement(offsets, lane) : offsets;
        _builder.CreateCall(helper, {base, offset, pointer});
        _builder.CreateBr(next);
        _builder.SetInsertPoint(next);
    }
}

}// namespace luisa::compute::simd::detail
