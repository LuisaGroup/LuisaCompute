#include "ast_xir_cache_clone.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

namespace {

void clone_metadata(const Value *source, Value *target) noexcept {
    luisa::vector<const Metadata *> metadata;
    for (auto *md : source->metadata_list()) { metadata.emplace_back(md); }
    // Metadata lists support only push_front; reverse insertion preserves order.
    for (auto iter = metadata.rbegin(); iter != metadata.rend(); ++iter) {
        target->metadata_list().push_front((*iter)->clone());
    }
}

}// namespace

FunctionDefinition *AstXirCacheCloner::import(const FunctionDefinition *source) noexcept {
    LUISA_ASSERT(source != nullptr, "Cannot import a null XIR function definition.");
    return _import_function(source)->definition();
}

Function *AstXirCacheCloner::_import_function(const Function *source) noexcept {
    LUISA_ASSERT(source->parent_module() != &_target, "Cannot import an XIR function into its own module.");
    if (auto iter = _values.find(source); iter != _values.end()) {
        return static_cast<Function *>(iter->second);
    }

    Function *target = nullptr;
    switch (source->derived_function_tag()) {
        case DerivedFunctionTag::KERNEL: {
            auto *kernel = _target.create_kernel();
            kernel->set_block_size(static_cast<const KernelFunction *>(source)->block_size());
            target = kernel;
            break;
        }
        case DerivedFunctionTag::CALLABLE:
            target = _target.create_callable(source->type());
            break;
        case DerivedFunctionTag::EXTERNAL:
            target = _target.create_external_function(source->type());
            break;
        case DerivedFunctionTag::RASTER_STAGE:
            target = _target.create_raster_stage(
                source->type(), static_cast<const RasterStageFunction *>(source)->stage());
            break;
    }
    LUISA_ASSERT(target != nullptr, "Unsupported XIR cache function kind.");

    // Publish the skeleton before cloning any body, including recursive calls.
    _values.emplace(source, target);
    clone_metadata(source, target);
    for (auto *arg : source->arguments()) {
        Argument *cloned = nullptr;
        switch (arg->derived_argument_tag()) {
            case DerivedArgumentTag::VALUE: cloned = target->create_value_argument(arg->type()); break;
            case DerivedArgumentTag::REFERENCE: cloned = target->create_reference_argument(arg->type()); break;
            case DerivedArgumentTag::RESOURCE: cloned = target->create_resource_argument(arg->type()); break;
        }
        LUISA_ASSERT(cloned != nullptr, "Unsupported XIR cache function argument kind.");
        _values.emplace(arg, cloned);
        clone_metadata(arg, cloned);
    }
    if (auto *definition = source->definition()) {
        _clone_body(definition, target->definition());
    }
    return target;
}

void AstXirCacheCloner::_clone_body(const FunctionDefinition *source, FunctionDefinition *target) noexcept {
    struct PendingBlock {
        const Instruction *next;
        BasicBlock *target;
    };
    luisa::vector<PendingBlock> blocks;
    blocks.reserve(source->basic_blocks().count_size());
    for (auto *block : source->basic_blocks()) {
        auto *cloned = target->create_basic_block();
        _values.emplace(block, cloned);
        clone_metadata(block, cloned);
        blocks.emplace_back(PendingBlock{block->instructions().head_sentinel()->next(), cloned});
        for (auto *instruction : block->instructions()) {
            LUISA_ASSERT(instruction->derived_instruction_tag() != DerivedInstructionTag::PHI,
                         "Cannot import PHI instructions into the AST-to-XIR cache.");
        }
    }
    LUISA_ASSERT(source->body_block() != nullptr, "Cannot import an XIR definition without an entry block.");
    target->set_body_block(static_cast<BasicBlock *>(resolve(source->body_block())));

    // Preserve instruction order per block, but do not assume the block list is
    // ordered by dominance. After excluding PHIs the instruction dependencies
    // of valid AST-to-XIR output are acyclic.
    XIRBuilder builder;
    for (;;) {
        auto blocked = false;
        auto progressed = false;
        for (auto &block : blocks) {
            while (!block.next->is_tail_sentinel()) {
                auto *inst = block.next;
                auto ready = true;
                for (auto *use : inst->operand_uses()) {
                    if (auto *operand = use->value(); operand != nullptr && _try_resolve(operand) == nullptr) {
                        ready = false;
                        break;
                    }
                }
                if (!ready) {
                    blocked = true;
                    break;
                }
                builder.set_insertion_point(block.target);
                auto *cloned = inst->clone(builder, *this);
                LUISA_ASSERT(cloned != nullptr, "Failed to import an XIR instruction.");
                _values.emplace(inst, cloned);
                clone_metadata(inst, cloned);
                block.next = inst->next();
                progressed = true;
            }
        }
        if (!blocked) { break; }
        LUISA_ASSERT(progressed, "Unresolved or cyclic instruction dependencies in an XIR cache import.");
    }
}

Value *AstXirCacheCloner::_try_resolve(const Value *value) noexcept {
    if (value == nullptr) { return nullptr; }
    if (auto iter = _values.find(value); iter != _values.end()) { return iter->second; }

    Value *cloned = nullptr;
    switch (value->derived_value_tag()) {
        case DerivedValueTag::FUNCTION:
            return _import_function(static_cast<const Function *>(value));
        case DerivedValueTag::CONSTANT:
            cloned = _target.create_constant(value->type(), static_cast<const Constant *>(value)->data());
            break;
        case DerivedValueTag::UNDEFINED:
            cloned = _target.create_undefined(value->type());
            break;
        case DerivedValueTag::SPECIAL_REGISTER:
            cloned = _target.create_special_register(static_cast<const SpecialRegister *>(value)->derived_special_register_tag());
            break;
        case DerivedValueTag::ARGUMENT:
        case DerivedValueTag::BASIC_BLOCK:
        case DerivedValueTag::INSTRUCTION:
            return nullptr;
    }
    LUISA_ASSERT(cloned != nullptr, "Unsupported value in an XIR cache import.");
    _values.emplace(value, cloned);
    clone_metadata(value, cloned);
    return cloned;
}

Value *AstXirCacheCloner::resolve(const Value *value) noexcept {
    auto *cloned = _try_resolve(value);
    LUISA_ASSERT(value == nullptr || cloned != nullptr, "Unresolved value in an XIR cache import.");
    return cloned;
}

}// namespace luisa::compute::xir
