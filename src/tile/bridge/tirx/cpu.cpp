#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/attrs.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <luisa/core/stl/unordered_map.h>

#include "execution.h"

#if defined(LUISA_TILE_HAS_ACCELERATE)
#include <Accelerate/Accelerate.h>

LUISA_EXTERN_C LUISA_TILE_TIRX_BRIDGE_API void luisa_tile_accelerate_expf(
    float *output, const float *input, int64_t element_count) noexcept {
    if (output == nullptr || input == nullptr || element_count <= 0 ||
        element_count > std::numeric_limits<int>::max()) { return; }
    auto count = static_cast<int>(element_count);
    vvexpf(output, input, &count);
}

LUISA_EXTERN_C LUISA_TILE_TIRX_BRIDGE_API void luisa_tile_accelerate_reduce_add_f32(
    const float *input, float *output, int64_t element_count) noexcept {
    if (output == nullptr || input == nullptr || element_count <= 0) { return; }
    vDSP_sve(input, 1, output, static_cast<vDSP_Length>(element_count));
}

LUISA_EXTERN_C LUISA_TILE_TIRX_BRIDGE_API void luisa_tile_accelerate_reduce_max_f32(
    const float *input, float *output, int64_t element_count) noexcept {
    if (output == nullptr || input == nullptr || element_count <= 0) { return; }
    vDSP_maxv(input, 1, output, static_cast<vDSP_Length>(element_count));
}

LUISA_EXTERN_C LUISA_TILE_TIRX_BRIDGE_API void luisa_tile_accelerate_reduce_min_f32(
    const float *input, float *output, int64_t element_count) noexcept {
    if (output == nullptr || input == nullptr || element_count <= 0) { return; }
    vDSP_minv(input, 1, output, static_cast<vDSP_Length>(element_count));
}
#endif

namespace luisa::compute::tile::bridge::tirx::detail {

namespace {

using BufferKey = const tvm::tirx::VarNode *;

struct StorageUse {
    uint64_t allocations{0u};
    bool escapes{false};
};

class StorageAudit final : public tvm::tirx::StmtExprVisitor {
private:
    bool _address{false};

protected:
    void VisitBufferDef(const tvm::tirx::BufferVar &buffer, bool allocate) final {
        auto &use = buffers[buffer.get()];
        use.allocations += allocate;
        use.escapes |= !allocate;
        StmtExprVisitor::VisitBufferDef(buffer, allocate);
    }
    void VisitExpr_(const tvm::tirx::VarNode *variable) final { buffers[variable].escapes = true; }
    // Only typed element loads/stores are nonescaping uses. Buffer regions,
    // raw data pointers, aliases, and opaque primitive operands are not.
    void VisitBufferUse(const tvm::tirx::BufferVar &buffer) final { buffers[buffer.get()].escapes = true; }
    void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        buffers[load->buffer.get()].escapes |= _address;
        for (auto &index : load->indices) { VisitExpr(index); }
        if (load->predicate) { VisitExpr(load->predicate.value()); }
    }
    void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        buffers[store->buffer.get()].escapes |= _address;
        VisitExpr(store->value);
        for (auto &index : store->indices) { VisitExpr(index); }
        if (store->predicate) { VisitExpr(store->predicate.value()); }
    }
    void VisitStmt_(const tvm::tirx::AttrStmtNode *attribute) final {
        if (auto node = attribute->node.as<tvm::Expr>()) { VisitExpr(node.value()); }
        if (auto region = attribute->node.as<tvm::tirx::BufferRegion>()) { VisitBufferUse(region.value()->buffer); }
        StmtExprVisitor::VisitStmt_(attribute);
    }
    void VisitExpr_(const tvm::CallNode *call) final {
        if (call->op.same_as(tvm::tirx::builtin::call_extern()) && call->args.size() == 4u) {
            auto callee = call->args[0u].as<tvm::tirx::StringImmNode>();
            if (callee != nullptr &&
                (callee->value == "luisa_tile_accelerate_expf" ||
                 callee->value == "luisa_tile_accelerate_reduce_add_f32" ||
                 callee->value == "luisa_tile_accelerate_reduce_max_f32" ||
                 callee->value == "luisa_tile_accelerate_reduce_min_f32")) {
                // This bridge-owned wrapper is synchronous and neither retains
                // nor aliases its array arguments. Keep those known raw-pointer
                // uses eligible for the ordinary bounded stack planner.
                VisitExpr(call->args[3u]);
                return;
            }
        }
        auto saved = _address;
        _address |= call->op.same_as(tvm::tirx::builtin::address_of());
        StmtExprVisitor::VisitExpr_(call);
        _address = saved;
    }

public:
    luisa::unordered_map<BufferKey, StorageUse> buffers;
};

class StackPlanner final : public tvm::tirx::StmtMutator {
private:
    const StorageAudit &_audit;
    uint64_t _remaining;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        auto result = StmtMutator::VisitStmt_(allocation).as_or_throw<tvm::tirx::AllocBuffer>();
        auto manual = result->annotations.count(manual_memory_annotation) != 0u;
        result.CopyOnWrite()->annotations.erase(manual_memory_annotation);
        if (manual || !result->annotations.empty() || _remaining == 0u) { return result; }
        auto buffer = result->buffer;
        auto &use = _audit.buffers.at(allocation->buffer.get());
        auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
        if (use.allocations != 1u || use.escapes || buffer.scope() != "local" ||
            buffer->shape.size() != 1u || !buffer->strides.empty() || buffer->layout ||
            !buffer->allocated_addr.empty() || offset == nullptr || offset->value != 0) { return result; }
        auto extent = buffer->shape[0u].as<tvm::IntImmNode>();
        auto type = buffer->dtype;
        if (extent == nullptr || extent->value <= 0 || extent->value > std::numeric_limits<int32_t>::max() ||
            type.IsScalableVector() || type.lanes() != 1 || type.bits() == 0 || type.bits() > 64 || type.bits() % 8 != 0) { return result; }
        auto element_bytes = static_cast<uint64_t>(type.bits() / 8u) * type.lanes();
        auto count = static_cast<uint64_t>(extent->value);
        if (element_bytes == 0u || count > _remaining / element_bytes) { return result; }
        auto bytes = count * element_bytes;
        // LLVM's allocator uses at most 16-byte explicit alignment. Charge
        // padding for every occurrence, including unrolled/branch copies;
        // do not assume mutually exclusive lifetimes share a stack slot.
        bytes = (bytes + 15u) / 16u * 16u;
        if (bytes > _remaining) { return result; }
        _remaining -= bytes;
        result.CopyOnWrite()->annotations.Set(tvm::tirx::transform::kDisableLowerTVMBuiltin, tvm::IntImm::Int32(1));
        return result;
    }

public:
    StackPlanner(const StorageAudit &audit, uint32_t budget) noexcept : _audit{audit}, _remaining{budget} {}
};

struct VectorExpMap {
    tvm::tirx::BufferVar output;
    tvm::PrimExpr input;
    const tvm::tirx::BufferStoreNode *store;
    int64_t element_count;
};

struct VectorReductionMap {
    tvm::tirx::BufferLoad source;
    tvm::tirx::BufferVar output;
    int64_t kind;
    int64_t element_count;
};

void flatten_sequence(
    const tvm::tirx::Stmt &statement,
    tvm::ffi::Array<tvm::tirx::Stmt> &result) {
    if (auto sequence = statement.as<tvm::tirx::SeqStmtNode>()) {
        for (auto &child : sequence->seq) { flatten_sequence(child, result); }
    } else {
        result.push_back(statement);
    }
}

[[nodiscard]] bool zero_index(const tvm::ffi::Array<tvm::PrimExpr> &indices) {
    return indices.size() == 1u && indices[0u].as<tvm::IntImmNode>() != nullptr &&
           indices[0u].as<tvm::IntImmNode>()->value == 0;
}

[[nodiscard]] std::optional<VectorReductionMap> match_vector_reduction(
    const tvm::tirx::For &loop) {
    auto contract = loop->annotations.Get(reduction_contract_annotation);
    auto kind = contract ? contract.value().as<tvm::IntImmNode>() : nullptr;
    auto minimum = loop->min.as<tvm::IntImmNode>();
    auto extent = loop->extent.as<tvm::IntImmNode>();
    auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
    if (kind == nullptr ||
        (kind->value != reduction_add_contract && kind->value != reduction_max_contract &&
         kind->value != reduction_min_contract) ||
        loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding ||
        minimum == nullptr || minimum->value != 0 || extent == nullptr || extent->value <= 0 ||
        (loop->step && (step == nullptr || step->value != 1))) { return std::nullopt; }
    tvm::ffi::Array<tvm::tirx::Stmt> statements;
    flatten_sequence(loop->body, statements);
    if (statements.size() != 3u) { return std::nullopt; }
    auto allocation = statements[0u].as<tvm::tirx::AllocBufferNode>();
    auto combine_store = statements[1u].as<tvm::tirx::BufferStoreNode>();
    auto output_store = statements[2u].as<tvm::tirx::BufferStoreNode>();
    if (allocation == nullptr || combine_store == nullptr || output_store == nullptr ||
        combine_store->predicate || output_store->predicate ||
        !combine_store->buffer.same_as(allocation->buffer) ||
        !zero_index(combine_store->indices) || !zero_index(output_store->indices)) {
        return std::nullopt;
    }
    auto forwarded = output_store->value.as<tvm::tirx::BufferLoadNode>();
    if (forwarded == nullptr || forwarded->predicate ||
        !forwarded->buffer.same_as(allocation->buffer) || !zero_index(forwarded->indices)) {
        return std::nullopt;
    }
    tvm::PrimExpr lhs;
    tvm::PrimExpr rhs;
    if (kind->value == reduction_add_contract) {
        auto add = combine_store->value.as<tvm::tirx::AddNode>();
        if (add == nullptr) { return std::nullopt; }
        lhs = add->a;
        rhs = add->b;
    } else if (kind->value == reduction_max_contract) {
        auto maximum = combine_store->value.as<tvm::tirx::MaxNode>();
        if (maximum == nullptr) { return std::nullopt; }
        lhs = maximum->a;
        rhs = maximum->b;
    } else {
        auto minimum_value = combine_store->value.as<tvm::tirx::MinNode>();
        if (minimum_value == nullptr) { return std::nullopt; }
        lhs = minimum_value->a;
        rhs = minimum_value->b;
    }
    auto lhs_load = lhs.as<tvm::tirx::BufferLoadNode>();
    auto rhs_load = rhs.as<tvm::tirx::BufferLoadNode>();
    auto output = output_store->buffer;
    const tvm::tirx::BufferLoadNode *source = nullptr;
    if (lhs_load != nullptr && lhs_load->buffer.same_as(output) && zero_index(lhs_load->indices)) {
        source = rhs_load;
    } else if (rhs_load != nullptr && rhs_load->buffer.same_as(output) && zero_index(rhs_load->indices)) {
        source = lhs_load;
    }
    if (source == nullptr || source->predicate || source->buffer.same_as(output) ||
        source->buffer.same_as(allocation->buffer)) { return std::nullopt; }
    auto input = source->buffer;
    auto offset = input->elem_offset.as<tvm::IntImmNode>();
    if (input->dtype != tvm::PrimType::Float(32) || input->shape.empty() ||
        source->indices.size() != input->shape.size() || !input->strides.empty() ||
        input->layout || !input->allocated_addr.empty() || offset == nullptr || offset->value != 0 ||
        !tvm::ffi::StructuralEqual{}(source->indices.back(), loop->loop_var)) {
        return std::nullopt;
    }
    auto last_extent = input->shape.back().as<tvm::IntImmNode>();
    if (last_extent == nullptr || last_extent->value != extent->value) { return std::nullopt; }
    auto depends_on_reduction = false;
    for (auto i = 0u; i + 1u < source->indices.size(); i++) {
        tvm::tirx::PostOrderVisit(source->indices[i], [&](const tvm::ffi::ObjectRef &node) {
            depends_on_reduction |= node.same_as(loop->loop_var);
        });
    }
    if (depends_on_reduction) { return std::nullopt; }
    return VectorReductionMap{
        tvm::ffi::GetRef<tvm::tirx::BufferLoad>(source),
        std::move(output), kind->value, extent->value};
}

[[nodiscard]] std::optional<VectorExpMap> match_vector_exp_map(
    const tvm::tirx::For &outer) {
    static const auto exp_op = tvm::Op::Get("tirx.exp");
    auto contract = outer->annotations.Get(materialized_exp_annotation);
    auto version = contract ? contract.value().as<tvm::IntImmNode>() : nullptr;
    if (version == nullptr || version->value != 1) { return std::nullopt; }
    tvm::ffi::Array<tvm::tirx::PrimVar> variables;
    tvm::ffi::Array<tvm::PrimExpr> extents;
    auto element_count = int64_t{1};
    auto loop = outer.get();
    const tvm::tirx::BufferStoreNode *store = nullptr;
    while (loop != nullptr) {
        auto minimum = loop->min.as<tvm::IntImmNode>();
        auto extent = loop->extent.as<tvm::IntImmNode>();
        auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
        if (loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding ||
            minimum == nullptr || minimum->value != 0 || extent == nullptr || extent->value <= 0 ||
            (loop->step && (step == nullptr || step->value != 1)) ||
            element_count > std::numeric_limits<int>::max() / extent->value) {
            return std::nullopt;
        }
        variables.push_back(loop->loop_var);
        extents.push_back(loop->extent);
        element_count *= extent->value;
        if (auto child = loop->body.as<tvm::tirx::ForNode>()) {
            loop = child;
            continue;
        }
        store = loop->body.as<tvm::tirx::BufferStoreNode>();
        break;
    }
    if (store == nullptr || store->predicate || store->indices.size() != variables.size()) {
        return std::nullopt;
    }
    auto output = store->buffer;
    auto type = output->dtype;
    auto offset = output->elem_offset.as<tvm::IntImmNode>();
    if (output.scope() != "local" || type != tvm::PrimType::Float(32) ||
        output->shape.size() != variables.size() || !output->strides.empty() ||
        output->layout || !output->allocated_addr.empty() || offset == nullptr || offset->value != 0) {
        return std::nullopt;
    }
    auto equal = tvm::ffi::StructuralEqual{};
    for (auto i = 0u; i < variables.size(); i++) {
        auto shape = output->shape[i].as<tvm::IntImmNode>();
        auto extent = extents[i].as<tvm::IntImmNode>();
        if (shape == nullptr || extent == nullptr || shape->value != extent->value ||
            !equal(store->indices[i], variables[i])) { return std::nullopt; }
    }
    auto call = store->value.as<tvm::CallNode>();
    if (call == nullptr || !call->op.same_as(exp_op) || call->args.size() != 1u ||
        call->ty != tvm::PrimType::Float(32)) { return std::nullopt; }
    auto input = call->args[0u].as<tvm::PrimExpr>();
    if (!input) { return std::nullopt; }
    auto depends_on_output = false;
    tvm::tirx::PostOrderVisit(input.value(), [&](const tvm::ffi::ObjectRef &node) {
        if (auto load = node.as<tvm::tirx::BufferLoadNode>()) {
            depends_on_output |= load->buffer.same_as(output);
        } else if (auto variable = node.as<tvm::tirx::VarNode>()) {
            depends_on_output |= variable == output.get();
        }
    });
    if (depends_on_output) { return std::nullopt; }
    return VectorExpMap{std::move(output), std::move(input.value()), store, element_count};
}

class ExpInputWriter final : public tvm::tirx::StmtMutator {
private:
    const tvm::tirx::BufferStoreNode *_store;
    tvm::tirx::BufferVar _input;
    tvm::PrimExpr _value;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        if (store != _store) { return StmtMutator::VisitStmt_(store); }
        return tvm::tirx::BufferStore{_input, _value, store->indices, {}, store->span};
    }

public:
    ExpInputWriter(const tvm::tirx::BufferStoreNode *store,
                   tvm::tirx::BufferVar input, tvm::PrimExpr value) noexcept
        : _store{store}, _input{std::move(input)}, _value{std::move(value)} {}
};

class VectorMathRealizer final : public tvm::tirx::StmtMutator {
private:
    uint64_t _temporary_index{0u};

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *node) final {
        auto loop = tvm::ffi::GetRef<tvm::tirx::For>(node);
        if (auto reduction = match_vector_reduction(loop)) {
            auto indices = reduction->source->indices;
            indices.Set(indices.size() - 1u, tvm::IntImm::Int64(0));
            auto source = tvm::Call{
                tvm::PointerType{tvm::PrimType::Float(32)},
                tvm::tirx::builtin::address_of(),
                {tvm::tirx::BufferLoad{reduction->source->buffer, std::move(indices)}}};
            auto output = tvm::Call{
                tvm::PointerType{tvm::PrimType::Float(32)},
                tvm::tirx::builtin::address_of(),
                {tvm::tirx::BufferLoad{reduction->output, {tvm::IntImm::Int64(0)}}}};
            auto name = reduction->kind == reduction_add_contract ?
                            "luisa_tile_accelerate_reduce_add_f32" :
                        reduction->kind == reduction_max_contract ?
                            "luisa_tile_accelerate_reduce_max_f32" :
                            "luisa_tile_accelerate_reduce_min_f32";
            return tvm::tirx::Evaluate{tvm::Call{
                tvm::PrimType::Void(), tvm::tirx::builtin::call_extern(),
                {tvm::tirx::StringImm{name}, std::move(source), std::move(output),
                 tvm::IntImm::Int64(reduction->element_count)}}};
        }
        auto matched = match_vector_exp_map(loop);
        if (!matched) { return StmtMutator::VisitStmt_(node); }
        auto input = tvm::tirx::decl_buffer(
            matched->output->shape, tvm::PrimType::Float(32),
            matched->output.name() + "_accelerate_input_" + std::to_string(_temporary_index++),
            "local");
        auto fill = ExpInputWriter{matched->store, input, matched->input}(loop);
        if (auto fill_loop = fill.as<tvm::tirx::For>()) {
            fill_loop.value().CopyOnWrite()->annotations.erase(materialized_exp_annotation);
            fill = fill_loop.value();
        }
        tvm::ffi::Array<tvm::Expr> arguments{
            tvm::tirx::StringImm{"luisa_tile_accelerate_expf"},
            matched->output.data(), input.data(),
            tvm::IntImm::Int64(matched->element_count)};
        auto invoke = tvm::tirx::Evaluate{tvm::Call{
            tvm::PrimType::Void(), tvm::tirx::builtin::call_extern(),
            std::move(arguments)}};
        return tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{
            tvm::tirx::AllocBuffer{std::move(input)}, std::move(fill), std::move(invoke)});
    }
};

// TVMx's current TIRx BufferVar projects its physical pointer through
// data(); the legacy TOPI helper still names the removed Buffer::data field.
// Keep this tiny adapter local until that upstream header is ported.
[[nodiscard]] tvm::Expr pack_buffer(tvm::tirx::BufferVar buffer) {
    auto shape = tvm::Call{
        tvm::PointerType{tvm::PrimType::Int(64)},
        tvm::tirx::builtin::tvm_stack_make_shape(), buffer->shape};
    tvm::Expr strides = tvm::PrimExpr{0};
    if (!buffer->strides.empty()) {
        strides = tvm::Call{
            tvm::PointerType{tvm::PrimType::Int(64)},
            tvm::tirx::builtin::tvm_stack_make_shape(), buffer->strides};
    }
    tvm::ffi::Array<tvm::Expr> arguments{
        buffer.data(), shape, strides,
        tvm::IntImm::Int32(static_cast<int64_t>(buffer->shape.size())),
        tvm::tirx::MakeConst(tvm::PrimType{buffer->dtype}, 0),
        buffer->elem_offset};
    return tvm::Call{
        tvm::PointerType::VoidPointerTy(),
        tvm::tirx::builtin::tvm_stack_make_array(), std::move(arguments)};
}

[[nodiscard]] tvm::PrimExpr call_packed(tvm::ffi::Array<tvm::Expr> arguments) {
    return tvm::Call{
        tvm::PrimType::Int(32), tvm::tirx::builtin::tvm_call_packed(),
        std::move(arguments)}.as_or_throw<tvm::PrimExpr>();
}

[[nodiscard]] int64_t required_positive_attribute(
    const tvm::tirx::PrimFunc &function, const char *name) {
    auto value = function->GetAttr<int64_t>(name);
    if (!value || value.value() <= 0) {
        throw std::runtime_error{std::string{"CPU matrix realization requires positive TIRx attribute '"} + name + "'"};
    }
    return value.value();
}

[[nodiscard]] tvm::tirx::BufferVar checked_matrix_parameter(
    const tvm::tirx::PrimFunc &function, size_t index,
    int64_t rows, int64_t columns) {
    if (index >= function->params.size()) {
        throw std::runtime_error{"CPU matrix realization has an incomplete buffer ABI"};
    }
    auto parameter = function->params[index];
    auto type = parameter->ty.as<tvm::tirx::BufferTypeNode>();
    if (type == nullptr || type->dtype != tvm::PrimType::Float(32) ||
        type->shape.size() != 2u || !type->strides.empty() ||
        type->layout || !type->allocated_addr.empty()) {
        throw std::runtime_error{"CPU CBLAS realization requires compact rank-2 FP32 buffers"};
    }
    auto row_extent = type->shape[0u].as<tvm::IntImmNode>();
    auto column_extent = type->shape[1u].as<tvm::IntImmNode>();
    auto offset = type->elem_offset.as<tvm::IntImmNode>();
    if (row_extent == nullptr || column_extent == nullptr || offset == nullptr ||
        row_extent->value != rows || column_extent->value != columns || offset->value != 0) {
        throw std::runtime_error{"CPU CBLAS realization buffer shape/offset disagrees with its whole-GEMM contract"};
    }
    return tvm::tirx::BufferVar{parameter};
}

}// namespace

tvm::tirx::PrimFunc realize_cpu_whole_gemm(
    tvm::tirx::PrimFunc function, bool noalias) {
    if (!noalias) {
        throw std::runtime_error{"CPU CBLAS realization requires the caller's noalias contract"};
    }
    auto version = function->GetAttr<int64_t>(whole_gemm_contract_annotation);
    if (!version || version.value() != 1) {
        throw std::runtime_error{"CPU CBLAS realization requires a proved whole-GEMM TileIR contract v1"};
    }
    if (function->params.size() != 3u) {
        throw std::runtime_error{"CPU whole-GEMM contract v1 requires exactly A, B, and C parameters"};
    }
    if (!tvm::ffi::Function::GetGlobal("tvm.contrib.cblas.matmul")) {
        throw std::runtime_error{"CPU CBLAS realization requested, but tvm.contrib.cblas.matmul is not registered"};
    }
    auto m = required_positive_attribute(function, whole_gemm_m_annotation);
    auto n = required_positive_attribute(function, whole_gemm_n_annotation);
    auto k = required_positive_attribute(function, whole_gemm_k_annotation);
    auto a = checked_matrix_parameter(function, 0u, m, k);
    auto b = checked_matrix_parameter(function, 1u, k, n);
    auto c = checked_matrix_parameter(function, 2u, m, n);
    tvm::ffi::Array<tvm::Expr> arguments{
        tvm::tirx::StringImm{"tvm.contrib.cblas.matmul"},
        pack_buffer(a),
        pack_buffer(b),
        pack_buffer(c),
        tvm::IntImm::Int32(0),
        tvm::IntImm::Int32(0)};
    function.CopyOnWrite()->body = tvm::tirx::Evaluate{
        call_packed(std::move(arguments))};
    return tvm::WithAttr(
        std::move(function), cpu_matrix_realization_annotation,
        tvm::ffi::String{"cblas"});
}

tvm::tirx::Stmt realize_cpu_vector_math(tvm::tirx::Stmt body) {
#if defined(LUISA_TILE_HAS_ACCELERATE)
    return VectorMathRealizer{}(std::move(body));
#else
    static_cast<void>(body);
    throw std::runtime_error{"Apple Accelerate array-math realization is unavailable in this build"};
#endif
}

tvm::tirx::Stmt plan_cpu_storage(tvm::tirx::Stmt body, uint32_t stack_budget) {
    StorageAudit audit;
    audit(body);
    return StackPlanner{audit, stack_budget}(std::move(body));
}

}// namespace luisa::compute::tile::bridge::tirx::detail
