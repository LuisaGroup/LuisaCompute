#include "cuda_tile_streamed_sum_ir.h"
#include <algorithm>
#include <bit>
#include <limits>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/verifier.h>

namespace luisa::compute::cuda::native_tile {
namespace {
namespace streamed_sum_detail {
using namespace tile;
using tile::Function;
using tile::Type;

bool add(uint64_t &a, uint64_t b) noexcept {
    if (a > std::numeric_limits<uint64_t>::max() - b) { return false; }
    a += b;
    return true;
}
bool multiply(uint64_t a, uint64_t b, uint64_t &out) noexcept {
    if (b != 0u && a > std::numeric_limits<uint64_t>::max() / b) { return false; }
    out = a * b;
    return true;
}
uint64_t volume(const IndexSpace *space) noexcept {
    auto value = space == nullptr ? luisa::optional<uint64_t>{} : space->static_volume();
    return value ? *value : 0u;
}
bool integer_type(const Type &type) noexcept {
    return type.kind() == TypeKind::INDEX || type == Type::scalar(ScalarType::INT64);
}
const Value *strip(const Value *v) noexcept {
    for (auto depth = 0u; depth < 32u; depth++) {
        auto op = v->defining_operation();
        if (op == nullptr || op->kind() != OperationKind::ELEMENTWISE ||
            op->elementwise_op() != ElementwiseOp::CAST || op->operand_count() != 1u ||
            !integer_type(v->type()) || !integer_type(op->operand(0u)->type())) { return v; }
        v = op->operand(0u);
    }
    return nullptr;
}
bool integer(const Value *v, int64_t n) noexcept {
    v = strip(v);
    auto op = v == nullptr ? nullptr : v->defining_operation();
    auto a = op != nullptr && op->kind() == OperationKind::CONSTANT ? op->attribute("value") : nullptr;
    auto x = a == nullptr ? nullptr : luisa::get_if<int64_t>(&a->value());
    return v != nullptr && integer_type(v->type()) && x != nullptr && *x == n;
}
bool row_origin(const Value *v, const Value *row) noexcept {
    v = strip(v);
    if (v == row) { return true; }
    auto op = v == nullptr ? nullptr : v->defining_operation();
    return op != nullptr && op->kind() == OperationKind::ELEMENTWISE &&
           op->elementwise_op() == ElementwiseOp::MUL && op->operand_count() == 2u &&
           ((strip(op->operand(0u)) == row && integer(op->operand(1u), 1)) ||
            (strip(op->operand(1u)) == row && integer(op->operand(0u), 1)));
}
bool storage(ScalarType t) noexcept {
    return t == ScalarType::FLOAT32 || t == ScalarType::FLOAT16 || t == ScalarType::BFLOAT16;
}
const Operation *find(const Block &b, uint64_t id) noexcept {
    for (auto op : b.operations()) {
        if (op->id() == id) { return op; }
        for (auto &&region : op->regions()) {
            for (auto child : region->blocks()) {
                if (auto result = find(*child, id)) { return result; }
            }
        }
    }
    return nullptr;
}

struct Slice {
    const Block *root{}, *program{};
    const Operation *parallel{}, *load{}, *cast{}, *map{}, *sum{}, *store{};
    const Value *source{};
    Dim column;
    uint64_t rows{}, width{}, columns{}, chunk{}, trips{};
    DisjointRequirement disjoint;
    luisa::unordered_set<const Operation *> dead;
};

// Bounded local DCE admission only. No effects, structured execution, unknown
// opcodes or explicit constraints are removed even if their results are unused.
bool discardable(const Operation &op, size_t &budget) noexcept {
    if (++budget > 256u || op.memory_effect() != MemoryEffect::NONE ||
        op.execution_scope_constraint() || op.resource_class_constraint() || op.memory_layout()) { return false; }
    for (auto &&a : op.attributes()) {
        if (op.kind() != OperationKind::CONSTANT || a.name != "value") { return false; }
    }
    switch (op.kind()) {
        case OperationKind::CONSTANT:
        case OperationKind::TILE_EXTRACT:
        case OperationKind::YIELD: break;
        case OperationKind::TILE_MAP:
            if (op.region_count() != 1u || op.region(0u)->block_count() != 1u) { return false; }
            break;
        case OperationKind::ELEMENTWISE:
            switch (op.elementwise_op()) {
                case ElementwiseOp::ADD:
                case ElementwiseOp::SUB:
                case ElementwiseOp::MUL:
                case ElementwiseOp::DIV:
                case ElementwiseOp::MOD:
                case ElementwiseOp::NEG:
                case ElementwiseOp::MIN:
                case ElementwiseOp::MAX:
                case ElementwiseOp::CAST:
                case ElementwiseOp::SELECT:
                case ElementwiseOp::EQ:
                case ElementwiseOp::NE:
                case ElementwiseOp::LT:
                case ElementwiseOp::LE:
                case ElementwiseOp::GT:
                case ElementwiseOp::GE:
                case ElementwiseOp::LOGICAL_AND:
                case ElementwiseOp::LOGICAL_OR:
                case ElementwiseOp::LOGICAL_NOT:
                case ElementwiseOp::EXP:
                case ElementwiseOp::LOG:
                case ElementwiseOp::SQRT:
                case ElementwiseOp::TANH:
                case ElementwiseOp::ABS:
                case ElementwiseOp::BITCAST: break;
                default: return false;
            }
            break;
        default: return false;
    }
    for (auto &&region : op.regions()) {
        if (region->block_count() != 1u) { return false; }
        for (auto child : region->block(0u)->operations()) {
            if (!discardable(*child, budget)) { return false; }
        }
    }
    return true;
}

const Operation *program_owner(const Operation *op, const Block *program) noexcept {
    for (auto depth = 0u; op != nullptr && depth < 32u; depth++) {
        auto block = op->parent_block();
        if (block == program) { return op; }
        op = block == nullptr || block->parent_region() == nullptr ? nullptr : block->parent_region()->parent_operation();
    }
    return nullptr;
}

size_t live_uses(const Value &v, const Slice &s) noexcept {
    size_t count = 0u;
    for (auto use : v.use_list()) {
        auto owner = program_owner(use->user(), s.program);
        if (owner == nullptr || !s.dead.contains(owner)) { count++; }
    }
    return count;
}

bool collect_dead(Slice &s) noexcept {
    luisa::vector<const Operation *> operations;
    for (auto op : s.program->operations()) {
        if (operations.size() == 256u) { return false; }
        operations.emplace_back(op);
    }
    // Verified SSA use edges point to later owner operations. One reverse pass
    // is sufficient; the original use lists/function remain completely intact.
    for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
        auto op = *it;
        if (op->result_count() == 0u) { continue; }
        auto unused = true;
        for (auto i = 0u; i < op->result_count(); i++) { unused &= live_uses(*op->result(i), s) == 0u; }
        if (!unused) { continue; }
        size_t budget = 0u;
        if (discardable(*op, budget)) { s.dead.emplace(op); }
    }
    return true;
}

// Traverse only the already verified, bounded pure dataflow. Region results
// also depend on their YIELD operands; SSA block arguments are otherwise leaves.
bool depends_on(const Value *value, const Value *wanted, size_t &budget) noexcept {
    if (value == wanted) { return true; }
    if (value == nullptr || ++budget > 256u) { return false; }
    auto op = value->defining_operation();
    if (op == nullptr) { return false; }
    for (auto i = 0u; i < op->operand_count(); i++) {
        if (depends_on(op->operand(i), wanted, budget)) { return true; }
    }
    if (op->kind() == OperationKind::TILE_MAP) {
        auto block = op->region(0u)->block(0u);
        auto terminal = block->operation(block->operation_count() - 1u);
        if (terminal->kind() == OperationKind::YIELD && value->index() < terminal->operand_count()) {
            return depends_on(terminal->operand(value->index()), wanted, budget);
        }
    }
    return false;
}

// First slice deliberately supports row-only consumers. No surviving value
// may retain the original full contribution Tile after the SUM is replaced.
bool supported_block(const Block &block, const Slice &s, size_t &budget) noexcept {
    for (auto op : block.operations()) {
        if (s.dead.contains(op)) { continue; }
        if (++budget > 256u || op->execution_scope_constraint() || op->resource_class_constraint() || op->memory_layout()) { return false; }
        for (auto &&a : op->attributes()) {
            if (op->kind() != OperationKind::CONSTANT || a.name != "value") { return false; }
        }
        switch (op->kind()) {
            case OperationKind::CONSTANT:
            case OperationKind::YIELD:
            case OperationKind::TILE_EXTRACT: break;
            case OperationKind::ELEMENTWISE:
                switch (op->elementwise_op()) {
                    case ElementwiseOp::ADD:
                    case ElementwiseOp::SUB:
                    case ElementwiseOp::MUL:
                    case ElementwiseOp::DIV:
                    case ElementwiseOp::NEG:
                    case ElementwiseOp::CAST:
                    case ElementwiseOp::SQRT:
                    case ElementwiseOp::ABS: break;
                    default: return false;
                }
                break;
            case OperationKind::PARALLEL:
                if (op != s.parallel) { return false; }
                break;
            case OperationKind::REDUCE:
                if (op != s.sum) { return false; }
                break;
            case OperationKind::VIEW_LOAD:
                if (op != s.load) { return false; }
                break;
            case OperationKind::VIEW_STORE:
                if (op != s.store) { return false; }
                break;
            case OperationKind::TILE_MAP:
                if (volume(&*op->domain()) != 1u) { return false; }
                break;
            default: return false;
        }
        for (auto i = 0u; i < op->result_count(); i++) {
            auto result = op->result(i);
            if (result->type().is_tile() && op != s.load && op != s.cast && volume(result->type().index_space()) != 1u) { return false; }
        }
        for (auto &&region : op->regions()) {
            if (region->block_count() != 1u || !supported_block(*region->block(0u), s, budget)) { return false; }
        }
    }
    return true;
}

bool match(const Function &f, uint32_t chunk, Slice &s, luisa::string &error) noexcept {
    auto fail = [&](luisa::string_view reason) { error.assign(reason.data(), reason.size()); return false; };
    auto analysis = analyze_collective_work(f);
    if (!analysis.ok() || analysis.collectives.size() != 1u || analysis.collectives.front().kind != CollectiveKind::SUM ||
        analysis.collectives.front().element != ScalarType::FLOAT32) { return fail("requires one admitted unordered FP32 SUM"); }
    s.root = f.body().block(0u);
    for (auto op : s.root->operations()) {
        if (op->kind() == OperationKind::PARALLEL) {
            s.parallel = op;
        } else if (op->kind() != OperationKind::CONSTANT) {
            return fail("root contains nonconstant work outside the program");
        }
    }
    if (s.parallel == nullptr || s.parallel->domain()->rank() != 1u || s.root->argument_count() > 31u) { return fail("requires one rank-one direct-buffer grid"); }
    s.program = s.parallel->region(0u)->block(0u);
    if (!collect_dead(s)) { return fail("pure-dead program analysis budget exceeded"); }
    s.sum = find(*s.program, analysis.collectives.front().operation_id);
    s.map = s.sum == nullptr ? nullptr : s.sum->parent_block()->parent_region()->parent_operation();
    if (s.map == nullptr || s.map->kind() != OperationKind::TILE_MAP || s.map->parent_block() != s.program ||
        s.map->domain()->rank() != 1u || volume(&*s.map->domain()) != 1u || s.map->result_count() != 1u ||
        s.map->result(0u)->type().scalar_type() != ScalarType::FLOAT32) { return fail("SUM must have one direct BR1 FP32 result map"); }
    for (auto op : s.program->operations()) {
        if (op->kind() == OperationKind::VIEW_LOAD) {
            if (s.load != nullptr) { return fail("extra input snapshot"); }
            s.load = op;
        }
        if (op->kind() == OperationKind::VIEW_STORE) {
            if (s.store != nullptr) { return fail("extra output effect"); }
            s.store = op;
        }
    }
    if (s.load == nullptr || s.store == nullptr || !s.load->domain() || !s.store->domain() ||
        s.load->domain()->rank() != 2u || s.load->operand_count() != 3u || s.load->result_count() != 1u ||
        s.load->bounds_mode() != BoundsMode::ZERO || s.store->bounds_mode() != BoundsMode::ZERO ||
        s.store->domain()->rank() < 1u || s.store->domain()->rank() > 2u || volume(&*s.store->domain()) != 1u ||
        s.store->operand_count() != s.store->domain()->rank() + 2u) { return fail("requires a plain rank2 ZERO snapshot and one row store"); }
    auto input = s.load->operand(0u), output = s.store->operand(0u);
    if (input->argument_block() != s.root || output->argument_block() != s.root || input == output ||
        !input->type().is_view() || !output->type().is_view() || !storage(input->type().scalar_type()) || !storage(output->type().scalar_type())) { return fail("requires distinct direct floating storage roots"); }
    for (auto &&arg : s.root->arguments()) {
        if (!arg->type().is_view() || !storage(arg->type().scalar_type()) || volume(arg->type().index_space()) == 0u) { return fail("unsupported or dynamic root ABI"); }
    }
    auto in = input->type().index_space(), out = output->type().index_space();
    if (in->rank() != 2u || out->rank() != s.store->domain()->rank() || volume(in) == 0u || volume(out) == 0u) { return fail("unsupported static view ranks"); }
    s.rows = in->axis(0u).extent.constant_value();
    s.columns = in->axis(1u).extent.constant_value();
    s.width = volume(&*s.load->domain());
    s.column = s.load->domain()->axis(1u).dimension;
    s.chunk = chunk;
    if (s.rows == 0u || s.columns == 0u || s.rows > 0x7fffffffu || s.width < s.columns || !std::has_single_bit(s.width) ||
        (chunk != 1024u && chunk != 2048u) || s.width <= chunk || s.width % chunk != 0u || s.width / chunk > 16u ||
        s.load->domain()->axis(0u).extent != Extent::constant(1u) ||
        s.map->domain()->axis(0u).dimension != s.load->domain()->axis(0u).dimension ||
        s.sum->domain()->rank() != 1u || s.sum->domain()->axis(0u).dimension != s.column ||
        analysis.programs != s.rows || analysis.collectives.front().contribution_extent != s.width ||
        analysis.collectives.front().independent_elements != 1u || out->axis(0u).extent != Extent::constant(s.rows) ||
        (out->rank() == 2u && out->axis(1u).extent != Extent::constant(1u))) { return fail("requires bounded BR1 contiguous row geometry"); }
    s.trips = s.width / chunk;
    if (!row_origin(s.load->operand(1u), s.program->argument(0u)) || !integer(s.load->operand(2u), 0) ||
        !row_origin(s.store->operand(1u), s.program->argument(0u)) ||
        (out->rank() == 2u && !integer(s.store->operand(2u), 0)) ||
        s.store->domain()->axis(0u).dimension != s.map->domain()->axis(0u).dimension) { return fail("unproved row origins or output coverage"); }
    const Operation *extract = nullptr;
    for (auto op : s.sum->region(0u)->block(0u)->operations()) {
        if (op->kind() == OperationKind::TILE_EXTRACT) { extract = op; }
    }
    if (extract == nullptr || extract->operand_count() != 3u ||
        extract->operand(2u) != s.sum->region(0u)->block(0u)->argument(0u)) { return fail("SUM does not reduce the final source axis"); }
    s.source = extract->operand(0u);
    if (s.source != s.load->result(0u)) {
        s.cast = s.source->defining_operation();
        if (s.cast == nullptr || s.cast->parent_block() != s.program || s.cast->kind() != OperationKind::ELEMENTWISE ||
            s.cast->elementwise_op() != ElementwiseOp::CAST || s.cast->operand_count() != 1u ||
            s.cast->operand(0u) != s.load->result(0u) || s.source->type().scalar_type() != ScalarType::FLOAT32 ||
            *s.source->type().index_space() != *s.load->result(0u)->type().index_space()) { return fail("only an exact storage-to-FP32 producer cast is admitted"); }
    }
    if (live_uses(*s.load->result(0u), s) != 1u || live_uses(*s.source, s) != 1u) { return fail("full snapshot escapes the single SUM"); }
    uint64_t input_bytes{}, output_bytes{}, envelope{};
    constexpr auto limit = uint64_t{std::numeric_limits<int64_t>::max()};
    if (!multiply(volume(in), scalar_type_size(input->type().scalar_type()), input_bytes) || input_bytes > limit ||
        !multiply(volume(out), scalar_type_size(output->type().scalar_type()), output_bytes) || output_bytes > limit ||
        !multiply(s.rows - 1u, s.columns, envelope) || !add(envelope, s.width - 1u) || envelope > limit) { return fail("index or static byte range overflow"); }
    s.disjoint = {{static_cast<uint32_t>(input->index()), 0u, input_bytes}, {static_cast<uint32_t>(output->index()), 0u, output_bytes}};
    size_t budget = 0u;
    if (!supported_block(*s.root, s, budget)) { return fail("non-row epilogue, unsupported opcode/attribute or size budget"); }
    budget = 0u;
    if (!depends_on(s.map->result(0u), s.sum->result(0u), budget)) { return fail("SUM does not reach the row map result"); }
    budget = 0u;
    if (!depends_on(s.store->operand(s.store->operand_count() - 1u), s.map->result(0u), budget)) { return fail("SUM map does not reach the output store"); }
    return true;
}

class Clone {
private:
    const Slice &_s;
    Module &_module;
    luisa::vector<std::pair<Dim, Dim>> _dims;
    luisa::unordered_map<const Value *, Value *> _values;
    bool _ok{true};
    Dim dim(Dim old) noexcept {
        for (auto [a, b] : _dims) {
            if (a == old) { return b; }
        }
        auto created = _module.dimensions().create_dimension(old.context()->name(old));
        _dims.emplace_back(old, created);
        return created;
    }
    IndexSpace space(const IndexSpace &old, bool chunk = false) noexcept {
        IndexSpace result;
        for (auto &&axis : old.axes()) {
            if (!axis.extent.is_constant() || !result.add(dim(axis.dimension), chunk && axis.dimension == _s.column ? _s.chunk : axis.extent.constant_value())) { _ok = false; }
        }
        return result;
    }
    Type type(const Type &old, bool chunk = false) noexcept {
        if (old.is_tile()) { return Type::tile(old.scalar_type(), space(*old.index_space(), chunk)); }
        if (old.is_view()) { return Type::view(old.scalar_type(), space(*old.index_space())); }
        return old;
    }
    Value *value(const Value *old) noexcept {
        auto it = _values.find(old);
        if (it == _values.end()) {
            _ok = false;
            return nullptr;
        }
        return it->second;
    }
    void arguments(const Block &old, Block &next, bool existing) noexcept {
        for (auto i = 0u; i < old.argument_count(); i++) {
            _values[old.argument(i)] = existing ? next.argument(i) : next.add_argument(type(old.argument(i)->type()), old.argument(i)->name());
        }
    }
    Operation *operation(const Operation &old, IRBuilder &builder, bool chunk = false) noexcept {
        luisa::vector<Value *> operands;
        for (auto i = 0u; i < old.operand_count(); i++) { operands.emplace_back(value(old.operand(i))); }
        if (!_ok) { return nullptr; }
        luisa::vector<Type> types;
        for (auto i = 0u; i < old.result_count(); i++) { types.emplace_back(type(old.result(i)->type(), chunk)); }
        Operation *next = nullptr;
        switch (old.kind()) {
            case OperationKind::ELEMENTWISE: next = builder.create_elementwise(old.elementwise_op(), operands, types.front()); break;
            case OperationKind::TILE_MAP: next = builder.create_tile_map(types.front()); break;
            case OperationKind::REDUCE:
                next = builder.create_structured(old.kind(), space(*old.domain(), chunk), operands, types);
                next->set_reduction_policy(old.reduction_policy());
                break;
            case OperationKind::VIEW_STORE:
                next = builder.create_tile_store(operands.front(), {operands.data() + 1u, operands.size() - 2u}, space(*old.domain()), operands.back(), old.bounds_mode());
                break;
            default: next = builder.create(old.kind(), operands, types); break;
        }
        if (next == nullptr) {
            _ok = false;
            return nullptr;
        }
        for (auto &&a : old.attributes()) { next->set_attribute(a.name, a.value); }
        for (auto i = 0u; i < old.result_count(); i++) { _values[old.result(i)] = next->result(i); }
        for (auto i = 0u; i < old.region_count(); i++) {
            auto a = old.region(i)->block(0u);
            auto b = next->region(i)->block(0u);
            arguments(*a, *b, true);
            IRBuilder child{b};
            for (auto op : a->operations()) {
                if (operation(*op, child, chunk) == nullptr) { return nullptr; }
            }
        }
        return next;
    }
    Value *constant(IRBuilder &b, const Type &t, Attribute v) noexcept {
        Type types[]{t};
        auto op = b.create(OperationKind::CONSTANT, {}, types);
        op->set_attribute("value", std::move(v));
        return op->result(0u);
    }
    void yield(IRBuilder &b, Value *v) noexcept {
        Value *operands[]{v};
        static_cast<void>(b.create(OperationKind::YIELD, operands));
    }

    bool streamed_map(IRBuilder &b) noexcept {
        auto old_body = _s.map->region(0u)->block(0u);
        // Accumulate corresponding contribution lanes across chunks before
        // communicating once. This is an admitted FP32 unordered SUM tree;
        // it does not reassociate an ordered reducer or change its precision.
        auto carry_type = Type::tile(ScalarType::FLOAT32, space(*_s.load->domain(), true));
        auto initial_map = b.create_tile_map(carry_type);
        IRBuilder seed_builder{initial_map->region(0u)->block(0u)};
        auto seed = constant(seed_builder, Type::scalar(ScalarType::FLOAT32), Attribute{0.0});
        yield(seed_builder, seed);
        IndexSpace iterations;
        static_cast<void>(iterations.add(_module.dimensions().create_dimension("contribution_chunk"), _s.trips));
        Value *initial[]{initial_map->result(0u)};
        Type results[]{carry_type};
        auto loop = b.create_structured(OperationKind::SERIAL, iterations, initial, results);
        auto loop_body = loop->region(0u)->block(0u);
        IRBuilder inner{loop_body};
        auto c = constant(inner, Type::scalar(ScalarType::INT64), Attribute{static_cast<int64_t>(_s.chunk)});
        Value *multiply_args[]{loop_body->argument(0u), c};
        auto start = inner.create_elementwise(ElementwiseOp::MUL, multiply_args, Type::scalar(ScalarType::INT64))->result(0u);
        Value *origins[]{value(_s.load->operand(1u)), start};
        if (!_ok) { return false; }
        auto load = inner.create_tile_load(value(_s.load->operand(0u)), origins, space(*_s.load->domain(), true), _s.load->bounds_mode());
        _values[_s.load->result(0u)] = load->result(0u);
        if (_s.cast != nullptr && operation(*_s.cast, inner, true) == nullptr) { return false; }
        Value *merge_args[]{loop_body->argument(1u), value(_s.source)};
        if (!_ok) { return false; }
        auto old_merge = _s.sum->region(0u)->block(0u)->operation(1u);
        if (old_merge->operand(0u) != _s.sum->region(0u)->block(0u)->argument(1u)) { std::swap(merge_args[0u], merge_args[1u]); }
        yield(inner, inner.create_elementwise(ElementwiseOp::ADD, merge_args, carry_type)->result(0u));

        // No value defined inside SERIAL escapes its body. Replace the closed
        // snapshot's one surviving use with the final FP32 chunk Tile and
        // clone its original literal-seeded REDUCE and pure epilogue once.
        _values[_s.source] = loop->result(0u);
        auto final = b.create_tile_map(type(_s.map->result(0u)->type()));
        auto final_body = final->region(0u)->block(0u);
        arguments(*old_body, *final_body, true);
        IRBuilder final_builder{final_body};
        for (auto op : old_body->operations()) {
            if (operation(*op, final_builder, true) == nullptr) { return false; }
        }
        _values[_s.map->result(0u)] = final->result(0u);
        return _ok;
    }

public:
    Clone(const Slice &s, Module &module) noexcept : _s{s}, _module{module} {}
    Function *run(const Function &old) noexcept {
        auto f = _module.create_function(old.name());
        auto root = f->body().append_block();
        arguments(*_s.root, *root, false);
        IRBuilder b{root};
        for (auto op : _s.root->operations()) {
            if (op != _s.parallel) {
                if (operation(*op, b) == nullptr) { return nullptr; }
                continue;
            }
            auto parallel = b.create_structured(OperationKind::PARALLEL, space(*op->domain()));
            auto program = parallel->region(0u)->block(0u);
            arguments(*_s.program, *program, true);
            IRBuilder body{program};
            for (auto child : _s.program->operations()) {
                if (child == _s.load || child == _s.cast || _s.dead.contains(child)) { continue; }
                if (child == _s.map) {
                    if (!streamed_map(body)) { return nullptr; }
                } else if (operation(*child, body) == nullptr) {
                    return nullptr;
                }
            }
        }
        return _ok ? f : nullptr;
    }
};

bool measure(const Block &block, uint64_t repetitions, StreamedSumIRFacts &facts) noexcept {
    auto tile_value = [&](const Value *v) {
        if (!v->type().is_tile()) { return true; }
        auto n = volume(v->type().index_space());
        facts.largest_materialized_tile_elements = std::max(facts.largest_materialized_tile_elements, n);
        uint64_t bytes{};
        return n != 0u && multiply(n, scalar_type_size(v->type().scalar_type()), bytes) && add(facts.explicit_tile_storage_upper_bound_per_program, bytes);
    };
    for (auto &&v : block.arguments()) {
        if (!tile_value(v.get())) { return false; }
    }
    for (auto op : block.operations()) {
        for (auto i = 0u; i < op->result_count(); i++) {
            if (!tile_value(op->result(i))) { return false; }
        }
        uint64_t amount{};
        if (op->kind() == OperationKind::ELEMENTWISE) {
            auto n = op->result(0u)->type().is_tile() ? volume(op->result(0u)->type().index_space()) : 1u;
            if (!multiply(n, repetitions, amount) || !add(facts.elementwise_elements_per_program, amount)) { return false; }
        } else if (op->kind() == OperationKind::VIEW_LOAD || op->kind() == OperationKind::VIEW_STORE) {
            if (!multiply(volume(&*op->domain()), scalar_type_size(op->operand(0u)->type().scalar_type()), amount) ||
                !multiply(amount, repetitions, amount) || !add(op->kind() == OperationKind::VIEW_LOAD ? facts.nominal_read_bytes_per_program : facts.nominal_write_bytes_per_program, amount)) { return false; }
        }
        auto nested = repetitions;
        if (op->kind() == OperationKind::SERIAL || op->kind() == OperationKind::REDUCE || op->kind() == OperationKind::TILE_MAP) {
            if (!multiply(nested, volume(&*op->domain()), nested)) { return false; }
            if (op->kind() == OperationKind::SERIAL && !add(facts.serial_iterations, nested)) { return false; }
            if (op->kind() == OperationKind::REDUCE &&
                (!add(facts.collective_invocations_per_program, repetitions) || !add(facts.contribution_elements_per_program, nested))) { return false; }
        }
        for (auto &&region : op->regions()) {
            if (!measure(*region->block(0u), nested, facts)) { return false; }
        }
    }
    return true;
}
}
}// namespace ::streamed_sum_detail

StreamedSumIR build_streamed_sum_ir(const tile::Function &original, uint32_t chunk_extent, bool enable_fast_math) noexcept {
    StreamedSumIR result;
    if (enable_fast_math) {
        result.error = "strict prototype only";
        return result;
    }
    streamed_sum_detail::Slice slice;
    if (!streamed_sum_detail::match(original, chunk_extent, slice, result.error)) { return result; }
    result.module = luisa::make_unique<tile::Module>();
    result.function = streamed_sum_detail::Clone{slice, *result.module}.run(original);
    if (result.function == nullptr || !tile::verify(*result.module).ok()) {
        result.error = "owned candidate clone failed verification";
        result.function = nullptr;
        return result;
    }
    result.disjoint = slice.disjoint;
    result.facts.discarded_pure_program_operations = slice.dead.size();
    const tile::Operation *parallel = nullptr;
    for (auto op : result.function->body().block(0u)->operations()) {
        if (op->kind() == tile::OperationKind::PARALLEL) { parallel = op; }
    }
    if (parallel != nullptr) { result.facts.programs = streamed_sum_detail::volume(&*parallel->domain()); }
    if (parallel == nullptr || !streamed_sum_detail::measure(*parallel->region(0u)->block(0u), 1u, result.facts)) {
        result.error = "candidate logical accounting overflow";
        result.function = nullptr;
    }
    return result;
}
}// namespace luisa::compute::cuda::native_tile
