#include <luisa/ir/ir.h>

namespace luisa::compute::ir_v2 {

const VectorElementType &VectorType::element() const noexcept { return detail::from_inner_ref(_inner.element); }
const uint32_t &VectorType::length() const noexcept { return detail::from_inner_ref(_inner.length); }
const VectorElementType &MatrixType::element() const noexcept { return detail::from_inner_ref(_inner.element); }
const uint32_t &MatrixType::dimension() const noexcept { return detail::from_inner_ref(_inner.dimension); }
luisa::span<const CArc<Type>> StructType::fields() const noexcept {
    return {reinterpret_cast<const CArc<Type> *>(_inner.fields.ptr), _inner.fields.len};
}

const size_t &StructType::alignment() const noexcept { return detail::from_inner_ref(_inner.alignment); }
const size_t &StructType::size() const noexcept { return detail::from_inner_ref(_inner.size); }
const CArc<Type> &ArrayType::element() const noexcept { return detail::from_inner_ref(_inner.element); }
const size_t &ArrayType::length() const noexcept { return detail::from_inner_ref(_inner.length); }
const CArc<Type> &Node::type_() const noexcept { return detail::from_inner_ref(_inner.type_); }
const NodeRef &Node::next() const noexcept { return detail::from_inner_ref(_inner.next); }
const NodeRef &Node::prev() const noexcept { return detail::from_inner_ref(_inner.prev); }
const CArc<Instruction> &Node::instruction() const noexcept { return detail::from_inner_ref(_inner.instruction); }

// including extra code from data/NodeRef.cpp
[[nodiscard]] const Node *NodeRef::operator->() const noexcept {
    return get();
}

[[nodiscard]] const Node *NodeRef::get() const noexcept {
    auto node = raw::luisa_compute_ir_node_get(_inner);
    return reinterpret_cast<const Node *>(node);
}

void NodeRef::insert_before_self(NodeRef node) noexcept {
    raw::luisa_compute_ir_node_insert_before_self(_inner, node.raw());
}

void NodeRef::insert_after_self(NodeRef node) noexcept {
    raw::luisa_compute_ir_node_insert_after_self(_inner, node.raw());
}

void NodeRef::replace_with(NodeRef node) noexcept {
    raw::luisa_compute_ir_node_replace_with(_inner, reinterpret_cast<const raw::Node *>(node.get()));
}

void NodeRef::remove() noexcept {
    raw::luisa_compute_ir_node_remove(_inner);
}
// end include

const NodeRef &PhiIncoming::value() const noexcept { return detail::from_inner_ref(_inner.value); }
const Pooled<BasicBlock> &PhiIncoming::block() const noexcept { return detail::from_inner_ref(_inner.block); }
const int32_t &SwitchCase::value() const noexcept { return detail::from_inner_ref(_inner.value); }
const Pooled<BasicBlock> &SwitchCase::block() const noexcept { return detail::from_inner_ref(_inner.block); }
const NodeRef &Instruction::Local::init() const noexcept { return detail::from_inner_ref(_inner.init); }
const bool &Instruction::Argument::by_value() const noexcept { return detail::from_inner_ref(_inner.by_value); }
const NodeRef &Instruction::Update::var() const noexcept { return detail::from_inner_ref(_inner.var); }
const NodeRef &Instruction::Update::value() const noexcept { return detail::from_inner_ref(_inner.value); }
const Pooled<BasicBlock> &Instruction::Loop::body() const noexcept { return detail::from_inner_ref(_inner.body); }
const NodeRef &Instruction::Loop::cond() const noexcept { return detail::from_inner_ref(_inner.cond); }
const Pooled<BasicBlock> &Instruction::GenericLoop::prepare() const noexcept { return detail::from_inner_ref(_inner.prepare); }
const NodeRef &Instruction::GenericLoop::cond() const noexcept { return detail::from_inner_ref(_inner.cond); }
const Pooled<BasicBlock> &Instruction::GenericLoop::body() const noexcept { return detail::from_inner_ref(_inner.body); }
const Pooled<BasicBlock> &Instruction::GenericLoop::update() const noexcept { return detail::from_inner_ref(_inner.update); }
const NodeRef &Instruction::If::cond() const noexcept { return detail::from_inner_ref(_inner.cond); }
const Pooled<BasicBlock> &Instruction::If::true_branch() const noexcept { return detail::from_inner_ref(_inner.true_branch); }
const Pooled<BasicBlock> &Instruction::If::false_branch() const noexcept { return detail::from_inner_ref(_inner.false_branch); }
const NodeRef &Instruction::Switch::value() const noexcept { return detail::from_inner_ref(_inner.value); }
const Pooled<BasicBlock> &Instruction::Switch::default_() const noexcept { return detail::from_inner_ref(_inner.default_); }
luisa::span<const SwitchCase> Instruction::Switch::cases() const noexcept {
    return {reinterpret_cast<const SwitchCase *>(_inner.cases.ptr), _inner.cases.len};
}

const Pooled<BasicBlock> &Instruction::AdScope::body() const noexcept { return detail::from_inner_ref(_inner.body); }
const bool &Instruction::AdScope::forward() const noexcept { return detail::from_inner_ref(_inner.forward); }
const size_t &Instruction::AdScope::n_forward_grads() const noexcept { return detail::from_inner_ref(_inner.n_forward_grads); }
const NodeRef &Instruction::RayQuery::ray_query() const noexcept { return detail::from_inner_ref(_inner.ray_query); }
const Pooled<BasicBlock> &Instruction::RayQuery::on_triangle_hit() const noexcept { return detail::from_inner_ref(_inner.on_triangle_hit); }
const Pooled<BasicBlock> &Instruction::RayQuery::on_procedural_hit() const noexcept { return detail::from_inner_ref(_inner.on_procedural_hit); }
luisa::span<const uint8_t> Instruction::Print::fmt() const noexcept {
    return {reinterpret_cast<const uint8_t *>(_inner.fmt.ptr), _inner.fmt.len};
}

luisa::span<const NodeRef> Instruction::Print::args() const noexcept {
    return {reinterpret_cast<const NodeRef *>(_inner.args.ptr), _inner.args.len};
}

const NodeRef &BasicBlock::first() const noexcept { return detail::from_inner_ref(_inner.first); }
const NodeRef &BasicBlock::last() const noexcept { return detail::from_inner_ref(_inner.last); }
const ModuleKind &Module::kind() const noexcept { return detail::from_inner_ref(_inner.kind); }
const Pooled<BasicBlock> &Module::entry() const noexcept { return detail::from_inner_ref(_inner.entry); }
const ModuleFlags &Module::flags() const noexcept { return detail::from_inner_ref(_inner.flags); }
const CArc<ModulePools> &Module::pools() const noexcept { return detail::from_inner_ref(_inner.pools); }
const Module &CallableModule::module() const noexcept { return detail::from_inner_ref(_inner.module); }
const CArc<Type> &CallableModule::ret_type() const noexcept { return detail::from_inner_ref(_inner.ret_type); }
luisa::span<const NodeRef> CallableModule::args() const noexcept {
    return {reinterpret_cast<const NodeRef *>(_inner.args.ptr), _inner.args.len};
}

luisa::span<const Capture> CallableModule::captures() const noexcept {
    return {reinterpret_cast<const Capture *>(_inner.captures.ptr), _inner.captures.len};
}

luisa::span<const CArc<CpuCustomOp>> CallableModule::cpu_custom_ops() const noexcept {
    return {reinterpret_cast<const CArc<CpuCustomOp> *>(_inner.cpu_custom_ops.ptr), _inner.cpu_custom_ops.len};
}

const CArc<ModulePools> &CallableModule::pools() const noexcept { return detail::from_inner_ref(_inner.pools); }
const uint64_t &BufferBinding::handle() const noexcept { return detail::from_inner_ref(_inner.handle); }
const uint64_t &BufferBinding::offset() const noexcept { return detail::from_inner_ref(_inner.offset); }
const size_t &BufferBinding::size() const noexcept { return detail::from_inner_ref(_inner.size); }
const uint64_t &TextureBinding::handle() const noexcept { return detail::from_inner_ref(_inner.handle); }
const uint32_t &TextureBinding::level() const noexcept { return detail::from_inner_ref(_inner.level); }
const uint64_t &BindlessArrayBinding::handle() const noexcept { return detail::from_inner_ref(_inner.handle); }
const uint64_t &AccelBinding::handle() const noexcept { return detail::from_inner_ref(_inner.handle); }
const NodeRef &Capture::node() const noexcept { return detail::from_inner_ref(_inner.node); }
const Binding &Capture::binding() const noexcept { return detail::from_inner_ref(_inner.binding); }
const Module &KernelModule::module() const noexcept { return detail::from_inner_ref(_inner.module); }
luisa::span<const Capture> KernelModule::captures() const noexcept {
    return {reinterpret_cast<const Capture *>(_inner.captures.ptr), _inner.captures.len};
}

luisa::span<const NodeRef> KernelModule::args() const noexcept {
    return {reinterpret_cast<const NodeRef *>(_inner.args.ptr), _inner.args.len};
}

luisa::span<const NodeRef> KernelModule::shared() const noexcept {
    return {reinterpret_cast<const NodeRef *>(_inner.shared.ptr), _inner.shared.len};
}

luisa::span<const CArc<CpuCustomOp>> KernelModule::cpu_custom_ops() const noexcept {
    return {reinterpret_cast<const CArc<CpuCustomOp> *>(_inner.cpu_custom_ops.ptr), _inner.cpu_custom_ops.len};
}

const std::array<uint32_t, 3> &KernelModule::block_size() const noexcept { return detail::from_inner_ref(_inner.block_size); }
const CArc<ModulePools> &KernelModule::pools() const noexcept { return detail::from_inner_ref(_inner.pools); }
const Module &BlockModule::module() const noexcept { return detail::from_inner_ref(_inner.module); }
const Pooled<BasicBlock> &IrBuilder::bb() const noexcept { return detail::from_inner_ref(_inner.bb); }
const CArc<ModulePools> &IrBuilder::pools() const noexcept { return detail::from_inner_ref(_inner.pools); }
const NodeRef &IrBuilder::insert_point() const noexcept { return detail::from_inner_ref(_inner.insert_point); }

// including extra code from data/IrBuilder.cpp
NodeRef IrBuilder::call(const Func &f, luisa::span<const NodeRef> args, const CArc<Type> &type) noexcept {
    auto f_inner = f._inner;
    auto slice = raw::CSlice{reinterpret_cast<const raw::NodeRef *>(args.data()), args.size()};
    auto node = luisa_compute_ir_build_call(&_inner, f_inner, slice, luisa::bit_cast<CArc<raw::Type>>(type.clone()));
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::phi(luisa::span<const PhiIncoming> incoming, const CArc<Type> &type) noexcept {
    auto incoming_ = raw::CSlice{reinterpret_cast<const raw::PhiIncoming *>(incoming.data()), incoming.size()};
    auto node = luisa_compute_ir_build_phi(&_inner, incoming_, luisa::bit_cast<CArc<raw::Type>>(type.clone()));
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::local(NodeRef init) noexcept {
    auto node = luisa_compute_ir_build_local(&_inner, init._inner);
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::local(const CppOwnedCArc<Type> &type) noexcept {
    auto node = luisa_compute_ir_build_local_zero_init(&_inner, luisa::bit_cast<CArc<raw::Type>>(type.clone()));
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::if_(const NodeRef &cond, const Pooled<BasicBlock> &true_branch, const Pooled<BasicBlock> &false_branch) noexcept {
    auto node = luisa_compute_ir_build_if(&_inner,
                                          cond._inner,
                                          reinterpret_cast<const Pooled<raw::BasicBlock> &>(true_branch),
                                          reinterpret_cast<const Pooled<raw::BasicBlock> &>(false_branch));
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::switch_(const NodeRef &value, luisa::span<const SwitchCase> cases, const Pooled<BasicBlock> &default_) noexcept {
    auto cases_ = raw::CSlice{reinterpret_cast<const raw::SwitchCase *>(cases.data()), cases.size()};
    auto node = luisa_compute_ir_build_switch(&_inner, value._inner, cases_, reinterpret_cast<const Pooled<raw::BasicBlock> &>(default_));
    return NodeRef::from_raw(node);
}
// NodeRef IrBuilder::loop(const Pooled<BasicBlock> &body, const NodeRef &cond) noexcept;
NodeRef IrBuilder::generic_loop(const Pooled<BasicBlock> &prepare, const NodeRef &cond, const Pooled<BasicBlock> &body, const Pooled<BasicBlock> &update) noexcept {
    auto node = luisa_compute_ir_build_generic_loop(&_inner,
                                                    reinterpret_cast<const Pooled<raw::BasicBlock> &>(prepare),
                                                    cond._inner,
                                                    reinterpret_cast<const Pooled<raw::BasicBlock> &>(body),
                                                    reinterpret_cast<const Pooled<raw::BasicBlock> &>(update));
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::loop(const Pooled<BasicBlock> &body, const NodeRef &cond) noexcept {
    auto node = luisa_compute_ir_build_loop(&_inner,
                                            reinterpret_cast<const Pooled<raw::BasicBlock> &>(body),
                                            cond._inner);
    return NodeRef::from_raw(node);
}
Pooled<BasicBlock> IrBuilder::finish(IrBuilder &&builder) noexcept {
    auto block = luisa_compute_ir_build_finish(builder._inner);
    return luisa::bit_cast<Pooled<BasicBlock>>(block);
}

void IrBuilder::set_insert_point(const NodeRef &node) noexcept {
    raw::luisa_compute_ir_builder_set_insert_point(&_inner, node._inner);
}
// end include

}// namespace luisa::compute::ir_v2
