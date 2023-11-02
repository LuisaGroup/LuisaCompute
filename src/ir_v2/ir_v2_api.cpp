#include <luisa/ir_v2/ir_v2_defs.h>
#include <luisa/ir_v2/ir_v2.h>
#include <luisa/ir_v2/ir_v2_api.h>
namespace luisa::compute::ir_v2 {
static AssumeFn *Func_as_AssumeFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<AssumeFn>();
}
static UnreachableFn *Func_as_UnreachableFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<UnreachableFn>();
}
static AssertFn *Func_as_AssertFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<AssertFn>();
}
static BindlessAtomicExchangeFn *Func_as_BindlessAtomicExchangeFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<BindlessAtomicExchangeFn>();
}
static BindlessAtomicCompareExchangeFn *Func_as_BindlessAtomicCompareExchangeFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<BindlessAtomicCompareExchangeFn>();
}
static BindlessAtomicFetchAddFn *Func_as_BindlessAtomicFetchAddFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<BindlessAtomicFetchAddFn>();
}
static BindlessAtomicFetchSubFn *Func_as_BindlessAtomicFetchSubFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<BindlessAtomicFetchSubFn>();
}
static BindlessAtomicFetchAndFn *Func_as_BindlessAtomicFetchAndFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<BindlessAtomicFetchAndFn>();
}
static BindlessAtomicFetchOrFn *Func_as_BindlessAtomicFetchOrFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<BindlessAtomicFetchOrFn>();
}
static BindlessAtomicFetchXorFn *Func_as_BindlessAtomicFetchXorFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<BindlessAtomicFetchXorFn>();
}
static BindlessAtomicFetchMinFn *Func_as_BindlessAtomicFetchMinFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<BindlessAtomicFetchMinFn>();
}
static BindlessAtomicFetchMaxFn *Func_as_BindlessAtomicFetchMaxFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<BindlessAtomicFetchMaxFn>();
}
static CallableFn *Func_as_CallableFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<CallableFn>();
}
static CpuExtFn *Func_as_CpuExtFn(CFunc *self) {
    return reinterpret_cast<Func *>(self)->as<CpuExtFn>();
}
static RustyFuncTag Func_tag(const CFunc *self) {
    return static_cast<RustyFuncTag>(reinterpret_cast<const Func *>(self)->tag());
}
static Slice<const char> AssumeFn_msg(AssumeFn *self) {
    return self->msg;
}
static void AssumeFn_set_msg(AssumeFn *self, Slice<const char> value) {
    self->msg = value.to_string();
}
static CFunc AssumeFn_new(Pool *pool, Slice<const char> msg) {
    auto data = luisa::unique_ptr<AssumeFn>();
    AssumeFn_set_msg(data.get(), msg);
    auto tag = AssumeFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static Slice<const char> UnreachableFn_msg(UnreachableFn *self) {
    return self->msg;
}
static void UnreachableFn_set_msg(UnreachableFn *self, Slice<const char> value) {
    self->msg = value.to_string();
}
static CFunc UnreachableFn_new(Pool *pool, Slice<const char> msg) {
    auto data = luisa::unique_ptr<UnreachableFn>();
    UnreachableFn_set_msg(data.get(), msg);
    auto tag = UnreachableFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static Slice<const char> AssertFn_msg(AssertFn *self) {
    return self->msg;
}
static void AssertFn_set_msg(AssertFn *self, Slice<const char> value) {
    self->msg = value.to_string();
}
static CFunc AssertFn_new(Pool *pool, Slice<const char> msg) {
    auto data = luisa::unique_ptr<AssertFn>();
    AssertFn_set_msg(data.get(), msg);
    auto tag = AssertFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static const Type *BindlessAtomicExchangeFn_ty(BindlessAtomicExchangeFn *self) {
    return self->ty;
}
static void BindlessAtomicExchangeFn_set_ty(BindlessAtomicExchangeFn *self, const Type *value) {
    self->ty = value;
}
static CFunc BindlessAtomicExchangeFn_new(Pool *pool, const Type *ty) {
    auto data = luisa::unique_ptr<BindlessAtomicExchangeFn>();
    BindlessAtomicExchangeFn_set_ty(data.get(), ty);
    auto tag = BindlessAtomicExchangeFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static const Type *BindlessAtomicCompareExchangeFn_ty(BindlessAtomicCompareExchangeFn *self) {
    return self->ty;
}
static void BindlessAtomicCompareExchangeFn_set_ty(BindlessAtomicCompareExchangeFn *self, const Type *value) {
    self->ty = value;
}
static CFunc BindlessAtomicCompareExchangeFn_new(Pool *pool, const Type *ty) {
    auto data = luisa::unique_ptr<BindlessAtomicCompareExchangeFn>();
    BindlessAtomicCompareExchangeFn_set_ty(data.get(), ty);
    auto tag = BindlessAtomicCompareExchangeFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static const Type *BindlessAtomicFetchAddFn_ty(BindlessAtomicFetchAddFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchAddFn_set_ty(BindlessAtomicFetchAddFn *self, const Type *value) {
    self->ty = value;
}
static CFunc BindlessAtomicFetchAddFn_new(Pool *pool, const Type *ty) {
    auto data = luisa::unique_ptr<BindlessAtomicFetchAddFn>();
    BindlessAtomicFetchAddFn_set_ty(data.get(), ty);
    auto tag = BindlessAtomicFetchAddFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static const Type *BindlessAtomicFetchSubFn_ty(BindlessAtomicFetchSubFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchSubFn_set_ty(BindlessAtomicFetchSubFn *self, const Type *value) {
    self->ty = value;
}
static CFunc BindlessAtomicFetchSubFn_new(Pool *pool, const Type *ty) {
    auto data = luisa::unique_ptr<BindlessAtomicFetchSubFn>();
    BindlessAtomicFetchSubFn_set_ty(data.get(), ty);
    auto tag = BindlessAtomicFetchSubFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static const Type *BindlessAtomicFetchAndFn_ty(BindlessAtomicFetchAndFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchAndFn_set_ty(BindlessAtomicFetchAndFn *self, const Type *value) {
    self->ty = value;
}
static CFunc BindlessAtomicFetchAndFn_new(Pool *pool, const Type *ty) {
    auto data = luisa::unique_ptr<BindlessAtomicFetchAndFn>();
    BindlessAtomicFetchAndFn_set_ty(data.get(), ty);
    auto tag = BindlessAtomicFetchAndFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static const Type *BindlessAtomicFetchOrFn_ty(BindlessAtomicFetchOrFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchOrFn_set_ty(BindlessAtomicFetchOrFn *self, const Type *value) {
    self->ty = value;
}
static CFunc BindlessAtomicFetchOrFn_new(Pool *pool, const Type *ty) {
    auto data = luisa::unique_ptr<BindlessAtomicFetchOrFn>();
    BindlessAtomicFetchOrFn_set_ty(data.get(), ty);
    auto tag = BindlessAtomicFetchOrFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static const Type *BindlessAtomicFetchXorFn_ty(BindlessAtomicFetchXorFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchXorFn_set_ty(BindlessAtomicFetchXorFn *self, const Type *value) {
    self->ty = value;
}
static CFunc BindlessAtomicFetchXorFn_new(Pool *pool, const Type *ty) {
    auto data = luisa::unique_ptr<BindlessAtomicFetchXorFn>();
    BindlessAtomicFetchXorFn_set_ty(data.get(), ty);
    auto tag = BindlessAtomicFetchXorFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static const Type *BindlessAtomicFetchMinFn_ty(BindlessAtomicFetchMinFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchMinFn_set_ty(BindlessAtomicFetchMinFn *self, const Type *value) {
    self->ty = value;
}
static CFunc BindlessAtomicFetchMinFn_new(Pool *pool, const Type *ty) {
    auto data = luisa::unique_ptr<BindlessAtomicFetchMinFn>();
    BindlessAtomicFetchMinFn_set_ty(data.get(), ty);
    auto tag = BindlessAtomicFetchMinFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static const Type *BindlessAtomicFetchMaxFn_ty(BindlessAtomicFetchMaxFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchMaxFn_set_ty(BindlessAtomicFetchMaxFn *self, const Type *value) {
    self->ty = value;
}
static CFunc BindlessAtomicFetchMaxFn_new(Pool *pool, const Type *ty) {
    auto data = luisa::unique_ptr<BindlessAtomicFetchMaxFn>();
    BindlessAtomicFetchMaxFn_set_ty(data.get(), ty);
    auto tag = BindlessAtomicFetchMaxFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static CallableModule *CallableFn_module(CallableFn *self) {
    return self->module.get();
}
static void CallableFn_set_module(CallableFn *self, CallableModule *value) {
    self->module = luisa::static_pointer_cast<std::decay_t<decltype(self->module)>::element_type>(value->shared_from_this());
}
static CFunc CallableFn_new(Pool *pool, CallableModule *module) {
    auto data = luisa::unique_ptr<CallableFn>();
    CallableFn_set_module(data.get(), module);
    auto tag = CallableFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static CpuExternFn *CpuExtFn_f(CpuExtFn *self) {
    return self->f.get();
}
static void CpuExtFn_set_f(CpuExtFn *self, CpuExternFn *value) {
    self->f = luisa::static_pointer_cast<std::decay_t<decltype(self->f)>::element_type>(value->shared_from_this());
}
static CFunc CpuExtFn_new(Pool *pool, CpuExternFn *f) {
    auto data = luisa::unique_ptr<CpuExtFn>();
    CpuExtFn_set_f(data.get(), f);
    auto tag = CpuExtFn::static_tag();
    auto cobj = CFunc{};
    auto obj = Func(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static CFunc Func_new(Pool *pool, RustyFuncTag tag) {
    auto obj = Func(static_cast<FuncTag>(tag));
    auto cobj = CFunc{};
    std::memcpy(&cobj, &obj, sizeof(CFunc));
    (void)obj.steal();
    return cobj;
}
static ArgumentInst *Instruction_as_ArgumentInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<ArgumentInst>();
}
static ConstantInst *Instruction_as_ConstantInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<ConstantInst>();
}
static CallInst *Instruction_as_CallInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<CallInst>();
}
static PhiInst *Instruction_as_PhiInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<PhiInst>();
}
static IfInst *Instruction_as_IfInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<IfInst>();
}
static GenericLoopInst *Instruction_as_GenericLoopInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<GenericLoopInst>();
}
static SwitchInst *Instruction_as_SwitchInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<SwitchInst>();
}
static LocalInst *Instruction_as_LocalInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<LocalInst>();
}
static ReturnInst *Instruction_as_ReturnInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<ReturnInst>();
}
static PrintInst *Instruction_as_PrintInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<PrintInst>();
}
static CommentInst *Instruction_as_CommentInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<CommentInst>();
}
static UpdateInst *Instruction_as_UpdateInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<UpdateInst>();
}
static RayQueryInst *Instruction_as_RayQueryInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<RayQueryInst>();
}
static RevAutodiffInst *Instruction_as_RevAutodiffInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<RevAutodiffInst>();
}
static FwdAutodiffInst *Instruction_as_FwdAutodiffInst(CInstruction *self) {
    return reinterpret_cast<Instruction *>(self)->as<FwdAutodiffInst>();
}
static RustyInstructionTag Instruction_tag(const CInstruction *self) {
    return static_cast<RustyInstructionTag>(reinterpret_cast<const Instruction *>(self)->tag());
}
static bool ArgumentInst_by_value(ArgumentInst *self) {
    return self->by_value;
}
static void ArgumentInst_set_by_value(ArgumentInst *self, bool value) {
    self->by_value = value;
}
static CInstruction ArgumentInst_new(Pool *pool, bool by_value) {
    auto data = luisa::unique_ptr<ArgumentInst>();
    ArgumentInst_set_by_value(data.get(), by_value);
    auto tag = ArgumentInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static const Type *ConstantInst_ty(ConstantInst *self) {
    return self->ty;
}
static Slice<uint8_t> ConstantInst_value(ConstantInst *self) {
    return self->value;
}
static void ConstantInst_set_ty(ConstantInst *self, const Type *value) {
    self->ty = value;
}
static void ConstantInst_set_value(ConstantInst *self, Slice<uint8_t> value) {
    self->value = value.to_vector();
}
static CInstruction ConstantInst_new(Pool *pool, const Type *ty, Slice<uint8_t> value) {
    auto data = luisa::unique_ptr<ConstantInst>();
    ConstantInst_set_ty(data.get(), ty);
    ConstantInst_set_value(data.get(), value);
    auto tag = ConstantInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static const CFunc *CallInst_func(CallInst *self) {
    return reinterpret_cast<const CFunc *>(&self->func);
}
static Slice<Node *> CallInst_args(CallInst *self) {
    return self->args;
}
static void CallInst_set_func(CallInst *self, CFunc value) {
    self->func = std::move(*reinterpret_cast<Func *>(&value));
}
static void CallInst_set_args(CallInst *self, Slice<Node *> value) {
    self->args = value.to_vector();
}
static CInstruction CallInst_new(Pool *pool, CFunc func, Slice<Node *> args) {
    auto data = luisa::unique_ptr<CallInst>();
    CallInst_set_func(data.get(), func);
    CallInst_set_args(data.get(), args);
    auto tag = CallInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static Slice<PhiIncoming> PhiInst_incomings(PhiInst *self) {
    return self->incomings;
}
static void PhiInst_set_incomings(PhiInst *self, Slice<PhiIncoming> value) {
    self->incomings = value.to_vector();
}
static CInstruction PhiInst_new(Pool *pool, Slice<PhiIncoming> incomings) {
    auto data = luisa::unique_ptr<PhiInst>();
    PhiInst_set_incomings(data.get(), incomings);
    auto tag = PhiInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static Node *IfInst_cond(IfInst *self) {
    return self->cond;
}
static const BasicBlock *IfInst_true_branch(IfInst *self) {
    return self->true_branch;
}
static const BasicBlock *IfInst_false_branch(IfInst *self) {
    return self->false_branch;
}
static void IfInst_set_cond(IfInst *self, Node *value) {
    self->cond = value;
}
static void IfInst_set_true_branch(IfInst *self, const BasicBlock *value) {
    self->true_branch = value;
}
static void IfInst_set_false_branch(IfInst *self, const BasicBlock *value) {
    self->false_branch = value;
}
static CInstruction IfInst_new(Pool *pool, Node *cond, const BasicBlock *true_branch, const BasicBlock *false_branch) {
    auto data = luisa::unique_ptr<IfInst>();
    IfInst_set_cond(data.get(), cond);
    IfInst_set_true_branch(data.get(), true_branch);
    IfInst_set_false_branch(data.get(), false_branch);
    auto tag = IfInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static const BasicBlock *GenericLoopInst_prepare(GenericLoopInst *self) {
    return self->prepare;
}
static Node *GenericLoopInst_cond(GenericLoopInst *self) {
    return self->cond;
}
static const BasicBlock *GenericLoopInst_body(GenericLoopInst *self) {
    return self->body;
}
static const BasicBlock *GenericLoopInst_update(GenericLoopInst *self) {
    return self->update;
}
static void GenericLoopInst_set_prepare(GenericLoopInst *self, const BasicBlock *value) {
    self->prepare = value;
}
static void GenericLoopInst_set_cond(GenericLoopInst *self, Node *value) {
    self->cond = value;
}
static void GenericLoopInst_set_body(GenericLoopInst *self, const BasicBlock *value) {
    self->body = value;
}
static void GenericLoopInst_set_update(GenericLoopInst *self, const BasicBlock *value) {
    self->update = value;
}
static CInstruction GenericLoopInst_new(Pool *pool, const BasicBlock *prepare, Node *cond, const BasicBlock *body, const BasicBlock *update) {
    auto data = luisa::unique_ptr<GenericLoopInst>();
    GenericLoopInst_set_prepare(data.get(), prepare);
    GenericLoopInst_set_cond(data.get(), cond);
    GenericLoopInst_set_body(data.get(), body);
    GenericLoopInst_set_update(data.get(), update);
    auto tag = GenericLoopInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static Node *SwitchInst_value(SwitchInst *self) {
    return self->value;
}
static Slice<SwitchCase> SwitchInst_cases(SwitchInst *self) {
    return self->cases;
}
static const BasicBlock *SwitchInst_default_(SwitchInst *self) {
    return self->default_;
}
static void SwitchInst_set_value(SwitchInst *self, Node *value) {
    self->value = value;
}
static void SwitchInst_set_cases(SwitchInst *self, Slice<SwitchCase> value) {
    self->cases = value.to_vector();
}
static void SwitchInst_set_default_(SwitchInst *self, const BasicBlock *value) {
    self->default_ = value;
}
static CInstruction SwitchInst_new(Pool *pool, Node *value, Slice<SwitchCase> cases, const BasicBlock *default_) {
    auto data = luisa::unique_ptr<SwitchInst>();
    SwitchInst_set_value(data.get(), value);
    SwitchInst_set_cases(data.get(), cases);
    SwitchInst_set_default_(data.get(), default_);
    auto tag = SwitchInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static Node *LocalInst_init(LocalInst *self) {
    return self->init;
}
static void LocalInst_set_init(LocalInst *self, Node *value) {
    self->init = value;
}
static CInstruction LocalInst_new(Pool *pool, Node *init) {
    auto data = luisa::unique_ptr<LocalInst>();
    LocalInst_set_init(data.get(), init);
    auto tag = LocalInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static Node *ReturnInst_value(ReturnInst *self) {
    return self->value;
}
static void ReturnInst_set_value(ReturnInst *self, Node *value) {
    self->value = value;
}
static CInstruction ReturnInst_new(Pool *pool, Node *value) {
    auto data = luisa::unique_ptr<ReturnInst>();
    ReturnInst_set_value(data.get(), value);
    auto tag = ReturnInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static Slice<const char> PrintInst_fmt(PrintInst *self) {
    return self->fmt;
}
static Slice<Node *> PrintInst_args(PrintInst *self) {
    return self->args;
}
static void PrintInst_set_fmt(PrintInst *self, Slice<const char> value) {
    self->fmt = value.to_string();
}
static void PrintInst_set_args(PrintInst *self, Slice<Node *> value) {
    self->args = value.to_vector();
}
static CInstruction PrintInst_new(Pool *pool, Slice<const char> fmt, Slice<Node *> args) {
    auto data = luisa::unique_ptr<PrintInst>();
    PrintInst_set_fmt(data.get(), fmt);
    PrintInst_set_args(data.get(), args);
    auto tag = PrintInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static Slice<const char> CommentInst_comment(CommentInst *self) {
    return self->comment;
}
static void CommentInst_set_comment(CommentInst *self, Slice<const char> value) {
    self->comment = value.to_string();
}
static CInstruction CommentInst_new(Pool *pool, Slice<const char> comment) {
    auto data = luisa::unique_ptr<CommentInst>();
    CommentInst_set_comment(data.get(), comment);
    auto tag = CommentInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static Node *UpdateInst_var(UpdateInst *self) {
    return self->var;
}
static Node *UpdateInst_value(UpdateInst *self) {
    return self->value;
}
static void UpdateInst_set_var(UpdateInst *self, Node *value) {
    self->var = value;
}
static void UpdateInst_set_value(UpdateInst *self, Node *value) {
    self->value = value;
}
static CInstruction UpdateInst_new(Pool *pool, Node *var, Node *value) {
    auto data = luisa::unique_ptr<UpdateInst>();
    UpdateInst_set_var(data.get(), var);
    UpdateInst_set_value(data.get(), value);
    auto tag = UpdateInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static Node *RayQueryInst_query(RayQueryInst *self) {
    return self->query;
}
static const BasicBlock *RayQueryInst_on_triangle_hit(RayQueryInst *self) {
    return self->on_triangle_hit;
}
static const BasicBlock *RayQueryInst_on_procedural_hit(RayQueryInst *self) {
    return self->on_procedural_hit;
}
static void RayQueryInst_set_query(RayQueryInst *self, Node *value) {
    self->query = value;
}
static void RayQueryInst_set_on_triangle_hit(RayQueryInst *self, const BasicBlock *value) {
    self->on_triangle_hit = value;
}
static void RayQueryInst_set_on_procedural_hit(RayQueryInst *self, const BasicBlock *value) {
    self->on_procedural_hit = value;
}
static CInstruction RayQueryInst_new(Pool *pool, Node *query, const BasicBlock *on_triangle_hit, const BasicBlock *on_procedural_hit) {
    auto data = luisa::unique_ptr<RayQueryInst>();
    RayQueryInst_set_query(data.get(), query);
    RayQueryInst_set_on_triangle_hit(data.get(), on_triangle_hit);
    RayQueryInst_set_on_procedural_hit(data.get(), on_procedural_hit);
    auto tag = RayQueryInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static const BasicBlock *RevAutodiffInst_body(RevAutodiffInst *self) {
    return self->body;
}
static void RevAutodiffInst_set_body(RevAutodiffInst *self, const BasicBlock *value) {
    self->body = value;
}
static CInstruction RevAutodiffInst_new(Pool *pool, const BasicBlock *body) {
    auto data = luisa::unique_ptr<RevAutodiffInst>();
    RevAutodiffInst_set_body(data.get(), body);
    auto tag = RevAutodiffInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static const BasicBlock *FwdAutodiffInst_body(FwdAutodiffInst *self) {
    return self->body;
}
static void FwdAutodiffInst_set_body(FwdAutodiffInst *self, const BasicBlock *value) {
    self->body = value;
}
static CInstruction FwdAutodiffInst_new(Pool *pool, const BasicBlock *body) {
    auto data = luisa::unique_ptr<FwdAutodiffInst>();
    FwdAutodiffInst_set_body(data.get(), body);
    auto tag = FwdAutodiffInst::static_tag();
    auto cobj = CInstruction{};
    auto obj = Instruction(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static CInstruction Instruction_new(Pool *pool, RustyInstructionTag tag) {
    auto obj = Instruction(static_cast<InstructionTag>(tag));
    auto cobj = CInstruction{};
    std::memcpy(&cobj, &obj, sizeof(CInstruction));
    (void)obj.steal();
    return cobj;
}
static FuncMetadata _func_metadata[] = {
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {true},
    {false},
    {true},
    {true},
    {false},
    {true},
    {false},
    {false},
    {false},
    {false},
    {true},
    {true},
    {true},
    {true},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {true},
    {true},
    {true},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {false},
    {false},
    {true},
    {false},
    {false},
    {false},
    {true},
    {false},
    {false},
    {true},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {true},
    {false},
    {false},
    {false},
    {true},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {false},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {true},
    {false},
    {false},
    {false},
};
static_assert(sizeof(_func_metadata) == sizeof(FuncMetadata) * 215);
const FuncMetadata *func_metadata() { return _func_metadata; }
static BufferBinding *Binding_as_BufferBinding(CBinding *self) {
    return reinterpret_cast<Binding *>(self)->as<BufferBinding>();
}
static TextureBinding *Binding_as_TextureBinding(CBinding *self) {
    return reinterpret_cast<Binding *>(self)->as<TextureBinding>();
}
static BindlessArrayBinding *Binding_as_BindlessArrayBinding(CBinding *self) {
    return reinterpret_cast<Binding *>(self)->as<BindlessArrayBinding>();
}
static AccelBinding *Binding_as_AccelBinding(CBinding *self) {
    return reinterpret_cast<Binding *>(self)->as<AccelBinding>();
}
static RustyBindingTag Binding_tag(const CBinding *self) {
    return static_cast<RustyBindingTag>(reinterpret_cast<const Binding *>(self)->tag());
}
static uint64_t BufferBinding_handle(BufferBinding *self) {
    return self->handle;
}
static uint64_t BufferBinding_offset(BufferBinding *self) {
    return self->offset;
}
static uint64_t BufferBinding_size(BufferBinding *self) {
    return self->size;
}
static void BufferBinding_set_handle(BufferBinding *self, uint64_t value) {
    self->handle = value;
}
static void BufferBinding_set_offset(BufferBinding *self, uint64_t value) {
    self->offset = value;
}
static void BufferBinding_set_size(BufferBinding *self, uint64_t value) {
    self->size = value;
}
static CBinding BufferBinding_new(Pool *pool, uint64_t handle, uint64_t offset, uint64_t size) {
    auto data = luisa::unique_ptr<BufferBinding>();
    BufferBinding_set_handle(data.get(), handle);
    BufferBinding_set_offset(data.get(), offset);
    BufferBinding_set_size(data.get(), size);
    auto tag = BufferBinding::static_tag();
    auto cobj = CBinding{};
    auto obj = Binding(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CBinding));
    (void)obj.steal();
    return cobj;
}
static uint64_t TextureBinding_handle(TextureBinding *self) {
    return self->handle;
}
static uint64_t TextureBinding_level(TextureBinding *self) {
    return self->level;
}
static void TextureBinding_set_handle(TextureBinding *self, uint64_t value) {
    self->handle = value;
}
static void TextureBinding_set_level(TextureBinding *self, uint64_t value) {
    self->level = value;
}
static CBinding TextureBinding_new(Pool *pool, uint64_t handle, uint64_t level) {
    auto data = luisa::unique_ptr<TextureBinding>();
    TextureBinding_set_handle(data.get(), handle);
    TextureBinding_set_level(data.get(), level);
    auto tag = TextureBinding::static_tag();
    auto cobj = CBinding{};
    auto obj = Binding(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CBinding));
    (void)obj.steal();
    return cobj;
}
static uint64_t BindlessArrayBinding_handle(BindlessArrayBinding *self) {
    return self->handle;
}
static void BindlessArrayBinding_set_handle(BindlessArrayBinding *self, uint64_t value) {
    self->handle = value;
}
static CBinding BindlessArrayBinding_new(Pool *pool, uint64_t handle) {
    auto data = luisa::unique_ptr<BindlessArrayBinding>();
    BindlessArrayBinding_set_handle(data.get(), handle);
    auto tag = BindlessArrayBinding::static_tag();
    auto cobj = CBinding{};
    auto obj = Binding(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CBinding));
    (void)obj.steal();
    return cobj;
}
static uint64_t AccelBinding_handle(AccelBinding *self) {
    return self->handle;
}
static void AccelBinding_set_handle(AccelBinding *self, uint64_t value) {
    self->handle = value;
}
static CBinding AccelBinding_new(Pool *pool, uint64_t handle) {
    auto data = luisa::unique_ptr<AccelBinding>();
    AccelBinding_set_handle(data.get(), handle);
    auto tag = AccelBinding::static_tag();
    auto cobj = CBinding{};
    auto obj = Binding(tag, std::move(data));
    std::memcpy(&cobj, &obj, sizeof(CBinding));
    (void)obj.steal();
    return cobj;
}
static CBinding Binding_new(Pool *pool, RustyBindingTag tag) {
    auto obj = Binding(static_cast<BindingTag>(tag));
    auto cobj = CBinding{};
    std::memcpy(&cobj, &obj, sizeof(CBinding));
    (void)obj.steal();
    return cobj;
}
extern "C" LC_IR_API IrV2BindingTable lc_ir_v2_binding_table() {
    return {
        Func_as_AssumeFn,
        Func_as_UnreachableFn,
        Func_as_AssertFn,
        Func_as_BindlessAtomicExchangeFn,
        Func_as_BindlessAtomicCompareExchangeFn,
        Func_as_BindlessAtomicFetchAddFn,
        Func_as_BindlessAtomicFetchSubFn,
        Func_as_BindlessAtomicFetchAndFn,
        Func_as_BindlessAtomicFetchOrFn,
        Func_as_BindlessAtomicFetchXorFn,
        Func_as_BindlessAtomicFetchMinFn,
        Func_as_BindlessAtomicFetchMaxFn,
        Func_as_CallableFn,
        Func_as_CpuExtFn,
        Func_tag,
        AssumeFn_msg,
        AssumeFn_set_msg,
        AssumeFn_new,
        UnreachableFn_msg,
        UnreachableFn_set_msg,
        UnreachableFn_new,
        AssertFn_msg,
        AssertFn_set_msg,
        AssertFn_new,
        BindlessAtomicExchangeFn_ty,
        BindlessAtomicExchangeFn_set_ty,
        BindlessAtomicExchangeFn_new,
        BindlessAtomicCompareExchangeFn_ty,
        BindlessAtomicCompareExchangeFn_set_ty,
        BindlessAtomicCompareExchangeFn_new,
        BindlessAtomicFetchAddFn_ty,
        BindlessAtomicFetchAddFn_set_ty,
        BindlessAtomicFetchAddFn_new,
        BindlessAtomicFetchSubFn_ty,
        BindlessAtomicFetchSubFn_set_ty,
        BindlessAtomicFetchSubFn_new,
        BindlessAtomicFetchAndFn_ty,
        BindlessAtomicFetchAndFn_set_ty,
        BindlessAtomicFetchAndFn_new,
        BindlessAtomicFetchOrFn_ty,
        BindlessAtomicFetchOrFn_set_ty,
        BindlessAtomicFetchOrFn_new,
        BindlessAtomicFetchXorFn_ty,
        BindlessAtomicFetchXorFn_set_ty,
        BindlessAtomicFetchXorFn_new,
        BindlessAtomicFetchMinFn_ty,
        BindlessAtomicFetchMinFn_set_ty,
        BindlessAtomicFetchMinFn_new,
        BindlessAtomicFetchMaxFn_ty,
        BindlessAtomicFetchMaxFn_set_ty,
        BindlessAtomicFetchMaxFn_new,
        CallableFn_module,
        CallableFn_set_module,
        CallableFn_new,
        CpuExtFn_f,
        CpuExtFn_set_f,
        CpuExtFn_new,
        Func_new,
        Instruction_as_ArgumentInst,
        Instruction_as_ConstantInst,
        Instruction_as_CallInst,
        Instruction_as_PhiInst,
        Instruction_as_IfInst,
        Instruction_as_GenericLoopInst,
        Instruction_as_SwitchInst,
        Instruction_as_LocalInst,
        Instruction_as_ReturnInst,
        Instruction_as_PrintInst,
        Instruction_as_CommentInst,
        Instruction_as_UpdateInst,
        Instruction_as_RayQueryInst,
        Instruction_as_RevAutodiffInst,
        Instruction_as_FwdAutodiffInst,
        Instruction_tag,
        ArgumentInst_by_value,
        ArgumentInst_set_by_value,
        ArgumentInst_new,
        ConstantInst_ty,
        ConstantInst_value,
        ConstantInst_set_ty,
        ConstantInst_set_value,
        ConstantInst_new,
        CallInst_func,
        CallInst_args,
        CallInst_set_func,
        CallInst_set_args,
        CallInst_new,
        PhiInst_incomings,
        PhiInst_set_incomings,
        PhiInst_new,
        IfInst_cond,
        IfInst_true_branch,
        IfInst_false_branch,
        IfInst_set_cond,
        IfInst_set_true_branch,
        IfInst_set_false_branch,
        IfInst_new,
        GenericLoopInst_prepare,
        GenericLoopInst_cond,
        GenericLoopInst_body,
        GenericLoopInst_update,
        GenericLoopInst_set_prepare,
        GenericLoopInst_set_cond,
        GenericLoopInst_set_body,
        GenericLoopInst_set_update,
        GenericLoopInst_new,
        SwitchInst_value,
        SwitchInst_cases,
        SwitchInst_default_,
        SwitchInst_set_value,
        SwitchInst_set_cases,
        SwitchInst_set_default_,
        SwitchInst_new,
        LocalInst_init,
        LocalInst_set_init,
        LocalInst_new,
        ReturnInst_value,
        ReturnInst_set_value,
        ReturnInst_new,
        PrintInst_fmt,
        PrintInst_args,
        PrintInst_set_fmt,
        PrintInst_set_args,
        PrintInst_new,
        CommentInst_comment,
        CommentInst_set_comment,
        CommentInst_new,
        UpdateInst_var,
        UpdateInst_value,
        UpdateInst_set_var,
        UpdateInst_set_value,
        UpdateInst_new,
        RayQueryInst_query,
        RayQueryInst_on_triangle_hit,
        RayQueryInst_on_procedural_hit,
        RayQueryInst_set_query,
        RayQueryInst_set_on_triangle_hit,
        RayQueryInst_set_on_procedural_hit,
        RayQueryInst_new,
        RevAutodiffInst_body,
        RevAutodiffInst_set_body,
        RevAutodiffInst_new,
        FwdAutodiffInst_body,
        FwdAutodiffInst_set_body,
        FwdAutodiffInst_new,
        Instruction_new,
        func_metadata,
        Binding_as_BufferBinding,
        Binding_as_TextureBinding,
        Binding_as_BindlessArrayBinding,
        Binding_as_AccelBinding,
        Binding_tag,
        BufferBinding_handle,
        BufferBinding_offset,
        BufferBinding_size,
        BufferBinding_set_handle,
        BufferBinding_set_offset,
        BufferBinding_set_size,
        BufferBinding_new,
        TextureBinding_handle,
        TextureBinding_level,
        TextureBinding_set_handle,
        TextureBinding_set_level,
        TextureBinding_new,
        BindlessArrayBinding_handle,
        BindlessArrayBinding_set_handle,
        BindlessArrayBinding_new,
        AccelBinding_handle,
        AccelBinding_set_handle,
        AccelBinding_new,
        Binding_new,
        ir_v2_binding_type_extract,
        ir_v2_binding_type_size,
        ir_v2_binding_type_alignment,
        ir_v2_binding_type_tag,
        ir_v2_binding_type_is_scalar,
        ir_v2_binding_type_is_bool,
        ir_v2_binding_type_is_int16,
        ir_v2_binding_type_is_int32,
        ir_v2_binding_type_is_int64,
        ir_v2_binding_type_is_uint16,
        ir_v2_binding_type_is_uint32,
        ir_v2_binding_type_is_uint64,
        ir_v2_binding_type_is_float16,
        ir_v2_binding_type_is_float32,
        ir_v2_binding_type_is_array,
        ir_v2_binding_type_is_vector,
        ir_v2_binding_type_is_struct,
        ir_v2_binding_type_is_custom,
        ir_v2_binding_type_is_matrix,
        ir_v2_binding_type_element,
        ir_v2_binding_type_description,
        ir_v2_binding_type_dimension,
        ir_v2_binding_type_members,
        ir_v2_binding_make_struct,
        ir_v2_binding_make_array,
        ir_v2_binding_make_vector,
        ir_v2_binding_make_matrix,
        ir_v2_binding_make_custom,
        ir_v2_binding_from_desc,
        ir_v2_binding_type_bool,
        ir_v2_binding_type_int16,
        ir_v2_binding_type_int32,
        ir_v2_binding_type_int64,
        ir_v2_binding_type_uint16,
        ir_v2_binding_type_uint32,
        ir_v2_binding_type_uint64,
        ir_v2_binding_type_float16,
        ir_v2_binding_type_float32,
        ir_v2_binding_node_prev,
        ir_v2_binding_node_next,
        ir_v2_binding_node_inst,
        ir_v2_binding_node_type,
        ir_v2_binding_node_get_index,
        ir_v2_binding_basic_block_first,
        ir_v2_binding_basic_block_last,
        ir_v2_binding_node_unlink,
        ir_v2_binding_node_set_next,
        ir_v2_binding_node_set_prev,
        ir_v2_binding_node_replace,
        ir_v2_binding_pool_new,
        ir_v2_binding_pool_drop,
        ir_v2_binding_pool_clone,
        ir_v2_binding_ir_builder_new,
        ir_v2_binding_ir_builder_new_without_bb,
        ir_v2_binding_ir_builder_drop,
        ir_v2_binding_ir_builder_set_insert_point,
        ir_v2_binding_ir_builder_insert_point,
        ir_v2_binding_ir_build_call,
        ir_v2_binding_ir_build_call_tag,
        ir_v2_binding_ir_build_if,
        ir_v2_binding_ir_build_generic_loop,
        ir_v2_binding_ir_build_switch,
        ir_v2_binding_ir_build_local,
        ir_v2_binding_ir_build_break,
        ir_v2_binding_ir_build_continue,
        ir_v2_binding_ir_build_return,
        ir_v2_binding_ir_builder_finish,
        ir_v2_binding_cpu_ext_fn_data,
        ir_v2_binding_cpu_ext_fn_new,
        ir_v2_binding_cpu_ext_fn_clone,
        ir_v2_binding_cpu_ext_fn_drop,
    };
}
}// namespace luisa::compute::ir_v2
