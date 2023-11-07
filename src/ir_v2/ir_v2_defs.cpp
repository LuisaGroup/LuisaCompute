#include <luisa/ir_v2/ir_v2.h>
namespace luisa::compute::ir_v2 {
Func::Func(AssumeFn v) : _data(luisa::make_unique<AssumeFn>(std::move(v))), _tag(AssumeFn::static_tag()) {}
Func::Func(UnreachableFn v) : _data(luisa::make_unique<UnreachableFn>(std::move(v))), _tag(UnreachableFn::static_tag()) {}
Func::Func(AssertFn v) : _data(luisa::make_unique<AssertFn>(std::move(v))), _tag(AssertFn::static_tag()) {}
Func::Func(BindlessAtomicExchangeFn v) : _data(luisa::make_unique<BindlessAtomicExchangeFn>(std::move(v))), _tag(BindlessAtomicExchangeFn::static_tag()) {}
Func::Func(BindlessAtomicCompareExchangeFn v) : _data(luisa::make_unique<BindlessAtomicCompareExchangeFn>(std::move(v))), _tag(BindlessAtomicCompareExchangeFn::static_tag()) {}
Func::Func(BindlessAtomicFetchAddFn v) : _data(luisa::make_unique<BindlessAtomicFetchAddFn>(std::move(v))), _tag(BindlessAtomicFetchAddFn::static_tag()) {}
Func::Func(BindlessAtomicFetchSubFn v) : _data(luisa::make_unique<BindlessAtomicFetchSubFn>(std::move(v))), _tag(BindlessAtomicFetchSubFn::static_tag()) {}
Func::Func(BindlessAtomicFetchAndFn v) : _data(luisa::make_unique<BindlessAtomicFetchAndFn>(std::move(v))), _tag(BindlessAtomicFetchAndFn::static_tag()) {}
Func::Func(BindlessAtomicFetchOrFn v) : _data(luisa::make_unique<BindlessAtomicFetchOrFn>(std::move(v))), _tag(BindlessAtomicFetchOrFn::static_tag()) {}
Func::Func(BindlessAtomicFetchXorFn v) : _data(luisa::make_unique<BindlessAtomicFetchXorFn>(std::move(v))), _tag(BindlessAtomicFetchXorFn::static_tag()) {}
Func::Func(BindlessAtomicFetchMinFn v) : _data(luisa::make_unique<BindlessAtomicFetchMinFn>(std::move(v))), _tag(BindlessAtomicFetchMinFn::static_tag()) {}
Func::Func(BindlessAtomicFetchMaxFn v) : _data(luisa::make_unique<BindlessAtomicFetchMaxFn>(std::move(v))), _tag(BindlessAtomicFetchMaxFn::static_tag()) {}
Func::Func(CallableFn v) : _data(luisa::make_unique<CallableFn>(std::move(v))), _tag(CallableFn::static_tag()) {}
Func::Func(CpuExtFn v) : _data(luisa::make_unique<CpuExtFn>(std::move(v))), _tag(CpuExtFn::static_tag()) {}
Instruction::Instruction(ArgumentInst v) : _data(luisa::make_unique<ArgumentInst>(std::move(v))), _tag(ArgumentInst::static_tag()) {}
Instruction::Instruction(ConstantInst v) : _data(luisa::make_unique<ConstantInst>(std::move(v))), _tag(ConstantInst::static_tag()) {}
Instruction::Instruction(CallInst v) : _data(luisa::make_unique<CallInst>(std::move(v))), _tag(CallInst::static_tag()) {}
Instruction::Instruction(PhiInst v) : _data(luisa::make_unique<PhiInst>(std::move(v))), _tag(PhiInst::static_tag()) {}
Instruction::Instruction(IfInst v) : _data(luisa::make_unique<IfInst>(std::move(v))), _tag(IfInst::static_tag()) {}
Instruction::Instruction(GenericLoopInst v) : _data(luisa::make_unique<GenericLoopInst>(std::move(v))), _tag(GenericLoopInst::static_tag()) {}
Instruction::Instruction(SwitchInst v) : _data(luisa::make_unique<SwitchInst>(std::move(v))), _tag(SwitchInst::static_tag()) {}
Instruction::Instruction(LocalInst v) : _data(luisa::make_unique<LocalInst>(std::move(v))), _tag(LocalInst::static_tag()) {}
Instruction::Instruction(ReturnInst v) : _data(luisa::make_unique<ReturnInst>(std::move(v))), _tag(ReturnInst::static_tag()) {}
Instruction::Instruction(PrintInst v) : _data(luisa::make_unique<PrintInst>(std::move(v))), _tag(PrintInst::static_tag()) {}
Instruction::Instruction(CommentInst v) : _data(luisa::make_unique<CommentInst>(std::move(v))), _tag(CommentInst::static_tag()) {}
Instruction::Instruction(UpdateInst v) : _data(luisa::make_unique<UpdateInst>(std::move(v))), _tag(UpdateInst::static_tag()) {}
Instruction::Instruction(RayQueryInst v) : _data(luisa::make_unique<RayQueryInst>(std::move(v))), _tag(RayQueryInst::static_tag()) {}
Instruction::Instruction(RevAutodiffInst v) : _data(luisa::make_unique<RevAutodiffInst>(std::move(v))), _tag(RevAutodiffInst::static_tag()) {}
Instruction::Instruction(FwdAutodiffInst v) : _data(luisa::make_unique<FwdAutodiffInst>(std::move(v))), _tag(FwdAutodiffInst::static_tag()) {}
const FuncMetadata &Func::metadata() const noexcept {
    return func_metadata()[static_cast<int>(tag())];
}
Binding::Binding(BufferBinding v) : _data(luisa::make_unique<BufferBinding>(std::move(v))), _tag(BufferBinding::static_tag()) {}
Binding::Binding(TextureBinding v) : _data(luisa::make_unique<TextureBinding>(std::move(v))), _tag(TextureBinding::static_tag()) {}
Binding::Binding(BindlessArrayBinding v) : _data(luisa::make_unique<BindlessArrayBinding>(std::move(v))), _tag(BindlessArrayBinding::static_tag()) {}
Binding::Binding(AccelBinding v) : _data(luisa::make_unique<AccelBinding>(std::move(v))), _tag(AccelBinding::static_tag()) {}
}// namespace luisa::compute::ir_v2
