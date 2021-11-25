#pragma once
#include <serde/IJsonDatabase.h>
#include <ast/function_builder.h>

namespace luisa::compute {
class FuncSerializer {
private:
    using FuncMap = vstd::HashMap<uint64, std::pair<IJsonDict*, std::shared_ptr<detail::FunctionBuilder>>>;
    static vstd::unique_ptr<IJsonDict> GetBuilderSerFunc(detail::FunctionBuilder const *b, IJsonDatabase *db);
    static void GetBuilderDeserFunc(IJsonDict *dict, detail::FunctionBuilder* builder, FuncMap &map);

public:
    static vstd::unique_ptr<IJsonArray> SerKernel(Function func, IJsonDatabase *db);
    static Function DeserKernel(IJsonArray *arr);
};
}// namespace luisa::compute