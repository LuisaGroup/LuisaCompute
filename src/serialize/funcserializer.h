#pragma once
#include <serde/IJsonDatabase.h>
#include <ast/function_builder.h>

namespace luisa::compute {
class FuncSerializer {
private:
    static vstd::unique_ptr<IJsonDict> GetBuilderSerFunc(detail::FunctionBuilder const *b, IJsonDatabase *db);
    static detail::FunctionBuilder const *GetBuilderSerFunc(IJsonDict *dict);

public:
    static vstd::unique_ptr<IJsonDict> SerFunc(Function func, IJsonDatabase *db);
    static Function DeserFunc(IJsonDict *dict);
};
}// namespace luisa::compute