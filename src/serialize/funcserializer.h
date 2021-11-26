#pragma once
#include <serde/IJsonDatabase.h>
#include <ast/function_builder.h>

namespace luisa::compute {
using namespace toolhub::db;
class FuncSerializer {
private:
    using FuncMap = vstd::HashMap<uint64, std::pair<IJsonDict *, std::shared_ptr<detail::FunctionBuilder>>>;
    static vstd::unique_ptr<IJsonDict> GetBuilderSerFunc(detail::FunctionBuilder const *b, IJsonDatabase *db);
    static void GetBuilderDeserFunc(IJsonDict *dict, detail::FunctionBuilder *builder, FuncMap &map);

public:
    static vstd::unique_ptr<IJsonArray> SerKernel(Function func, IJsonDatabase *db);
    static Function DeserKernel(IJsonArray *arr);
};
class ISerializer {
public:
    virtual vstd::unique_ptr<IJsonArray> SerKernel(Function func, IJsonDatabase *db) const = 0;
    virtual Function DeserKernel(IJsonArray *arr) const = 0;
};
// Entry:
// ISerializer const *Serialize_GetFactory();
#ifdef LUISA_SERIALIZE_PROJECT
class Serializer_Impl : public ISerializer {
public:
    vstd::unique_ptr<IJsonArray> SerKernel(Function func, IJsonDatabase *db) const override;
    Function DeserKernel(IJsonArray *arr) const override;
};
#endif
}// namespace luisa::compute