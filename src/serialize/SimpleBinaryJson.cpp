#pragma vengine_package vengine_database
#include <serialize/DatabaseInclude.h>
#include <serialize/SimpleBinaryJson.h>
namespace toolhub::db {
class Database_Impl final : public Database {
public:
    IJsonDatabase *CreateDatabase() const override;
    IJsonDatabase *CreateConcurrentDatabase() const override;
};
////////////////// Single Thread DB
SimpleBinaryJson::SimpleBinaryJson()
    : arrValuePool(32, false), dictValuePool(32, false) {
    root.New(this);
}
std::vector<uint8_t> SimpleBinaryJson::Serialize() {
    return root->Serialize();
}
bool SimpleBinaryJson::Read(
    std::span<uint8_t const> data,
    bool clearLast) {
    return root->Read(data, clearLast);
}
std::string SimpleBinaryJson::Print() {
    return root->Print();
}
IJsonDict *SimpleBinaryJson::GetRootNode() {
    return root;
}
SimpleBinaryJson ::~SimpleBinaryJson() {
    root.Delete();
}
UniquePtr<IJsonDict> SimpleBinaryJson::CreateDict() {
    return MakeUnique<IJsonDict>(dictValuePool.New(this));
}
UniquePtr<IJsonArray> SimpleBinaryJson::CreateArray() {
    return MakeUnique<IJsonArray>(arrValuePool.New(this));
}
SimpleJsonValueDict *SimpleBinaryJson::CreateDict_Nake() {
    return dictValuePool.New(this);
}
SimpleJsonValueArray *SimpleBinaryJson::CreateArray_Nake() {
    return arrValuePool.New(this);
}
////////////////// Multithread DB
ConcurrentBinaryJson::ConcurrentBinaryJson()
    : arrValuePool(32, false), dictValuePool(32, false) {
    root.New(this);
}
std::vector<uint8_t> ConcurrentBinaryJson::Serialize() {
    return root->Serialize();
}
bool ConcurrentBinaryJson::Read(
    std::span<uint8_t const> data,
    bool clearLast) {
    return root->Read(data, clearLast);
}
std::string ConcurrentBinaryJson::Print() {
    return root->Print();
}
IJsonDict *ConcurrentBinaryJson::GetRootNode() {
    return root;
}
ConcurrentBinaryJson ::~ConcurrentBinaryJson() {
    root.Delete();
}
UniquePtr<IJsonDict> ConcurrentBinaryJson::CreateDict() {
    return MakeUnique<IJsonDict>(dictValuePool.New_Lock(dictPoolMtx, this));
}
UniquePtr<IJsonArray> ConcurrentBinaryJson::CreateArray() {
    return MakeUnique<IJsonArray>(arrValuePool.New_Lock(arrPoolMtx, this));
}
ConcurrentJsonValueDict *ConcurrentBinaryJson::CreateDict_Nake() {
    return dictValuePool.New_Lock(dictPoolMtx, this);
}
ConcurrentJsonValueArray *ConcurrentBinaryJson::CreateArray_Nake() {
    return arrValuePool.New_Lock(arrPoolMtx, this);
}
vstd::MD5 ConcurrentBinaryJson::GetMD5() {
    return root->GetMD5();
}
vstd::MD5 SimpleBinaryJson::GetMD5() {
    return root->GetMD5();
}
IJsonDatabase *Database_Impl::CreateDatabase() const {
    return new SimpleBinaryJson();
}
IJsonDatabase *Database_Impl::CreateConcurrentDatabase() const {
    return new ConcurrentBinaryJson();
}

static vstd::optional<Database_Impl> database_Impl;
LUISA_EXPORT toolhub::db::Database const *Database_GetFactory() {
    database_Impl.New();
    return database_Impl;
}
}// namespace toolhub::db