#pragma vengine_package vengine_database
#include <serialize/Config.h>

#include <serialize/Common.h>
#include <serialize/SimpleBinaryJson.h>
#include <serialize/SimpleJsonValue.h>

namespace toolhub::db {

class DictIEnumerator final : public vstd::IEnumerable<JsonKeyPair>, public vstd::IOperatorNewBase {
public:
    KVMap::Iterator ite;
    KVMap::Iterator end;
    DictIEnumerator(
        KVMap::Iterator &&ite,
        KVMap::Iterator &&end)
        : ite(ite),
          end(end) {}
    void Dispose() override {
        delete this;
    }
    JsonKeyPair GetValue() override {
        return JsonKeyPair{ite->first.GetKey(), ite->second.GetVariant()};
    };
    bool End() override {
        return ite == end;
    };
    void GetNext() override {
        ++ite;
    };
};

class ArrayIEnumerator final : public vstd::IEnumerable<ReadJsonVariant>, public vstd::IOperatorNewBase {
public:
    using BegType = decltype(std::declval<const luisa::vector<SimpleJsonVariant>>().begin());
    using EndType = decltype(std::declval<const luisa::vector<SimpleJsonVariant>>().end());
    BegType ite;
    EndType end;
    ArrayIEnumerator(
        BegType ite,
        EndType end)
        : ite(ite),
          end(end) {}
    ReadJsonVariant GetValue() override {
        return ite->GetVariant();
    };
    bool End() override {
        return ite == end;
    };
    void GetNext() override {
        ++ite;
    };
    void Dispose() override {
        delete this;
    }
};

struct BinaryHeader {
    uint64 preDefine;
    uint64 size;
    uint64 postDefine;
};

static void SerPreProcess(luisa::vector<uint8_t> &data) {
    data.resize(sizeof(uint64));
}

template<bool isDict>
static void SerPostProcess(luisa::vector<uint8_t> &data) {
    uint64 hashValue;
    vstd::hash<BinaryHeader> hasher;
    if constexpr (isDict) {
        hashValue = hasher(BinaryHeader{3551484578062615571ull, data.size(), 13190554206192427769ull});
    } else {
        hashValue = hasher(BinaryHeader{917074095154020627ull, data.size(), 12719994496415311585ull});
    }
    *reinterpret_cast<uint64 *>(data.data()) = hashValue;
}

template<bool isDict>
static bool DeserCheck(std::span<uint8_t const> &sp) {
    uint64 hashValue;
    vstd::hash<BinaryHeader> hasher;
    if constexpr (isDict) {
        hashValue = hasher(BinaryHeader{3551484578062615571ull, sp.size(), 13190554206192427769ull});
    } else {
        hashValue = hasher(BinaryHeader{917074095154020627ull, sp.size(), 12719994496415311585ull});
    }
    bool v = (hashValue == *reinterpret_cast<uint64 const *>(sp.data()));
    sp = sp.subspan(sizeof(uint64));
    return v;
}

static void PrintString(luisa::string const &str, luisa::string &result) {
    result += '\"';
    char const *last = str.c_str();
    auto Flush = [&](char const *ptr) {
        result.insert(result.size(), last, ptr - last);
        last = ptr + 1;
    };
    for (auto &&i : str) {
        switch (i) {
            case '\t':
                Flush(&i);
                result += "\\t";
                break;
            case '\r':
                Flush(&i);
                result += "\\r";
                break;
            case '\n':
                Flush(&i);
                result += "\\n";
                break;
            case '\'':
                Flush(&i);
                result += "\\\'";
                break;
            case '\"':
                Flush(&i);
                result += "\\\"";
                break;
        }
    }
    Flush(str.data() + str.size());
    result += '\"';
}

template<typename Dict, typename Array>
static void PrintSimpleJsonVariant(SimpleJsonVariant const &v, luisa::string &str, size_t layer, size_t valueLayer, bool emptySpaceBeforeOb) {
    auto func = [&](auto &&v) {
        str.insert(str.size(), ' ', valueLayer);
        str += std::to_string(v);
    };
    switch (v.value.GetType()) {
        case 0:
            func(v.value.get<0>());
            break;
        case 1:
            func(v.value.get<1>());
            break;
        case 2:
            [&](luisa::string const &s) {
                str.insert(str.size(), ' ', valueLayer);
                PrintString(s, str);
            }(v.value.get<2>());
            break;
        case 3:
            [&](UniquePtr<IJsonDict> const &ptr) {
                if (emptySpaceBeforeOb)
                    str += '\n';
                static_cast<Dict *>(ptr.get())->M_Print(str, layer);
            }(v.value.get<3>());
            break;
        case 4:
            [&](UniquePtr<IJsonArray> const &ptr) {
                if (emptySpaceBeforeOb)
                    str += '\n';
                static_cast<Array *>(ptr.get())->M_Print(str, layer);
            }(v.value.get<4>());
            break;
        case 5:
            [&](vstd::Guid const &guid) {
                str.insert(str.size(), ' ', valueLayer);
                str += '$';
                size_t offst = str.size();
                str.resize(offst + 32);
                guid.ToString(str.data() + offst, false);
            }(v.value.get<5>());
            break;
    }
}

static void PrintKeyVariant(SimpleJsonKey const &v, luisa::string &str) {
    auto func = [&](auto &&v) {
        str += std::to_string(v);
    };
    switch (v.value.GetType()) {
        case 0:
            str += std::to_string(v.value.get<0>());
            break;
        case 1:
            PrintString(v.value.get<1>(), str);
            break;
        case 2:
            [&](vstd::Guid const &guid) {
                str += '$';
                size_t offst = str.size();
                str.resize(offst + 32);
                guid.ToString(str.data() + offst, false);
            }(v.value.get<2>());
            break;
    }
}

template<typename Dict, typename Array>
static void PrintDict(KVMap &vars, luisa::string &str, size_t space) {
    str.append(space, ' ');
    str += "{\n";
    space += 2;
    auto disp = vstd::create_disposer([&]() {
        space -= 2;
        str.append(space, ' ');
        str += '}';
    });
    size_t varsSize = vars.size() - 1;
    size_t index = 0;
    for (auto &&i : vars) {
        str.append(space, ' ');
        PrintKeyVariant(i.first, str);
        str += " : ";
        PrintSimpleJsonVariant<Dict, Array>(i.second, str, space, 0, true);
        if (index == varsSize) {
            str += '\n';
        } else {
            str += ",\n";
        }
        index++;
    }
}

template<typename Dict, typename Array>
static void PrintArray(luisa::vector<SimpleJsonVariant> &arr, luisa::string &str, size_t space) {
    str.append(space, ' ');
    str += "[\n";
    space += 2;
    auto disp = vstd::create_disposer([&]() {
        space -= 2;
        str.append(space, ' ');
        str += ']';
    });
    size_t arrSize = arr.size() - 1;
    size_t index = 0;
    for (auto &&i : arr) {
        PrintSimpleJsonVariant<Dict, Array>(i, str, space, space, false);
        if (index == arrSize) {
            str += '\n';
        } else {
            str += ",\n";
        }
        index++;
    }
}

//////////////////////////  Single Thread
SimpleJsonValueDict::SimpleJsonValueDict(SimpleBinaryJson *db)
    : SimpleJsonValue{db} {}

SimpleJsonValueDict::~SimpleJsonValueDict() = default;

ReadJsonVariant SimpleJsonValueDict::Get(Key const &key) {
    if (!key.valid())
        return {};
    auto ite = vars.Find(key);
    if (ite)
        return ite.Value().GetVariant();
    return {};
}

WriteJsonVariant SimpleJsonValueDict::GetAndSet(Key const &key, WriteJsonVariant &&newValue) {
    if (!key.valid())
        return {};
    auto ite = vars.Find(key);
    if (ite) {
        auto result = std::move(ite.Value().value);
        if (newValue.valid()) {
            ite.Value().value = std::move(newValue);
        } else {
            vars.Remove(ite);
        }
        return result;
    } else {
        if (newValue.valid()) {
            vars.ForceEmplace(key, std::move(newValue));
        }
        return {};
    }
}

WriteJsonVariant SimpleJsonValueDict::GetAndRemove(Key const &key) {
    if (!key.valid())
        return {};
    auto ite = vars.Find(key);
    if (ite) {
        auto result = std::move(ite.Value().value);
        vars.Remove(ite);
        return result;
    } else
        return {};
}

void SimpleJsonValueDict::Set(Key const &key, WriteJsonVariant &&value) {
    if (key.valid() && value.valid()) {
        vars.ForceEmplace(key, std::move(value));
    }
}

bool SimpleJsonValueDict::TrySet(Key const &key, WriteJsonVariant &&value) {
    if (key.valid() && value.valid()) {
        return vars.TryEmplace(key, std::move(value)).second;
    }
    return false;
}

void SimpleJsonValueDict::Remove(Key const &key) {
    if (key.valid()) {
        vars.TRemove(key);
    }
}

size_t SimpleJsonValueDict::Length() {
    return vars.size();
}

luisa::vector<uint8_t> SimpleJsonValueDict::Serialize() {
    luisa::vector<uint8_t> result;
    SerPreProcess(result);
    M_GetSerData(result);
    SerPostProcess<true>(result);
    return result;
}

void SimpleJsonValueDict::M_GetSerData(luisa::vector<uint8_t> &data) {
    PushDataToVector<uint64>(vars.size(), data);
    for (auto &&kv : vars) {
        PushDataToVector(kv.first.value, data);
        SimpleJsonLoader::Serialize(kv.second, data);
    }
}

void SimpleJsonValueDict::LoadFromSer(std::span<uint8_t const> &sp) {
    auto sz = PopValue<uint64>(sp);
    vars.reserve(sz);
    for (auto i : vstd::range(static_cast<int64_t>(sz))) {
        auto key = PopValue<SimpleJsonKey::ValueType>(sp);
        vars.Emplace(std::move(key), SimpleJsonLoader::DeSerialize(sp, db));
    }
}

void SimpleJsonValueDict::Reset() {
    vars.Clear();
}

void SimpleJsonValueDict::Dispose() {
    db->dictValuePool.Delete(this);
}

void SimpleJsonValueArray::Dispose() {
    db->arrValuePool.Delete(this);
}

SimpleJsonValueArray::SimpleJsonValueArray(
    SimpleBinaryJson *db) : SimpleJsonValue{db} {}

SimpleJsonValueArray::~SimpleJsonValueArray() = default;

size_t SimpleJsonValueArray::Length() {
    return arr.size();
}

luisa::vector<uint8_t> SimpleJsonValueArray::Serialize() {
    luisa::vector<uint8_t> result;
    SerPreProcess(result);
    M_GetSerData(result);
    SerPostProcess<false>(result);
    return result;
}

void SimpleJsonValueArray::M_GetSerData(luisa::vector<uint8_t> &data) {
    PushDataToVector<uint64>(arr.size(), data);
    for (auto &&v : arr) {
        SimpleJsonLoader::Serialize(v, data);
    }
}

void SimpleJsonValueArray::LoadFromSer(std::span<uint8_t const> &sp) {
    auto sz = PopValue<uint64>(sp);
    arr.reserve(sz);
    for (auto i : vstd::range(static_cast<int64_t>(sz))) {
        arr.emplace_back(SimpleJsonLoader::DeSerialize(sp, db));
    }
}

void SimpleJsonValueArray::Reset() {
    arr.clear();
}

ReadJsonVariant SimpleJsonValueArray::Get(size_t index) {
    if (index >= arr.size())
        return {};
    return arr[index].GetVariant();
}

void SimpleJsonValueArray::Set(size_t index, WriteJsonVariant &&value) {
    if (index < arr.size()) {
        if (value.valid())
            arr[index].Set(std::move(value));
        else
            arr[index].value.dispose();
    }
}

void SimpleJsonValueArray::Remove(size_t index) {
    if (index < arr.size()) {
        arr.erase(arr.begin() + index);
    }
}

void SimpleJsonValueArray::Add(WriteJsonVariant &&value) {
    if (value.valid()) {
        arr.emplace_back(std::move(value));
    }
}

WriteJsonVariant SimpleJsonValueArray::GetAndSet(size_t index, WriteJsonVariant &&newValue) {
    if (index >= arr.size())
        return {};
    WriteJsonVariant result = std::move(arr[index].value);
    if (newValue.valid())
        arr[index] = std::move(newValue);
    else
        arr[index].value.dispose();
    return result;
}
WriteJsonVariant SimpleJsonValueArray::GetAndRemove(size_t index) {
    if (index >= arr.size())
        return {};
    WriteJsonVariant result = std::move(arr[index].value);
    arr.erase(arr.begin() + index);
    return result;
}

void SimpleJsonValueDict::M_Print(luisa::string &str, size_t space) {
    PrintDict<SimpleJsonValueDict, SimpleJsonValueArray>(vars, str, space);
}

void SimpleJsonValueArray::M_Print(luisa::string &str, size_t space) {
    PrintArray<SimpleJsonValueDict, SimpleJsonValueArray>(arr, str, space);
}

//////////////////////////  Multi-Thread
ConcurrentJsonValueDict::ConcurrentJsonValueDict(ConcurrentBinaryJson *db) {
    this->db = db;
}

ConcurrentJsonValueDict::~ConcurrentJsonValueDict() = default;

ReadJsonVariant ConcurrentJsonValueDict::Get(Key const &key) {
    if (!key.valid())
        return {};
    auto ite = vars.Find(key);
    if (ite)
        return ite.Value().GetVariant_Concurrent();
    return {};
}

WriteJsonVariant ConcurrentJsonValueDict::GetAndSet(Key const &key, WriteJsonVariant &&newValue) {
    if (!key.valid())
        return {};
    std::lock_guard lck(mtx);
    auto ite = vars.Find(key);
    if (ite) {
        auto result = std::move(ite.Value().value);
        if (newValue.valid()) {
            ite.Value().value = std::move(newValue);
        } else {
            vars.Remove(ite);
        }
        return result;
    } else {
        if (newValue.valid()) {
            vars.ForceEmplace(key, std::move(newValue));
        }
        return {};
    }
}

WriteJsonVariant ConcurrentJsonValueDict::GetAndRemove(Key const &key) {
    if (!key.valid())
        return {};
    std::lock_guard lck(mtx);
    auto ite = vars.Find(key);
    if (ite) {
        auto result = std::move(ite.Value().value);
        vars.Remove(ite);
        return result;
    } else
        return {};
}
void ConcurrentJsonValueDict::Set(Key const &key, WriteJsonVariant &&value) {
    if (key.valid() && value.valid()) {
        vars.ForceEmplace_Lock(mtx, key, std::move(value));
    }
}

bool ConcurrentJsonValueDict::TrySet(Key const &key, WriteJsonVariant &&value) {
    if (key.valid() && value.valid()) {
        return vars.TryEmplace_Lock(mtx, key, std::move(value)).second;
    }
    return false;
}

void ConcurrentJsonValueDict::Remove(Key const &key) {
    if (key.valid()) {
        vars.TRemove_Lock(mtx, key);
    }
}

size_t ConcurrentJsonValueDict::Length() {
    return vars.size();
}

luisa::vector<uint8_t> ConcurrentJsonValueDict::Serialize() {
    luisa::vector<uint8_t> result;
    SerPreProcess(result);
    M_GetSerData(result);
    SerPostProcess<true>(result);
    return result;
}

void ConcurrentJsonValueDict::M_GetSerData(luisa::vector<uint8_t> &data) {
    std::lock_guard lck(mtx);
    PushDataToVector<uint64>(vars.size(), data);
    for (auto &&kv : vars) {
        PushDataToVector(kv.first.value, data);
        SimpleJsonLoader::Serialize_Concurrent(kv.second, data);
    }
}

void ConcurrentJsonValueDict::LoadFromSer(std::span<uint8_t const> &sp) {
    auto sz = PopValue<uint64>(sp);
    std::lock_guard lck(mtx);
    vars.reserve(sz);
    for (auto i : vstd::range(static_cast<int64_t>(sz))) {
        auto key = PopValue<SimpleJsonKey::ValueType>(sp);
        vars.Emplace(std::move(key), SimpleJsonLoader::DeSerialize_Concurrent(sp, db));
    }
}

void ConcurrentJsonValueDict::Reset() {
    std::lock_guard lck(mtx);
    vars.Clear();
}

void ConcurrentJsonValueDict::Dispose() {
    db->dictValuePool.Delete_Lock(db->dictPoolMtx, this);
}

void ConcurrentJsonValueArray::Dispose() {
    db->arrValuePool.Delete_Lock(db->arrPoolMtx, this);
}

ConcurrentJsonValueArray::ConcurrentJsonValueArray(
    ConcurrentBinaryJson *db) {
    this->db = db;
}

ConcurrentJsonValueArray::~ConcurrentJsonValueArray() = default;

size_t ConcurrentJsonValueArray::Length() {
    return arr.size();
}

luisa::vector<uint8_t> ConcurrentJsonValueArray::Serialize() {
    luisa::vector<uint8_t> result;
    SerPreProcess(result);
    M_GetSerData(result);
    SerPostProcess<false>(result);
    return result;
}
void ConcurrentJsonValueArray::M_GetSerData(luisa::vector<uint8_t> &data) {
    std::lock_guard lck(mtx);
    PushDataToVector<uint64>(arr.size(), data);
    for (auto &&v : arr) {
        SimpleJsonLoader::Serialize_Concurrent(v, data);
    }
}

void ConcurrentJsonValueArray::LoadFromSer(std::span<uint8_t const> &sp) {
    auto sz = PopValue<uint64>(sp);
    std::lock_guard lck(mtx);
    arr.reserve(sz);
    for (auto i : vstd::range(static_cast<int64_t>(sz))) {
        arr.emplace_back(SimpleJsonLoader::DeSerialize_Concurrent(sp, db));
    }
}

void ConcurrentJsonValueArray::Reset() {
    std::lock_guard lck(mtx);
    arr.clear();
}

ReadJsonVariant ConcurrentJsonValueArray::Get(size_t index) {
    if (index >= arr.size())
        return {};
    std::lock_guard lck(mtx);
    return arr[index].GetVariant_Concurrent();
}

void ConcurrentJsonValueArray::Set(size_t index, WriteJsonVariant &&value) {
    std::lock_guard lck(mtx);
    if (index < arr.size()) {
        if (value.valid())
            arr[index].Set(std::move(value));
        else
            arr[index].value.dispose();
    }
}

void ConcurrentJsonValueArray::Remove(size_t index) {
    std::lock_guard lck(mtx);
    if (index < arr.size()) {
        arr.erase(arr.begin() + index);
    }
}

void ConcurrentJsonValueArray::Add(WriteJsonVariant &&value) {
    if (value.valid()) {
        std::lock_guard lck(mtx);
        arr.emplace_back(std::move(value));
    }
}

WriteJsonVariant ConcurrentJsonValueArray::GetAndSet(size_t index, WriteJsonVariant &&newValue) {
    std::lock_guard lck(mtx);
    if (index >= arr.size())
        return {};
    WriteJsonVariant result = std::move(arr[index].value);
    if (newValue.valid())
        arr[index] = std::move(newValue);
    else
        arr[index].value.dispose();
    return result;
}

WriteJsonVariant ConcurrentJsonValueArray::GetAndRemove(size_t index) {
    std::lock_guard lck(mtx);
    if (index >= arr.size())
        return {};
    WriteJsonVariant result = std::move(arr[index].value);
    arr.erase(arr.begin() + index);
    return result;
}

void ConcurrentJsonValueDict::M_Print(luisa::string &str, size_t space) {
    std::lock_guard lck(mtx);
    PrintDict<ConcurrentJsonValueDict, ConcurrentJsonValueArray>(vars, str, space);
}

void ConcurrentJsonValueArray::M_Print(luisa::string &str, size_t space) {
    std::lock_guard lck(mtx);
    PrintArray<ConcurrentJsonValueDict, ConcurrentJsonValueArray>(arr, str, space);
}

vstd::MD5 SimpleJsonValueDict::GetMD5() {
    luisa::vector<uint8_t> vec;
    M_GetSerData(vec);
    return {vec};
}

vstd::MD5 SimpleJsonValueArray::GetMD5() {
    luisa::vector<uint8_t> vec;
    M_GetSerData(vec);
    return {vec};
}

vstd::MD5 ConcurrentJsonValueDict::GetMD5() {
    luisa::vector<uint8_t> vec;
    M_GetSerData(vec);
    return {vec};
}

vstd::MD5 ConcurrentJsonValueArray::GetMD5() {
    luisa::vector<uint8_t> vec;
    M_GetSerData(vec);
    return {vec};
}

bool SimpleJsonValueDict::Read(std::span<uint8_t const> sp, bool clearLast) {
    if (!DeserCheck<true>(sp))
        return false;
    if (clearLast) {
        vars.Clear();
    }
    LoadFromSer(sp);
    return true;
}

bool SimpleJsonValueArray::Read(std::span<uint8_t const> sp, bool clearLast) {
    if (!DeserCheck<false>(sp))
        return false;
    if (clearLast) {
        arr.clear();
    }
    LoadFromSer(sp);
    return true;
}

bool ConcurrentJsonValueDict::Read(std::span<uint8_t const> sp, bool clearLast) {
    if (!DeserCheck<true>(sp))
        return false;
    if (clearLast) {
        vars.Clear();
    }
    LoadFromSer(sp);
    return true;
}

bool ConcurrentJsonValueArray::Read(std::span<uint8_t const> sp, bool clearLast) {
    if (!DeserCheck<false>(sp))
        return false;
    if (clearLast) {
        arr.clear();
    }
    LoadFromSer(sp);
    return true;
}

vstd::Iterator<JsonKeyPair> SimpleJsonValueDict::begin() const {
    return new DictIEnumerator(vars.begin(), vars.end());
}

vstd::Iterator<JsonKeyPair> ConcurrentJsonValueDict::begin() const {
    return new DictIEnumerator(vars.begin(), vars.end());
}

vstd::Iterator<ReadJsonVariant> ConcurrentJsonValueArray::begin() const {
    return new ArrayIEnumerator(arr.begin(), arr.end());
}

vstd::Iterator<ReadJsonVariant> SimpleJsonValueArray::begin() const {
    return new ArrayIEnumerator(arr.begin(), arr.end());
}

}// namespace toolhub::db