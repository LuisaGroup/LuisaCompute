
#include <Codegen/ArgumentPacker.h>
#include <vstl/Serializer.h>
namespace toolhub::directx {
void ArgumentPacker::PackData(
    vstd::Iterator<std::pair<Variable const *, vstd::span<vbyte const>>> const &allVars,
    uint3 dispatchSize,
    vstd::vector<vbyte> &data) {
    auto ForceAlign = [&](size_t sz, vstd::span<vbyte const> &sp) {
        auto cbSize = data.size();
        cbSize = (cbSize + (sz - 1)) & ~(sz - 1);
        data.resize(cbSize);
        size_t ptr = reinterpret_cast<size_t>(sp.data());
        size_t alignedPtr = ((ptr + (sz - 1)) & ~(sz - 1)) - ptr;
        sp = {sp.data() + alignedPtr, sp.size() - alignedPtr};
    };
    auto PushSpan = [&](vstd::span<vbyte const> &sp, size_t size) {
        data.push_back_all(sp.data(), size);
        sp = vstd::span<vbyte const>{sp.data() + size, sp.size() - size};
    };
    struct LocalFloat4 {
        float4 f;
        LocalFloat4(float3 v)
            : f(v.x, v.y, v.z, 0) {}
    };
    struct Float4x3 {
        LocalFloat4 cols[3];
    };
    auto GetNext = [&](auto &&GetNextArray,
                       auto &&GetNextStruct,
                       auto &&GetNextVector,
                       Type const &tp, vstd::span<vbyte const> &sp) -> void {
        switch (tp.tag()) {
            case Type::Tag::BOOL:
                PushSpan(sp, 1);
                break;
            case Type::Tag::INT:
            case Type::Tag::UINT:
            case Type::Tag::FLOAT:
                ForceAlign(4, sp);
                assert(sp.size() >= 4);
                PushSpan(sp, 4);
                break;
            case Type::Tag::MATRIX:
                switch (tp.dimension()) {
                    case 2:
                        ForceAlign(8, sp);
                        PushSpan(sp, sizeof(float) * 4);
                        break;
                    case 3: {
                        ForceAlign(16, sp);
                        PushSpan(sp, sizeof(float) * 3 * 4);
                    } break;
                    case 4:
                        ForceAlign(16, sp);
                        PushSpan(sp, sizeof(float) * 4 * 4);
                        break;
                }
                break;
            case Type::Tag::ARRAY:
                GetNextArray(
                    GetNextArray,
                    tp,
                    sp);
                break;
            case Type::Tag::STRUCTURE:
                GetNextStruct(
                    GetNextStruct,
                    GetNextArray,
                    tp,
                    sp);
                break;
            case Type::Tag::VECTOR:
                GetNextVector(
                    GetNextStruct,
                    GetNextArray,
                    tp,
                    sp);
                break;
        }
    };
    auto GetNextVector = [&](auto &&GetNextStruct,
                             auto &&GetNextArray,
                             Type const &t, vstd::span<vbyte const> &sp) -> void {
        auto &&ele = t.element();
        float alignArray[] = {1, 2, 4, 4};
        switch (ele->tag()) {
            case Type::Tag::INT:
            case Type::Tag::UINT:
            case Type::Tag::FLOAT: {
                ForceAlign(alignArray[t.dimension()] * 4, sp);
                auto sz = 4 * ele->dimension();
                assert(sz >= sp.size());
                PushSpan(sp, sz);
            } break;
            case Type::Tag::BOOL: {
                ForceAlign(alignArray[t.dimension()], sp);
                auto sz = ele->dimension();
                PushSpan(sp, t.dimension());
            } break;
        }
    };
    auto GetNextStruct = [&](auto &&GetNextStruct,
                             auto &&GetNextArray,
                             Type const &t,
                             vstd::span<vbyte const> &sp) -> void {
        ForceAlign(16, sp);
        auto &&members = t.members();
        for (auto &&i : members) {
            GetNext(
                GetNextArray,
                GetNextStruct,
                GetNextVector,
                *i,
                sp);
        }
    };
    auto GetNextArray = [&](auto &&GetNextArray,
                            Type const &t,
                            vstd::span<vbyte const> &sp) -> void {
        ForceAlign(16, sp);
        auto &&ele = t.element();
        for (auto i : vstd::range(t.dimension())) {
            GetNext(
                GetNextArray,
                GetNextStruct,
                GetNextVector,
                t, sp);
        }
    };
    vstd::SerDe<uint3>::Set(dispatchSize, data);
    data.resize(16);
    for (; allVars; ++allVars) {
        auto &&kv = *allVars;
        GetNext(
            GetNextArray,
            GetNextStruct,
            GetNextVector,
            *kv.first->type(),
            kv.second);
    }
}
}// namespace toolhub::directx