#pragma vengine_package vengine_directx
#include <Codegen/ArgumentPacker.h>
#include <vstl/Serializer.h>
namespace toolhub::directx {
void ArgumentPacker::PackData(
    vstd::Iterator<std::pair<Variable const *, vstd::span<vbyte const>>> const &allVars,
    uint3 dispatchSize,
    vstd::vector<vbyte> &data) {
    size_t cbSize = 0;
    auto CalcAlign = [&](size_t nextSize) {
        auto leftSize = cbSize & 15;
        if (leftSize + nextSize > 16) {
            cbSize += (16 - leftSize);
            data.resize(cbSize);
        }
    };
    auto ForceAlign = [&]() {
        cbSize = (cbSize + 15) & ~15;
        data.resize(cbSize);
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
                vstd::SerDe<uint>::Set(sp[0] != 0 ? 1 : 0, data);
                cbSize += 4;
                break;
            case Type::Tag::INT:
            case Type::Tag::UINT:
            case Type::Tag::FLOAT:
                assert(sp.size() >= 4);
                PushSpan(sp, 4);
                cbSize += 4;
                break;
            case Type::Tag::MATRIX:
                if (tp.dimension() == 4) {
                    data.push_back_all(sp);
                } else {
                    auto matData = vstd::SerDe<float3x3>::Get(sp);
                    Float4x3 newMatData{
                        .cols = {
                            LocalFloat4(matData[0]),
                            LocalFloat4{matData[1]},
                            LocalFloat4{matData[2]}}};
                    vstd::SerDe<Float4x3>::Set(newMatData, data);
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
        switch (ele->tag()) {
            case Type::Tag::INT:
            case Type::Tag::UINT:
            case Type::Tag::FLOAT: {
                auto sz = 4 * ele->dimension();
                CalcAlign(sz);
                assert(sz >= sp.size());
                PushSpan(sp, sz);
                cbSize += sz;
            } break;
            case Type::Tag::BOOL: {
                auto sz = 4 * ele->dimension();
                CalcAlign(sz);
                for (auto &&i : sp) {
                    vstd::SerDe<uint>::Set((i != 0) ? 1 : 0, data);
                }
                cbSize += sz;
            } break;
        }
    };
    auto GetNextStruct = [&](auto &&GetNextStruct,
                             auto &&GetNextArray,
                             Type const &t,
                             vstd::span<vbyte const> &sp) -> void {
        ForceAlign();
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
        auto &&ele = t.element();
        for (auto i : vstd::range(t.dimension())) {
            ForceAlign();
            GetNext(
                GetNextArray,
                GetNextStruct,
                GetNextVector,
                t, sp);
        }
    };
    vstd::SerDe<uint3>::Set(dispatchSize, data);
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