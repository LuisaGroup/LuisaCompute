//
// Created by swfly on 2024/11/16.
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/pixel.h>

namespace luisa::compute::fallback::bc {
extern const int g_aWeights2[];
extern const int g_aWeights3[];
extern const int g_aWeights4[];

//-------------------------------------------------------------------------------------
// Constants
//-------------------------------------------------------------------------------------
constexpr uint16_t BC6H_MAX_REGIONS = 2;
constexpr uint16_t BC6H_MAX_INDICES = 16;
constexpr uint16_t BC7_MAX_REGIONS = 3;
constexpr uint16_t BC7_MAX_INDICES = 16;
constexpr uint16_t NUM_PIXELS_PER_BLOCK = 16;

constexpr uint16_t F16S_MASK = 0x8000; // f16 sign mask
constexpr uint16_t F16EM_MASK = 0x7fff;// f16 exp & mantissa mask
constexpr uint16_t F16MAX = 0x7bff;    // MAXFLT bit pattern for XMHALF

constexpr size_t BC6H_NUM_CHANNELS = 3;
constexpr size_t BC6H_MAX_SHAPES = 32;

constexpr size_t BC7_NUM_CHANNELS = 4;
constexpr size_t BC7_MAX_SHAPES = 64;

constexpr int32_t BC67_WEIGHT_MAX = 64;
constexpr uint32_t BC67_WEIGHT_SHIFT = 6;
constexpr int32_t BC67_WEIGHT_ROUND = 32;

constexpr float fEpsilon = (0.25f / 64.0f) * (0.25f / 64.0f);
constexpr float pC3[] = {2.0f / 2.0f, 1.0f / 2.0f, 0.0f / 2.0f};
constexpr float pD3[] = {0.0f / 2.0f, 1.0f / 2.0f, 2.0f / 2.0f};
constexpr float pC4[] = {3.0f / 3.0f, 2.0f / 3.0f, 1.0f / 3.0f, 0.0f / 3.0f};
constexpr float pD4[] = {0.0f / 3.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f / 3.0f};

#define SIGN_EXTEND(x, nb) ((((x) & (1 << ((nb) - 1))) ? ((~0) ^ ((1 << (nb)) - 1)) : 0) | (x))
class HDRColorA;
class LDRColorA;

template<size_t SizeInBytes>
class CBits {
public:
    uint8_t GetBit(size_t &uStartBit) const noexcept {
        LUISA_ASSUME(uStartBit < 128);
        const size_t uIndex = uStartBit >> 3;
        auto const ret = static_cast<uint8_t>((m_uBits[uIndex] >> (uStartBit - (uIndex << 3))) & 0x01);
        uStartBit++;
        return ret;
    }

    uint8_t GetBits(size_t &uStartBit, size_t uNumBits) const noexcept {
        if (uNumBits == 0) return 0;
        LUISA_ASSUME(uStartBit + uNumBits <= 128 && uNumBits <= 8);
        uint8_t ret;
        const size_t uIndex = uStartBit >> 3;
        const size_t uBase = uStartBit - (uIndex << 3);
        if (uBase + uNumBits > 8) {
            const size_t uFirstIndexBits = 8 - uBase;
            const size_t uNextIndexBits = uNumBits - uFirstIndexBits;
            ret = static_cast<uint8_t>((unsigned(m_uBits[uIndex]) >> uBase) | ((unsigned(m_uBits[uIndex + 1]) & ((1u << uNextIndexBits) - 1)) << uFirstIndexBits));
        } else {
            ret = static_cast<uint8_t>((m_uBits[uIndex] >> uBase) & ((1 << uNumBits) - 1));
        }
        LUISA_ASSUME(ret < (1 << uNumBits));
        uStartBit += uNumBits;
        return ret;
    }

    void SetBit(size_t &uStartBit, uint8_t uValue) noexcept {
        LUISA_ASSUME(uStartBit < 128 && uValue < 2);
        size_t uIndex = uStartBit >> 3;
        const size_t uBase = uStartBit - (uIndex << 3);
        m_uBits[uIndex] &= ~(1 << uBase);
        m_uBits[uIndex] |= uValue << uBase;
        uStartBit++;
    }

    void SetBits(size_t &uStartBit, size_t uNumBits, uint8_t uValue) noexcept {
        if (uNumBits == 0)
            return;
        LUISA_ASSUME(uStartBit + uNumBits <= 128 && uNumBits <= 8);
        LUISA_ASSUME(uValue < (1 << uNumBits));
        size_t uIndex = uStartBit >> 3;
        const size_t uBase = uStartBit - (uIndex << 3);
        if (uBase + uNumBits > 8) {
            const size_t uFirstIndexBits = 8 - uBase;
            const size_t uNextIndexBits = uNumBits - uFirstIndexBits;
            m_uBits[uIndex] &= ~(((1 << uFirstIndexBits) - 1) << uBase);
            m_uBits[uIndex] |= uValue << uBase;
            m_uBits[uIndex + 1] &= ~((1 << uNextIndexBits) - 1);
            m_uBits[uIndex + 1] |= uValue >> uFirstIndexBits;
        } else {
            m_uBits[uIndex] &= ~(((1 << uNumBits) - 1) << uBase);
            m_uBits[uIndex] |= uValue << uBase;
        }
        uStartBit += uNumBits;
    }

private:
    uint8_t m_uBits[SizeInBytes];
};
class HDRColorA {
public:
    float r, g, b, a;

public:
    HDRColorA() = default;
    HDRColorA(float _r, float _g, float _b, float _a) noexcept : r(_r), g(_g), b(_b), a(_a) {}

    HDRColorA(HDRColorA const &) = default;
    HDRColorA &operator=(const HDRColorA &) = default;

    HDRColorA(HDRColorA &&) = default;
    HDRColorA &operator=(HDRColorA &&) = default;

    // binary operators
    HDRColorA operator+(const HDRColorA &c) const noexcept {
        return HDRColorA(r + c.r, g + c.g, b + c.b, a + c.a);
    }

    HDRColorA operator-(const HDRColorA &c) const noexcept {
        return HDRColorA(r - c.r, g - c.g, b - c.b, a - c.a);
    }

    HDRColorA operator*(float f) const noexcept {
        return HDRColorA(r * f, g * f, b * f, a * f);
    }

    HDRColorA operator/(float f) const noexcept {
        const float fInv = 1.0f / f;
        return HDRColorA(r * fInv, g * fInv, b * fInv, a * fInv);
    }

    float operator*(const HDRColorA &c) const noexcept {
        return r * c.r + g * c.g + b * c.b + a * c.a;
    }

    // assignment operators
    HDRColorA &operator+=(const HDRColorA &c) noexcept {
        r += c.r;
        g += c.g;
        b += c.b;
        a += c.a;
        return *this;
    }

    HDRColorA &operator-=(const HDRColorA &c) noexcept {
        r -= c.r;
        g -= c.g;
        b -= c.b;
        a -= c.a;
        return *this;
    }

    HDRColorA &operator*=(float f) noexcept {
        r *= f;
        g *= f;
        b *= f;
        a *= f;
        return *this;
    }

    HDRColorA &operator/=(float f) noexcept {
        const float fInv = 1.0f / f;
        r *= fInv;
        g *= fInv;
        b *= fInv;
        a *= fInv;
        return *this;
    }

    HDRColorA &Clamp(float fMin, float fMax) noexcept {
        r = std::min<float>(fMax, std::max<float>(fMin, r));
        g = std::min<float>(fMax, std::max<float>(fMin, g));
        b = std::min<float>(fMax, std::max<float>(fMin, b));
        a = std::min<float>(fMax, std::max<float>(fMin, a));
        return *this;
    }

    HDRColorA(const LDRColorA &c) noexcept;
    HDRColorA &operator=(const LDRColorA &c) noexcept;
    LDRColorA ToLDRColorA() const noexcept;
};

inline HDRColorA *HDRColorALerp(HDRColorA *pOut, const HDRColorA *pC1, const HDRColorA *pC2, float s) noexcept {
    pOut->r = pC1->r + s * (pC2->r - pC1->r);
    pOut->g = pC1->g + s * (pC2->g - pC1->g);
    pOut->b = pC1->b + s * (pC2->b - pC1->b);
    pOut->a = pC1->a + s * (pC2->a - pC1->a);
    return pOut;
}

// #pragma pack(push, 1)
// BC1/DXT1 compression (4 bits per texel)
struct D3DX_BC1 {
    uint16_t rgb[2];// 565 colors
    uint32_t bitmap;// 2bpp rgb bitmap
};

// BC2/DXT2/3 compression (8 bits per texel)
struct D3DX_BC2 {
    uint32_t bitmap[2];// 4bpp alpha bitmap
    D3DX_BC1 bc1;      // BC1 rgb data
};

// BC3/DXT4/5 compression (8 bits per texel)
struct D3DX_BC3 {
    uint8_t alpha[2]; // alpha values
    uint8_t bitmap[6];// 3bpp alpha bitmap
    D3DX_BC1 bc1;     // BC1 rgb data
};
class LDRColorA {
public:
    uint8_t r, g, b, a;

    LDRColorA() = default;
    LDRColorA(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a) noexcept : r(_r), g(_g), b(_b), a(_a) {}

    LDRColorA(LDRColorA const &) = default;
    LDRColorA &operator=(const LDRColorA &) = default;

    LDRColorA(LDRColorA &&) = default;
    LDRColorA &operator=(LDRColorA &&) = default;

    const uint8_t &operator[](size_t uElement) const noexcept {
        switch (uElement) {
            case 0: return r;
            case 1: return g;
            case 2: return b;
            case 3: return a;
            default: LUISA_ASSUME(false); return r;
        }
    }

    uint8_t &operator[](size_t uElement) noexcept {
        switch (uElement) {
            case 0: return r;
            case 1: return g;
            case 2: return b;
            case 3: return a;
            default: LUISA_ASSUME(false); return r;
        }
    }

    LDRColorA operator=(const HDRColorA &c) noexcept {
        LDRColorA ret;
        HDRColorA tmp(c);
        tmp = tmp.Clamp(0.0f, 1.0f) * 255.0f;
        ret.r = uint8_t(tmp.r + 0.001f);
        ret.g = uint8_t(tmp.g + 0.001f);
        ret.b = uint8_t(tmp.b + 0.001f);
        ret.a = uint8_t(tmp.a + 0.001f);
        return ret;
    }

    static void InterpolateRGB(const LDRColorA &c0, const LDRColorA &c1, size_t wc, size_t wcprec, LDRColorA &out) noexcept {
        const int *aWeights = nullptr;
        switch (wcprec) {
            case 2:
                aWeights = g_aWeights2;
                LUISA_ASSUME(wc < 4);
                break;
            case 3:
                aWeights = g_aWeights3;
                LUISA_ASSUME(wc < 8);
                break;
            case 4:
                aWeights = g_aWeights4;
                LUISA_ASSUME(wc < 16);
                break;
            default:
                LUISA_ASSUME(false);
                out.r = out.g = out.b = 0;
                return;
        }
        out.r = uint8_t((uint32_t(c0.r) * uint32_t(BC67_WEIGHT_MAX - aWeights[wc]) + uint32_t(c1.r) * uint32_t(aWeights[wc]) + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT);
        out.g = uint8_t((uint32_t(c0.g) * uint32_t(BC67_WEIGHT_MAX - aWeights[wc]) + uint32_t(c1.g) * uint32_t(aWeights[wc]) + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT);
        out.b = uint8_t((uint32_t(c0.b) * uint32_t(BC67_WEIGHT_MAX - aWeights[wc]) + uint32_t(c1.b) * uint32_t(aWeights[wc]) + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT);
    }

    static void InterpolateA(const LDRColorA &c0, const LDRColorA &c1, size_t wa, size_t waprec, LDRColorA &out) noexcept {
        const int *aWeights = nullptr;
        switch (waprec) {
            case 2:
                aWeights = g_aWeights2;
                LUISA_ASSUME(wa < 4);
                break;
            case 3:
                aWeights = g_aWeights3;
                LUISA_ASSUME(wa < 8);
                break;
            case 4:
                aWeights = g_aWeights4;
                LUISA_ASSUME(wa < 16);
                break;
            default:
                LUISA_ASSUME(false);
                out.a = 0;
                return;
        }
        out.a = uint8_t((uint32_t(c0.a) * uint32_t(BC67_WEIGHT_MAX - aWeights[wa]) + uint32_t(c1.a) * uint32_t(aWeights[wa]) + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT);
    }

    static void Interpolate(const LDRColorA &c0, const LDRColorA &c1, size_t wc, size_t wa, size_t wcprec, size_t waprec, LDRColorA &out) noexcept {
        InterpolateRGB(c0, c1, wc, wcprec, out);
        InterpolateA(c0, c1, wa, waprec, out);
    }
};
struct LDREndPntPair {
    LDRColorA A;
    LDRColorA B;
};
class BC6HBlock : private CBits<16> {
public:
    void Decode(bool bSigned, HDRColorA *pOut) const noexcept;
    void Decode(bool bSigned, int x, int y, float *out) const noexcept;
private:
#pragma warning(push)
#pragma warning(disable : 4480)
    enum EField : uint8_t {
        NA,// N/A
        M, // Mode
        D, // Shape
        RW,
        RX,
        RY,
        RZ,
        GW,
        GX,
        GY,
        GZ,
        BW,
        BX,
        BY,
        BZ,
    };
#pragma warning(pop)

    struct ModeDescriptor {
        EField m_eField;
        uint8_t m_uBit;
    };

    struct ModeInfo {
        uint8_t uMode;
        uint8_t uPartitions;
        bool bTransformed;
        uint8_t uIndexPrec;
        LDRColorA RGBAPrec[BC6H_MAX_REGIONS][2];
    };

    static int Quantize(int iValue, int prec, bool bSigned) noexcept;
    static int Unquantize(int comp, uint8_t uBitsPerComp, bool bSigned) noexcept;
    static int FinishUnquantize(int comp, bool bSigned) noexcept;

private:
    static constexpr uint8_t c_NumModes = 14;
    static constexpr uint8_t c_NumModeInfo = 32;

    static const ModeDescriptor ms_aDesc[c_NumModes][82];
    static const ModeInfo ms_aInfo[c_NumModes];
    static const int ms_aModeToInfo[c_NumModeInfo];
};

class BC7Block : private CBits<16> {
public:
    void Decode(HDRColorA *pOut) const noexcept;
    void Decode(int x, int y, float *out) const noexcept;
private:
    struct ModeInfo {
        uint8_t uPartitions;
        uint8_t uPartitionBits;
        uint8_t uPBits;
        uint8_t uRotationBits;
        uint8_t uIndexModeBits;
        uint8_t uIndexPrec;
        uint8_t uIndexPrec2;
        LDRColorA RGBAPrec;
        LDRColorA RGBAPrecWithP;
    };

    static uint8_t Quantize(uint8_t comp, uint8_t uPrec) noexcept {
        LUISA_ASSUME(0 < uPrec && uPrec <= 8);
        const uint8_t rnd = std::min<uint8_t>(255u, static_cast<uint8_t>(unsigned(comp) + (1u << (7 - uPrec))));
        return uint8_t(rnd >> (8u - uPrec));
    }

    static LDRColorA Quantize(const LDRColorA &c, const LDRColorA &RGBAPrec) noexcept {
        LDRColorA q;
        q.r = Quantize(c.r, RGBAPrec.r);
        q.g = Quantize(c.g, RGBAPrec.g);
        q.b = Quantize(c.b, RGBAPrec.b);
        if (RGBAPrec.a)
            q.a = Quantize(c.a, RGBAPrec.a);
        else
            q.a = 255;
        return q;
    }

    static uint8_t Unquantize(uint8_t comp, size_t uPrec) noexcept {
        LUISA_ASSUME(0 < uPrec && uPrec <= 8);
        comp = static_cast<uint8_t>(unsigned(comp) << (8 - uPrec));
        return uint8_t(comp | (comp >> uPrec));
    }

    static LDRColorA Unquantize(const LDRColorA &c, const LDRColorA &RGBAPrec) noexcept {
        LDRColorA q;
        q.r = Unquantize(c.r, RGBAPrec.r);
        q.g = Unquantize(c.g, RGBAPrec.g);
        q.b = Unquantize(c.b, RGBAPrec.b);
        q.a = RGBAPrec.a > 0 ? Unquantize(c.a, RGBAPrec.a) : 255u;
        return q;
    }

private:
    static constexpr uint8_t c_NumModes = 8;

    static const ModeInfo ms_aInfo[c_NumModes];
};

void decode_bc1(const void *block, uint32_t x, uint32_t y, float *out) noexcept;
void decode_bc2(const void *block, uint32_t x, uint32_t y, float *out) noexcept;
void decode_bc3(const void *block, uint32_t x, uint32_t y, float *out) noexcept;
void decode_bc4(const void *block, uint32_t x, uint32_t y, float *out) noexcept;
void decode_bc5(const void *block, uint32_t x, uint32_t y, float *out) noexcept;

void decode_bc_pixel(
    PixelStorage storage,
    const std::byte *data,
    uint32_t width,
    uint32_t height,
    uint32_t x,
    uint32_t y,
    float *out) noexcept;

}// namespace luisa::compute::fallback::bc
