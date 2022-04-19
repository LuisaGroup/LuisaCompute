
//
// Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "D3D12MemAlloc.h"

#ifndef D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED
    #include <dxgi.h>
    #if D3D12MA_DXGI_1_4
        #include <dxgi1_4.h>
    #endif
#endif

#include <combaseapi.h>
#include <mutex>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <malloc.h> // for _aligned_malloc, _aligned_free

#ifndef _WIN32
    #include <shared_mutex>
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// Configuration Begin
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef D3D12MA_ASSERT
    #include <cassert>
    #define D3D12MA_ASSERT(cond) assert(cond)
#endif

// Assert that will be called very often, like inside data structures e.g. operator[].
// Making it non-empty can make program slow.
#ifndef D3D12MA_HEAVY_ASSERT
    #ifdef _DEBUG
        #define D3D12MA_HEAVY_ASSERT(expr)   //D3D12MA_ASSERT(expr)
    #else
        #define D3D12MA_HEAVY_ASSERT(expr)
    #endif
#endif

#ifndef D3D12MA_DEBUG_ALIGNMENT
    /*
    Minimum alignment of all allocations, in bytes.
    Set to more than 1 for debugging purposes only. Must be power of two.
    */
    #define D3D12MA_DEBUG_ALIGNMENT (1)
#endif

#ifndef D3D12MA_DEBUG_MARGIN
    // Minimum margin before and after every allocation, in bytes.
    // Set nonzero for debugging purposes only.
    #define D3D12MA_DEBUG_MARGIN (0)
#endif

#ifndef D3D12MA_DEBUG_GLOBAL_MUTEX
    /*
    Set this to 1 for debugging purposes only, to enable single mutex protecting all
    entry calls to the library. Can be useful for debugging multithreading issues.
    */
    #define D3D12MA_DEBUG_GLOBAL_MUTEX (0)
#endif

/*
Define this macro for debugging purposes only to force specific D3D12_RESOURCE_HEAP_TIER,
especially to test compatibility with D3D12_RESOURCE_HEAP_TIER_1 on modern GPUs.
*/
//#define D3D12MA_FORCE_RESOURCE_HEAP_TIER D3D12_RESOURCE_HEAP_TIER_1

#ifndef D3D12MA_DEFAULT_BLOCK_SIZE
   /// Default size of a block allocated as single ID3D12Heap.
   #define D3D12MA_DEFAULT_BLOCK_SIZE (64ull * 1024 * 1024)
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// Configuration End
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define D3D12MA_IID_PPV_ARGS(ppType)   __uuidof(**(ppType)), reinterpret_cast<void**>(ppType)

namespace D3D12MA
{

////////////////////////////////////////////////////////////////////////////////
// Private globals - CPU memory allocation

static void* DefaultAllocate(size_t Size, size_t Alignment, void* /*pUserData*/)
{
#ifdef _WIN32
    return _aligned_malloc(Size, Alignment);
#else
    return aligned_alloc(Alignment, Size);
#endif
}
static void DefaultFree(void* pMemory, void* /*pUserData*/)
{
#ifdef _WIN32
    return _aligned_free(pMemory);
#else
    return free(pMemory);
#endif
}

static void* Malloc(const ALLOCATION_CALLBACKS& allocs, size_t size, size_t alignment)
{
    void* const result = (*allocs.pAllocate)(size, alignment, allocs.pUserData);
    D3D12MA_ASSERT(result);
    return result;
}
static void Free(const ALLOCATION_CALLBACKS& allocs, void* memory)
{
    (*allocs.pFree)(memory, allocs.pUserData);
}

template<typename T>
static T* Allocate(const ALLOCATION_CALLBACKS& allocs)
{
    return (T*)Malloc(allocs, sizeof(T), __alignof(T));
}
template<typename T>
static T* AllocateArray(const ALLOCATION_CALLBACKS& allocs, size_t count)
{
    return (T*)Malloc(allocs, sizeof(T) * count, __alignof(T));
}

#define D3D12MA_NEW(allocs, type) new(D3D12MA::Allocate<type>(allocs))(type)
#define D3D12MA_NEW_ARRAY(allocs, type, count) new(D3D12MA::AllocateArray<type>((allocs), (count)))(type)

template<typename T>
void D3D12MA_DELETE(const ALLOCATION_CALLBACKS& allocs, T* memory)
{
    if(memory)
    {
        memory->~T();
        Free(allocs, memory);
    }
}
template<typename T>
void D3D12MA_DELETE_ARRAY(const ALLOCATION_CALLBACKS& allocs, T* memory, size_t count)
{
    if(memory)
    {
        for(size_t i = count; i--; )
        {
            memory[i].~T();
        }
        Free(allocs, memory);
    }
}

static void SetupAllocationCallbacks(ALLOCATION_CALLBACKS& outAllocs, const ALLOCATION_CALLBACKS* allocationCallbacks)
{
    if(allocationCallbacks)
    {
        outAllocs = *allocationCallbacks;
        D3D12MA_ASSERT(outAllocs.pAllocate != NULL && outAllocs.pFree != NULL);
    }
    else
    {
        outAllocs.pAllocate = &DefaultAllocate;
        outAllocs.pFree = &DefaultFree;
        outAllocs.pUserData = NULL;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Private globals - basic facilities

#define SAFE_RELEASE(ptr)   do { if(ptr) { (ptr)->Release(); (ptr) = NULL; } } while(false)

#define D3D12MA_VALIDATE(cond) do { if(!(cond)) { \
        D3D12MA_ASSERT(0 && "Validation failed: " #cond); \
        return false; \
    } } while(false)

const UINT NEW_BLOCK_SIZE_SHIFT_MAX = 3;

template<typename T>
static inline T D3D12MA_MIN(const T& a, const T& b)
{
    return a <= b ? a : b;
}
template<typename T>
static inline T D3D12MA_MAX(const T& a, const T& b)
{
    return a <= b ? b : a;
}

template<typename T>
static inline void D3D12MA_SWAP(T& a, T& b)
{
    T tmp = a; a = b; b = tmp;
}

#ifndef D3D12MA_MUTEX
    class Mutex
    {
    public:
        void Lock() { m_Mutex.lock(); }
        void Unlock() { m_Mutex.unlock(); }
    private:
        std::mutex m_Mutex;
    };
    #define D3D12MA_MUTEX Mutex
#endif

#ifdef _WIN32
    #if !defined(WINVER) || WINVER < 0x0600
        #error Required at least WinAPI version supporting: client = Windows Vista, server = Windows Server 2008.
    #endif
#endif // #ifdef _WIN32

#ifndef D3D12MA_RW_MUTEX
#ifdef _WIN32
    class RWMutex
    {
    public:
        RWMutex() { InitializeSRWLock(&m_Lock); }
        void LockRead() { AcquireSRWLockShared(&m_Lock); }
        void UnlockRead() { ReleaseSRWLockShared(&m_Lock); }
        void LockWrite() { AcquireSRWLockExclusive(&m_Lock); }
        void UnlockWrite() { ReleaseSRWLockExclusive(&m_Lock); }
    private:
        SRWLOCK m_Lock;
    };
#else // #ifdef _WIN32
    class RWMutex
    {
    public:
        RWMutex() {}
        void LockRead() { m_Mutex.lock_shared(); }
        void UnlockRead() { m_Mutex.unlock_shared(); }
        void LockWrite() { m_Mutex.lock(); }
        void UnlockWrite() { m_Mutex.unlock(); }
    private:
        std::shared_timed_mutex m_Mutex;
    };
#endif // #ifdef _WIN32
    #define D3D12MA_RW_MUTEX RWMutex
#endif // #ifndef D3D12MA_RW_MUTEX

// Scans integer for index of first nonzero bit from the Least Significant Bit (LSB). If mask is 0 then returns UINT8_MAX
static inline UINT8 BitScanLSB(UINT64 mask)
{
#if defined(_MSC_VER) && defined(_WIN64)
    unsigned long pos;
    if (_BitScanForward64(&pos, mask))
        return static_cast<UINT8>(pos);
    return UINT8_MAX;
#elif defined __GNUC__ || defined __clang__
    return static_cast<UINT8>(__builtin_ffsll(mask)) - 1U;
#else
    UINT8 pos = 0;
    UINT64 bit = 1;
    do
    {
        if (mask & bit)
            return pos;
        bit <<= 1;
    } while (pos++ < 63);
    return UINT8_MAX;
#endif
}

// Scans integer for index of first nonzero bit from the Least Significant Bit (LSB). If mask is 0 then returns UINT8_MAX
static inline UINT8 BitScanLSB(UINT32 mask)
{
#ifdef _MSC_VER
    unsigned long pos;
    if (_BitScanForward(&pos, mask))
        return static_cast<UINT8>(pos);
    return UINT8_MAX;
#elif defined __GNUC__ || defined __clang__
    return static_cast<UINT8>(__builtin_ffs(mask)) - 1U;
#else
    UINT8 pos = 0;
    UINT32 bit = 1;
    do
    {
        if (mask & bit)
            return pos;
        bit <<= 1;
    } while (pos++ < 31);
    return UINT8_MAX;
#endif
}

// Scans integer for index of first nonzero bit from the Most Significant Bit (MSB). If mask is 0 then returns UINT8_MAX
static inline UINT8 BitScanMSB(UINT64 mask)
{
#if defined(_MSC_VER) && defined(_WIN64)
    unsigned long pos;
    if (_BitScanReverse64(&pos, mask))
        return static_cast<UINT8>(pos);
#elif defined __GNUC__ || defined __clang__
    if (mask)
        return 63 - static_cast<UINT8>(__builtin_clzll(mask));
#else
    UINT8 pos = 63;
    UINT64 bit = 1ULL << 63;
    do
    {
        if (mask & bit)
            return pos;
        bit >>= 1;
    } while (pos-- > 0);
#endif
    return UINT8_MAX;
}

// Scans integer for index of first nonzero bit from the Most Significant Bit (MSB). If mask is 0 then returns UINT8_MAX
static inline UINT8 BitScanMSB(UINT32 mask)
{
#ifdef _MSC_VER
    unsigned long pos;
    if (_BitScanReverse(&pos, mask))
        return static_cast<UINT8>(pos);
#elif defined __GNUC__ || defined __clang__
    if (mask)
        return 31 - static_cast<UINT8>(__builtin_clz(mask));
#else
    UINT8 pos = 31;
    UINT32 bit = 1UL << 31;
    do
    {
        if (mask & bit)
            return pos;
        bit >>= 1;
    } while (pos-- > 0);
#endif
    return UINT8_MAX;
}

/*
Returns true if given number is a power of two.
T must be unsigned integer number or signed integer but always nonnegative.
For 0 returns true.
*/
template <typename T>
inline bool IsPow2(T x)
{
    return (x & (x-1)) == 0;
}

// Aligns given value up to nearest multiply of align value. For example: AlignUp(11, 8) = 16.
// Use types like UINT, uint64_t as T.
template <typename T>
static inline T AlignUp(T val, T alignment)
{
    D3D12MA_HEAVY_ASSERT(IsPow2(alignment));
	return (val + alignment - 1) & ~(alignment - 1);
}
// Aligns given value down to nearest multiply of align value. For example: AlignUp(11, 8) = 8.
// Use types like UINT, uint64_t as T.
template <typename T>
static inline T AlignDown(T val, T alignment)
{
    D3D12MA_HEAVY_ASSERT(IsPow2(alignment));
    return val & ~(alignment - 1);
}

// Division with mathematical rounding to nearest number.
template <typename T>
static inline T RoundDiv(T x, T y)
{
	return (x + (y / (T)2)) / y;
}
template <typename T>
static inline T DivideRoundingUp(T x, T y)
{
    return (x + y - 1) / y;
}

// Helper RAII class to lock a mutex in constructor and unlock it in destructor (at the end of scope).
struct MutexLock
{
public:
    MutexLock(D3D12MA_MUTEX& mutex, bool useMutex = true) :
        m_pMutex(useMutex ? &mutex : NULL)
    {
        if(m_pMutex)
        {
            m_pMutex->Lock();
        }
    }
    ~MutexLock()
    {
        if(m_pMutex)
        {
            m_pMutex->Unlock();
        }
    }
private:
    D3D12MA_MUTEX* m_pMutex;

    D3D12MA_CLASS_NO_COPY(MutexLock)
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for reading.
struct MutexLockRead
{
public:
    MutexLockRead(D3D12MA_RW_MUTEX& mutex, bool useMutex) :
        m_pMutex(useMutex ? &mutex : NULL)
    {
        if(m_pMutex)
        {
            m_pMutex->LockRead();
        }
    }
    ~MutexLockRead()
    {
        if(m_pMutex)
        {
            m_pMutex->UnlockRead();
        }
    }
private:
    D3D12MA_RW_MUTEX* m_pMutex;

    D3D12MA_CLASS_NO_COPY(MutexLockRead)
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for writing.
struct MutexLockWrite
{
public:
    MutexLockWrite(D3D12MA_RW_MUTEX& mutex, bool useMutex) :
        m_pMutex(useMutex ? &mutex : NULL)
    {
        if(m_pMutex)
        {
            m_pMutex->LockWrite();
        }
    }
    ~MutexLockWrite()
    {
        if(m_pMutex)
        {
            m_pMutex->UnlockWrite();
        }
    }
private:
    D3D12MA_RW_MUTEX* m_pMutex;

    D3D12MA_CLASS_NO_COPY(MutexLockWrite)
};

#if D3D12MA_DEBUG_GLOBAL_MUTEX
    static D3D12MA_MUTEX g_DebugGlobalMutex;
    #define D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK MutexLock debugGlobalMutexLock(g_DebugGlobalMutex, true);
#else
    #define D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
#endif

// Minimum size of a free suballocation to register it in the free suballocation collection.
static const UINT64 MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER = 16;

/*
Performs binary search and returns iterator to first element that is greater or
equal to `key`, according to comparison `cmp`.

Cmp should return true if first argument is less than second argument.

Returned value is the found element, if present in the collection or place where
new element with value (key) should be inserted.
*/
template <typename CmpLess, typename IterT, typename KeyT>
static IterT BinaryFindFirstNotLess(IterT beg, IterT end, const KeyT &key, const CmpLess& cmp)
{
    size_t down = 0, up = (end - beg);
    while(down < up)
    {
        const size_t mid = (down + up) / 2;
        if(cmp(*(beg+mid), key))
        {
            down = mid + 1;
        }
        else
        {
            up = mid;
        }
    }
    return beg + down;
}

/*
Performs binary search and returns iterator to an element that is equal to `key`,
according to comparison `cmp`.

Cmp should return true if first argument is less than second argument.

Returned value is the found element, if present in the collection or end if not
found.
*/
template<typename CmpLess, typename IterT, typename KeyT>
IterT BinaryFindSorted(const IterT& beg, const IterT& end, const KeyT& value, const CmpLess& cmp)
{
    IterT it = BinaryFindFirstNotLess<CmpLess, IterT, KeyT>(beg, end, value, cmp);
    if(it == end ||
        (!cmp(*it, value) && !cmp(value, *it)))
    {
        return it;
    }
    return end;
}

static UINT HeapTypeToIndex(D3D12_HEAP_TYPE type)
{
    switch(type)
    {
    case D3D12_HEAP_TYPE_DEFAULT:  return 0;
    case D3D12_HEAP_TYPE_UPLOAD:   return 1;
    case D3D12_HEAP_TYPE_READBACK: return 2;
    case D3D12_HEAP_TYPE_CUSTOM:   return 3;
    default: D3D12MA_ASSERT(0); return UINT_MAX;
    }
}

static const WCHAR* const HeapTypeNames[] = {
    L"DEFAULT",
    L"UPLOAD",
    L"READBACK",
    L"CUSTOM",
};

// Stat helper functions

static void AddStatInfo(StatInfo& dst, const StatInfo& src)
{
    dst.BlockCount += src.BlockCount;
    dst.AllocationCount += src.AllocationCount;
    dst.UnusedRangeCount += src.UnusedRangeCount;
    dst.UsedBytes += src.UsedBytes;
    dst.UnusedBytes += src.UnusedBytes;
    dst.AllocationSizeMin = D3D12MA_MIN(dst.AllocationSizeMin, src.AllocationSizeMin);
    dst.AllocationSizeMax = D3D12MA_MAX(dst.AllocationSizeMax, src.AllocationSizeMax);
    dst.UnusedRangeSizeMin = D3D12MA_MIN(dst.UnusedRangeSizeMin, src.UnusedRangeSizeMin);
    dst.UnusedRangeSizeMax = D3D12MA_MAX(dst.UnusedRangeSizeMax, src.UnusedRangeSizeMax);
}

static void PostProcessStatInfo(StatInfo& statInfo)
{
    statInfo.AllocationSizeAvg = statInfo.AllocationCount ?
        statInfo.UsedBytes / statInfo.AllocationCount : 0;
    statInfo.UnusedRangeSizeAvg = statInfo.UnusedRangeCount ?
        statInfo.UnusedBytes / statInfo.UnusedRangeCount : 0;
}

static UINT64 HeapFlagsToAlignment(D3D12_HEAP_FLAGS flags)
{
    /*
    Documentation of D3D12_HEAP_DESC structure says:

    - D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT   defined as 64KB.
    - D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT   defined as 4MB. An
    application must decide whether the heap will contain multi-sample
    anti-aliasing (MSAA), in which case, the application must choose [this flag].

    https://docs.microsoft.com/en-us/windows/desktop/api/d3d12/ns-d3d12-d3d12_heap_desc
    */

    const D3D12_HEAP_FLAGS denyAllTexturesFlags =
        D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES;
    const bool canContainAnyTextures =
        (flags & denyAllTexturesFlags) != denyAllTexturesFlags;
    return canContainAnyTextures ?
        D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
}

static bool IsFormatCompressed(DXGI_FORMAT format)
{
    switch(format)
    {
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        return true;
    default:
        return false;
    }
}

// Only some formats are supported. For others it returns 0.
static UINT GetBitsPerPixel(DXGI_FORMAT format)
{
    switch(format)
    {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
        return 128;
    case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
        return 96;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT:
        return 64;
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
        return 64;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
        return 64;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return 32;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT:
        return 32;
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_SINT:
        return 32;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
        return 32;
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
        return 32;
    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_SINT:
        return 16;
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_SINT:
        return 16;
    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_A8_UNORM:
        return 8;
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
        return 4;
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
        return 8;
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
        return 8;
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
        return 4;
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
        return 8;
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
        return 8;
    case DXGI_FORMAT_BC7_TYPELESS:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        return 8;
    default:
        return 0;
    }
}

static const D3D12_HEAP_FLAGS RESOURCE_CLASS_HEAP_FLAGS =
    D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES;

enum class ResourceClass
{
    Unknown, Buffer, Non_RT_DS_Texture, RT_DS_Texture
};

template<typename D3D12_RESOURCE_DESC_T>
static inline ResourceClass ResourceDescToResourceClass(const D3D12_RESOURCE_DESC_T& resDesc)
{
    if(resDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
        return ResourceClass::Buffer;
    // Else: it's surely a texture.
    const bool isRenderTargetOrDepthStencil =
        (resDesc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) != 0;
    return isRenderTargetOrDepthStencil ? ResourceClass::RT_DS_Texture : ResourceClass::Non_RT_DS_Texture;
}

static inline ResourceClass HeapFlagsToResourceClass(D3D12_HEAP_FLAGS heapFlags)
{
    const bool allowBuffers         = (heapFlags & D3D12_HEAP_FLAG_DENY_BUFFERS           ) == 0;
    const bool allowRtDsTextures    = (heapFlags & D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES    ) == 0;
    const bool allowNonRtDsTextures = (heapFlags & D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES) == 0;

    const uint8_t allowedGroupCount = (allowBuffers ? 1 : 0) + (allowRtDsTextures ? 1 : 0) + (allowNonRtDsTextures ? 1 : 0);
    if(allowedGroupCount != 1)
        return ResourceClass::Unknown;
    
    if(allowRtDsTextures)
        return ResourceClass::RT_DS_Texture;
    if(allowNonRtDsTextures)
        return ResourceClass::Non_RT_DS_Texture;
    return ResourceClass::Buffer;
}

// This algorithm is overly conservative.
template<typename D3D12_RESOURCE_DESC_T>
static bool CanUseSmallAlignment(const D3D12_RESOURCE_DESC_T& resourceDesc)
{
    if(resourceDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        return false;
    if((resourceDesc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) != 0)
        return false;
    if(resourceDesc.SampleDesc.Count > 1)
        return false;
    if(resourceDesc.DepthOrArraySize != 1)
        return false;

    UINT sizeX = (UINT)resourceDesc.Width;
    UINT sizeY = resourceDesc.Height;
    UINT bitsPerPixel = GetBitsPerPixel(resourceDesc.Format);
    if(bitsPerPixel == 0)
        return false;

    if(IsFormatCompressed(resourceDesc.Format))
    {
        sizeX = DivideRoundingUp(sizeX, 4u);
        sizeY = DivideRoundingUp(sizeY, 4u);
        bitsPerPixel *= 16;
    }

    UINT tileSizeX = 0, tileSizeY = 0;
    switch(bitsPerPixel)
    {
    case   8: tileSizeX = 64; tileSizeY = 64; break;
    case  16: tileSizeX = 64; tileSizeY = 32; break;
    case  32: tileSizeX = 32; tileSizeY = 32; break;
    case  64: tileSizeX = 32; tileSizeY = 16; break;
    case 128: tileSizeX = 16; tileSizeY = 16; break;
    default: return false;
    }

    const UINT tileCount = DivideRoundingUp(sizeX, tileSizeX) * DivideRoundingUp(sizeY, tileSizeY);
    return tileCount <= 16;
}

static inline bool IsHeapTypeStandard(D3D12_HEAP_TYPE type)
{
    return type == D3D12_HEAP_TYPE_DEFAULT ||
        type == D3D12_HEAP_TYPE_UPLOAD ||
        type == D3D12_HEAP_TYPE_READBACK;
}

static inline D3D12_HEAP_PROPERTIES StandardHeapTypeToHeapProperties(D3D12_HEAP_TYPE type)
{
    D3D12MA_ASSERT(IsHeapTypeStandard(type));
    D3D12_HEAP_PROPERTIES result = {};
    result.Type = type;
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Private class Vector

/*
Dynamically resizing continuous array. Class with interface similar to std::vector.
T must be POD because constructors and destructors are not called and memcpy is
used for these objects.
*/
template<typename T>
class Vector
{
public:
    using value_type = T;

    // allocationCallbacks externally owned, must outlive this object.
    Vector(const ALLOCATION_CALLBACKS& allocationCallbacks) :
        m_AllocationCallbacks(allocationCallbacks),
        m_pArray(NULL),
        m_Count(0),
        m_Capacity(0)
    {
    }

    Vector(size_t count, const ALLOCATION_CALLBACKS& allocationCallbacks) :
        m_AllocationCallbacks(allocationCallbacks),
        m_pArray(count ? AllocateArray<T>(allocationCallbacks, count) : NULL),
        m_Count(count),
        m_Capacity(count)
    {
    }

    Vector(const Vector<T>& src) :
        m_AllocationCallbacks(src.m_AllocationCallbacks),
        m_pArray(src.m_Count ? AllocateArray<T>(src.m_AllocationCallbacks, src.m_Count) : NULL),
        m_Count(src.m_Count),
        m_Capacity(src.m_Count)
    {
        if(m_Count > 0)
        {
            memcpy(m_pArray, src.m_pArray, m_Count * sizeof(T));
        }
    }

    ~Vector()
    {
        Free(m_AllocationCallbacks, m_pArray);
    }

    Vector& operator=(const Vector<T>& rhs)
    {
        if(&rhs != this)
        {
            resize(rhs.m_Count);
            if(m_Count != 0)
            {
                memcpy(m_pArray, rhs.m_pArray, m_Count * sizeof(T));
            }
        }
        return *this;
    }

    bool empty() const { return m_Count == 0; }
    size_t size() const { return m_Count; }
    T* data() { return m_pArray; }
    const T* data() const { return m_pArray; }

    T& operator[](size_t index)
    {
        D3D12MA_HEAVY_ASSERT(index < m_Count);
        return m_pArray[index];
    }
    const T& operator[](size_t index) const
    {
        D3D12MA_HEAVY_ASSERT(index < m_Count);
        return m_pArray[index];
    }

    T& front()
    {
        D3D12MA_HEAVY_ASSERT(m_Count > 0);
        return m_pArray[0];
    }
    const T& front() const
    {
        D3D12MA_HEAVY_ASSERT(m_Count > 0);
        return m_pArray[0];
    }
    T& back()
    {
        D3D12MA_HEAVY_ASSERT(m_Count > 0);
        return m_pArray[m_Count - 1];
    }
    const T& back() const
    {
        D3D12MA_HEAVY_ASSERT(m_Count > 0);
        return m_pArray[m_Count - 1];
    }

    void reserve(size_t newCapacity, bool freeMemory = false)
    {
        newCapacity = D3D12MA_MAX(newCapacity, m_Count);

        if((newCapacity < m_Capacity) && !freeMemory)
        {
            newCapacity = m_Capacity;
        }

        if(newCapacity != m_Capacity)
        {
            T* const newArray = newCapacity ? AllocateArray<T>(m_AllocationCallbacks, newCapacity) : NULL;
            if(m_Count != 0)
            {
                memcpy(newArray, m_pArray, m_Count * sizeof(T));
            }
            Free(m_AllocationCallbacks, m_pArray);
            m_Capacity = newCapacity;
            m_pArray = newArray;
        }
    }

    void resize(size_t newCount, bool freeMemory = false)
    {
        size_t newCapacity = m_Capacity;
        if(newCount > m_Capacity)
        {
            newCapacity = D3D12MA_MAX(newCount, D3D12MA_MAX(m_Capacity * 3 / 2, (size_t)8));
        }
        else if(freeMemory)
        {
            newCapacity = newCount;
        }

        if(newCapacity != m_Capacity)
        {
            T* const newArray = newCapacity ? AllocateArray<T>(m_AllocationCallbacks, newCapacity) : NULL;
            const size_t elementsToCopy = D3D12MA_MIN(m_Count, newCount);
            if(elementsToCopy != 0)
            {
                memcpy(newArray, m_pArray, elementsToCopy * sizeof(T));
            }
            Free(m_AllocationCallbacks, m_pArray);
            m_Capacity = newCapacity;
            m_pArray = newArray;
        }

        m_Count = newCount;
    }

    void clear(bool freeMemory = false)
    {
        resize(0, freeMemory);
    }

    void insert(size_t index, const T& src)
    {
        D3D12MA_HEAVY_ASSERT(index <= m_Count);
        const size_t oldCount = size();
        resize(oldCount + 1);
        if(index < oldCount)
        {
            memmove(m_pArray + (index + 1), m_pArray + index, (oldCount - index) * sizeof(T));
        }
        m_pArray[index] = src;
    }

    void remove(size_t index)
    {
        D3D12MA_HEAVY_ASSERT(index < m_Count);
        const size_t oldCount = size();
        if(index < oldCount - 1)
        {
            memmove(m_pArray + index, m_pArray + (index + 1), (oldCount - index - 1) * sizeof(T));
        }
        resize(oldCount - 1);
    }

    void push_back(const T& src)
    {
        const size_t newIndex = size();
        resize(newIndex + 1);
        m_pArray[newIndex] = src;
    }

    void pop_back()
    {
        D3D12MA_HEAVY_ASSERT(m_Count > 0);
        resize(size() - 1);
    }

    void push_front(const T& src)
    {
        insert(0, src);
    }

    void pop_front()
    {
        D3D12MA_HEAVY_ASSERT(m_Count > 0);
        remove(0);
    }

    using iterator = T*;

    iterator begin() { return m_pArray; }
    const iterator cbegin() const { return m_pArray; }

    iterator rbegin() { return end() - 1; }
    const iterator crbegin() const { return cend() - 1; }

    iterator end() { return m_pArray + m_Count; }
    const iterator cend() const { return m_pArray + m_Count; }

    iterator rend() { return begin() - 1; }
    const iterator crend() const { return cbegin() - 1; }

    template<typename CmpLess>
    size_t InsertSorted(const T& value, const CmpLess& cmp)
    {
        const size_t indexToInsert = BinaryFindFirstNotLess<CmpLess, iterator, T>(
            m_pArray,
            m_pArray + m_Count,
            value,
            cmp) - m_pArray;
        insert(indexToInsert, value);
        return indexToInsert;
    }

    template<typename CmpLess>
    bool RemoveSorted(const T& value, const CmpLess& cmp)
    {
        const iterator it = BinaryFindFirstNotLess(
            m_pArray,
            m_pArray + m_Count,
            value,
            cmp);
        if((it != end()) && !cmp(*it, value) && !cmp(value, *it))
        {
            size_t indexToRemove = it - begin();
            remove(indexToRemove);
            return true;
        }
        return false;
    }

private:
    const ALLOCATION_CALLBACKS& m_AllocationCallbacks;
    T* m_pArray;
    size_t m_Count;
    size_t m_Capacity;
};

////////////////////////////////////////////////////////////////////////////////
// Private class StringBuilder

class StringBuilder
{
public:
    StringBuilder(const ALLOCATION_CALLBACKS& allocationCallbacks) : m_Data(allocationCallbacks) { }

    size_t GetLength() const { return m_Data.size(); }
    LPCWSTR GetData() const { return m_Data.data(); }

    void Add(WCHAR ch) { m_Data.push_back(ch); }
    void Add(LPCWSTR str);
    void AddNewLine() { Add(L'\n'); }
    void AddNumber(UINT num);
    void AddNumber(UINT64 num);

private:
    Vector<WCHAR> m_Data;
};

void StringBuilder::Add(LPCWSTR str)
{
    const size_t len = wcslen(str);
    if (len > 0)
    {
        const size_t oldCount = m_Data.size();
        m_Data.resize(oldCount + len);
        memcpy(m_Data.data() + oldCount, str, len * sizeof(WCHAR));
    }
}

void StringBuilder::AddNumber(UINT num)
{
    WCHAR buf[11];
    buf[10] = L'\0';
    WCHAR *p = &buf[10];
    do
    {
        *--p = L'0' + (num % 10);
        num /= 10;
    }
    while (num);
    Add(p);
}

void StringBuilder::AddNumber(UINT64 num)
{
    WCHAR buf[21];
    buf[20] = L'\0';
    WCHAR *p = &buf[20];
    do
    {
        *--p = L'0' + (num % 10);
        num /= 10;
    }
    while (num);
    Add(p);
}

////////////////////////////////////////////////////////////////////////////////
// Private class JsonWriter

/*
Allows to conveniently build a correct JSON document to be written to the
StringBuilder passed to the constructor.
*/
class JsonWriter
{
public:
    // stringBuilder - string builder to write the document to. Must remain alive for the whole lifetime of this object.
    JsonWriter(const ALLOCATION_CALLBACKS& allocationCallbacks, StringBuilder& stringBuilder);
    ~JsonWriter();

    // Begins object by writing "{".
    // Inside an object, you must call pairs of WriteString and a value, e.g.:
    // j.BeginObject(true); j.WriteString("A"); j.WriteNumber(1); j.WriteString("B"); j.WriteNumber(2); j.EndObject();
    // Will write: { "A": 1, "B": 2 }
    void BeginObject(bool singleLine = false);
    // Ends object by writing "}".
    void EndObject();

    // Begins array by writing "[".
    // Inside an array, you can write a sequence of any values.
    void BeginArray(bool singleLine = false);
    // Ends array by writing "[".
    void EndArray();

    // Writes a string value inside "".
    // pStr can contain any UTF-16 characters, including '"', new line etc. - they will be properly escaped.
    void WriteString(LPCWSTR pStr);

    // Begins writing a string value.
    // Call BeginString, ContinueString, ContinueString, ..., EndString instead of
    // WriteString to conveniently build the string content incrementally, made of
    // parts including numbers.
    void BeginString(LPCWSTR pStr = NULL);
    // Posts next part of an open string.
    void ContinueString(LPCWSTR pStr);
    // Posts next part of an open string. The number is converted to decimal characters.
    void ContinueString(UINT num);
    void ContinueString(UINT64 num);
    // Posts next part of an open string. Pointer value is converted to characters
    // using "%p" formatting - shown as hexadecimal number, e.g.: 000000081276Ad00
    // void ContinueString_Pointer(const void* ptr);
    // Ends writing a string value by writing '"'.
    void EndString(LPCWSTR pStr = NULL);

    void AddAllocationToObject(const Allocation& alloc);

    // Writes a number value.
    void WriteNumber(UINT num);
    void WriteNumber(UINT64 num);
    // Writes a boolean value - false or true.
    void WriteBool(bool b);
    // Writes a null value.
    void WriteNull();

private:
    static const WCHAR* const INDENT;

    enum CollectionType
    {
        COLLECTION_TYPE_OBJECT,
        COLLECTION_TYPE_ARRAY,
    };
    struct StackItem
    {
        CollectionType type;
        UINT valueCount;
        bool singleLineMode;
    };

    StringBuilder& m_SB;
    Vector<StackItem> m_Stack;
    bool m_InsideString;

    void BeginValue(bool isString);
    void WriteIndent(bool oneLess = false);
};

const WCHAR* const JsonWriter::INDENT = L"  ";

JsonWriter::JsonWriter(const ALLOCATION_CALLBACKS& allocationCallbacks, StringBuilder& stringBuilder) :
    m_SB(stringBuilder),
    m_Stack(allocationCallbacks),
    m_InsideString(false)
{
}

JsonWriter::~JsonWriter()
{
    D3D12MA_ASSERT(!m_InsideString);
    D3D12MA_ASSERT(m_Stack.empty());
}

void JsonWriter::BeginObject(bool singleLine)
{
    D3D12MA_ASSERT(!m_InsideString);

    BeginValue(false);
    m_SB.Add(L'{');

    StackItem stackItem;
    stackItem.type = COLLECTION_TYPE_OBJECT;
    stackItem.valueCount = 0;
    stackItem.singleLineMode = singleLine;
    m_Stack.push_back(stackItem);
}

void JsonWriter::EndObject()
{
    D3D12MA_ASSERT(!m_InsideString);
    D3D12MA_ASSERT(!m_Stack.empty() && m_Stack.back().type == COLLECTION_TYPE_OBJECT);
    D3D12MA_ASSERT(m_Stack.back().valueCount % 2 == 0);

    WriteIndent(true);
    m_SB.Add(L'}');

    m_Stack.pop_back();
}

void JsonWriter::BeginArray(bool singleLine)
{
    D3D12MA_ASSERT(!m_InsideString);

    BeginValue(false);
    m_SB.Add(L'[');

    StackItem stackItem;
    stackItem.type = COLLECTION_TYPE_ARRAY;
    stackItem.valueCount = 0;
    stackItem.singleLineMode = singleLine;
    m_Stack.push_back(stackItem);
}

void JsonWriter::EndArray()
{
    D3D12MA_ASSERT(!m_InsideString);
    D3D12MA_ASSERT(!m_Stack.empty() && m_Stack.back().type == COLLECTION_TYPE_ARRAY);

    WriteIndent(true);
    m_SB.Add(L']');

    m_Stack.pop_back();
}

void JsonWriter::WriteString(LPCWSTR pStr)
{
    BeginString(pStr);
    EndString();
}

void JsonWriter::BeginString(LPCWSTR pStr)
{
    D3D12MA_ASSERT(!m_InsideString);

    BeginValue(true);
    m_InsideString = true;
    m_SB.Add(L'"');
    if (pStr != NULL)
    {
        ContinueString(pStr);
    }
}

void JsonWriter::ContinueString(LPCWSTR pStr)
{
    D3D12MA_ASSERT(m_InsideString);
    D3D12MA_ASSERT(pStr);

    for (const WCHAR *p = pStr; *p; ++p)
    {
        // the strings we encode are assumed to be in UTF-16LE format, the native
        // windows wide character Unicode format. In this encoding Unicode code
        // points U+0000 to U+D7FF and U+E000 to U+FFFF are encoded in two bytes,
        // and everything else takes more than two bytes. We will reject any
        // multi wchar character encodings for simplicity.
        UINT val = (UINT)*p;
        D3D12MA_ASSERT(((val <= 0xD7FF) || (0xE000 <= val && val <= 0xFFFF)) &&
            "Character not currently supported.");
        switch (*p)
        {
        case L'"':  m_SB.Add(L'\\'); m_SB.Add(L'"');  break;
        case L'\\': m_SB.Add(L'\\'); m_SB.Add(L'\\'); break;
        case L'/':  m_SB.Add(L'\\'); m_SB.Add(L'/');  break;
        case L'\b': m_SB.Add(L'\\'); m_SB.Add(L'b');  break;
        case L'\f': m_SB.Add(L'\\'); m_SB.Add(L'f');  break;
        case L'\n': m_SB.Add(L'\\'); m_SB.Add(L'n');  break;
        case L'\r': m_SB.Add(L'\\'); m_SB.Add(L'r');  break;
        case L'\t': m_SB.Add(L'\\'); m_SB.Add(L't');  break;
        default:
            // conservatively use encoding \uXXXX for any Unicode character
            // requiring more than one byte.
            if (32 <= val && val < 256)
                m_SB.Add(*p);
            else
            {
                m_SB.Add(L'\\');
                m_SB.Add(L'u');
                for (UINT i = 0; i < 4; ++i)
                {
                    UINT hexDigit = (val & 0xF000) >> 12;
                    val <<= 4;
                    if (hexDigit < 10)
                        m_SB.Add(L'0' + (WCHAR)hexDigit);
                    else
                        m_SB.Add(L'A' + (WCHAR)hexDigit);
                }
            }
            break;
        }
    }
}

void JsonWriter::ContinueString(UINT num)
{
    D3D12MA_ASSERT(m_InsideString);
    m_SB.AddNumber(num);
}

void JsonWriter::ContinueString(UINT64 num)
{
    D3D12MA_ASSERT(m_InsideString);
    m_SB.AddNumber(num);
}

void JsonWriter::EndString(LPCWSTR pStr)
{
    D3D12MA_ASSERT(m_InsideString);

    if (pStr)
        ContinueString(pStr);
    m_SB.Add(L'"');
    m_InsideString = false;
}

void JsonWriter::WriteNumber(UINT num)
{
    D3D12MA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.AddNumber(num);
}

void JsonWriter::WriteNumber(UINT64 num)
{
    D3D12MA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.AddNumber(num);
}

void JsonWriter::WriteBool(bool b)
{
    D3D12MA_ASSERT(!m_InsideString);
    BeginValue(false);
    if (b)
        m_SB.Add(L"true");
    else
        m_SB.Add(L"false");
}

void JsonWriter::WriteNull()
{
    D3D12MA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.Add(L"null");
}

void JsonWriter::BeginValue(bool isString)
{
    if (!m_Stack.empty())
    {
        StackItem& currItem = m_Stack.back();
        if (currItem.type == COLLECTION_TYPE_OBJECT && currItem.valueCount % 2 == 0)
        {
            D3D12MA_ASSERT(isString);
        }

        if (currItem.type == COLLECTION_TYPE_OBJECT && currItem.valueCount % 2 == 1)
        {
            m_SB.Add(L':'); m_SB.Add(L' ');
        }
        else if (currItem.valueCount > 0)
        {
            m_SB.Add(L','); m_SB.Add(L' ');
            WriteIndent();
        }
        else
        {
            WriteIndent();
        }
        ++currItem.valueCount;
    }
}

void JsonWriter::WriteIndent(bool oneLess)
{
    if (!m_Stack.empty() && !m_Stack.back().singleLineMode)
    {
        m_SB.AddNewLine();

        size_t count = m_Stack.size();
        if (count > 0 && oneLess)
        {
            --count;
        }
        for (size_t i = 0; i < count; ++i)
        {
            m_SB.Add(INDENT);
        }
    }
}

void JsonWriter::AddAllocationToObject(const Allocation& alloc)
{
    WriteString(L"Type");
    switch (alloc.m_PackedData.GetResourceDimension()) {
    case D3D12_RESOURCE_DIMENSION_UNKNOWN:
        WriteString(L"UNKNOWN");
        break;
    case D3D12_RESOURCE_DIMENSION_BUFFER:
        WriteString(L"BUFFER");
        break;
    case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
        WriteString(L"TEXTURE1D");
        break;
    case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
        WriteString(L"TEXTURE2D");
        break;
    case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
        WriteString(L"TEXTURE3D");
        break;
    default: D3D12MA_ASSERT(0); break;
    }
    WriteString(L"Size");
    WriteNumber(alloc.GetSize());
    LPCWSTR name = alloc.GetName();
    if(name != NULL)
    {
        WriteString(L"Name");
        WriteString(name);
    }
    if(alloc.m_PackedData.GetResourceFlags())
    {
        WriteString(L"Flags");
        WriteNumber((UINT)alloc.m_PackedData.GetResourceFlags());
    }
    if(alloc.m_PackedData.GetTextureLayout())
    {
        WriteString(L"Layout");
        WriteNumber((UINT)alloc.m_PackedData.GetTextureLayout());
    }
    if(alloc.m_CreationFrameIndex)
    {
        WriteString(L"CreationFrameIndex");
        WriteNumber(alloc.m_CreationFrameIndex);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Private class PoolAllocator

/*
Allocator for objects of type T using a list of arrays (pools) to speed up
allocation. Number of elements that can be allocated is not bounded because
allocator can create multiple blocks.
T should be POD because constructor and destructor is not called in Alloc or
Free.
*/
template<typename T>
class PoolAllocator
{
    D3D12MA_CLASS_NO_COPY(PoolAllocator)
public:
    // allocationCallbacks externally owned, must outlive this object.
    PoolAllocator(const ALLOCATION_CALLBACKS& allocationCallbacks, UINT firstBlockCapacity);
    ~PoolAllocator() { Clear(); }
    void Clear();
    template<typename... Types> T* Alloc(Types... args);
    void Free(T* ptr);

private:
    union Item
    {
        UINT NextFreeIndex; // UINT32_MAX means end of list.
        alignas(T) char Value[sizeof(T)];
    };

    struct ItemBlock
    {
        Item* pItems;
        UINT Capacity;
        UINT FirstFreeIndex;
    };

    const ALLOCATION_CALLBACKS& m_AllocationCallbacks;
    const UINT m_FirstBlockCapacity;
    Vector<ItemBlock> m_ItemBlocks;

    ItemBlock& CreateNewBlock();
};

template<typename T>
PoolAllocator<T>::PoolAllocator(const ALLOCATION_CALLBACKS& allocationCallbacks, UINT firstBlockCapacity) :
    m_AllocationCallbacks(allocationCallbacks),
    m_FirstBlockCapacity(firstBlockCapacity),
    m_ItemBlocks(allocationCallbacks)
{
    D3D12MA_ASSERT(m_FirstBlockCapacity > 1);
}

template<typename T>
void PoolAllocator<T>::Clear()
{
    for(size_t i = m_ItemBlocks.size(); i--; )
    {
        D3D12MA_DELETE_ARRAY(m_AllocationCallbacks, m_ItemBlocks[i].pItems, m_ItemBlocks[i].Capacity);
    }
    m_ItemBlocks.clear(true);
}

template<typename T>
template<typename... Types> T* PoolAllocator<T>::Alloc(Types... args)
{
    for(size_t i = m_ItemBlocks.size(); i--; )
    {
        ItemBlock& block = m_ItemBlocks[i];
        // This block has some free items: Use first one.
        if(block.FirstFreeIndex != UINT32_MAX)
        {
            Item* const pItem = &block.pItems[block.FirstFreeIndex];
            block.FirstFreeIndex = pItem->NextFreeIndex;
            T* result = (T*)&pItem->Value;
            new(result)T(std::forward<Types>(args)...); // Explicit constructor call.
            return result;
        }
    }

    // No block has free item: Create new one and use it.
    ItemBlock& newBlock = CreateNewBlock();
    Item* const pItem = &newBlock.pItems[0];
    newBlock.FirstFreeIndex = pItem->NextFreeIndex;
    T* result = (T*)pItem->Value;
    new(result)T(std::forward<Types>(args)...); // Explicit constructor call.
    return result;
}

template<typename T>
void PoolAllocator<T>::Free(T* ptr)
{
    // Search all memory blocks to find ptr.
    for(size_t i = m_ItemBlocks.size(); i--; )
    {
        ItemBlock& block = m_ItemBlocks[i];

        Item* pItemPtr;
        memcpy(&pItemPtr, &ptr, sizeof(pItemPtr));

        // Check if pItemPtr is in address range of this block.
        if((pItemPtr >= block.pItems) && (pItemPtr < block.pItems + block.Capacity))
        {
            ptr->~T(); // Explicit destructor call.
            const UINT index = static_cast<UINT>(pItemPtr - block.pItems);
            pItemPtr->NextFreeIndex = block.FirstFreeIndex;
            block.FirstFreeIndex = index;
            return;
        }
    }
    D3D12MA_ASSERT(0 && "Pointer doesn't belong to this memory pool.");
}

template<typename T>
typename PoolAllocator<T>::ItemBlock& PoolAllocator<T>::CreateNewBlock()
{
    const UINT newBlockCapacity = m_ItemBlocks.empty() ?
        m_FirstBlockCapacity : m_ItemBlocks.back().Capacity * 3 / 2;

    const ItemBlock newBlock = {
        D3D12MA_NEW_ARRAY(m_AllocationCallbacks, Item, newBlockCapacity),
        newBlockCapacity,
        0 };

    m_ItemBlocks.push_back(newBlock);

    // Setup singly-linked list of all free items in this block.
    for(UINT i = 0; i < newBlockCapacity - 1; ++i)
    {
        newBlock.pItems[i].NextFreeIndex = i + 1;
    }
    newBlock.pItems[newBlockCapacity - 1].NextFreeIndex = UINT32_MAX;
    return m_ItemBlocks.back();
}

////////////////////////////////////////////////////////////////////////////////
// Private class List

/*
Doubly linked list, with elements allocated out of PoolAllocator.
Has custom interface, as well as STL-style interface, including iterator and
const_iterator.
*/
template<typename T>
class List
{
    D3D12MA_CLASS_NO_COPY(List)
public:
    struct Item
    {
        Item* pPrev;
        Item* pNext;
        T Value;
    };

    // allocationCallbacks externally owned, must outlive this object.
    List(const ALLOCATION_CALLBACKS& allocationCallbacks);

    // Intentionally not calling Clear, because that would be unnecessary
    // computations to return all items to m_ItemAllocator as free.
    // ~List() {}
    
    void Clear();

    size_t GetCount() const { return m_Count; }
    bool IsEmpty() const { return m_Count == 0; }

    Item* Front() { return m_pFront; }
    const Item* Front() const { return m_pFront; }
    Item* Back() { return m_pBack; }
    const Item* Back() const { return m_pBack; }

    Item* PushBack();
    Item* PushFront();
    Item* PushBack(const T& value);
    Item* PushFront(const T& value);
    void PopBack();
    void PopFront();

    // Item can be null - it means PushBack.
    Item* InsertBefore(Item* pItem);
    // Item can be null - it means PushFront.
    Item* InsertAfter(Item* pItem);

    Item* InsertBefore(Item* pItem, const T& value);
    Item* InsertAfter(Item* pItem, const T& value);

    void Remove(Item* pItem);

    class reverse_iterator;
    class const_reverse_iterator;
    class iterator
    {
    public:
        iterator() :
            m_pList(NULL),
            m_pItem(NULL)
        {
        }

        iterator(const reverse_iterator& src) :
            m_pList(src.m_pList),
            m_pItem(src.m_pItem)
        {
        }

        T& operator*() const
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            return m_pItem->Value;
        }
        T* operator->() const
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            return &m_pItem->Value;
        }

        iterator& operator++()
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            m_pItem = m_pItem->pNext;
            return *this;
        }
        iterator& operator--()
        {
            if(m_pItem != NULL)
            {
                m_pItem = m_pItem->pPrev;
            }
            else
            {
                D3D12MA_HEAVY_ASSERT(!m_pList->IsEmpty());
                m_pItem = m_pList->Back();
            }
            return *this;
        }

        iterator operator++(int)
        {
            iterator result = *this;
            ++*this;
            return result;
        }
        iterator operator--(int)
        {
            iterator result = *this;
            --*this;
            return result;
        }

        bool operator==(const iterator& rhs) const
        {
            D3D12MA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem == rhs.m_pItem;
        }
        bool operator!=(const iterator& rhs) const
        {
            D3D12MA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem != rhs.m_pItem;
        }

    private:
        List<T>* m_pList;
        Item* m_pItem;

        iterator(List<T>* pList, Item* pItem) :
            m_pList(pList),
            m_pItem(pItem)
        {
        }

        friend class List<T>;
        friend class const_iterator;
    };
    
    class reverse_iterator
    {
    public:
        reverse_iterator() :
            m_pList(NULL),
            m_pItem(NULL)
        {
        }

        reverse_iterator(const iterator& src) :
            m_pList(src.m_pList),
            m_pItem(src.m_pItem)
        {
        }

        T& operator*() const
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            return m_pItem->Value;
        }
        T* operator->() const
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            return &m_pItem->Value;
        }

        reverse_iterator& operator++()
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            m_pItem = m_pItem->pPrev;
            return *this;
        }
        reverse_iterator& operator--()
        {
            if(m_pItem != NULL)
            {
                m_pItem = m_pItem->pNext;
            }
            else
            {
                D3D12MA_HEAVY_ASSERT(!m_pList->IsEmpty());
                m_pItem = m_pList->Front();
            }
            return *this;
        }

        reverse_iterator operator++(int)
        {
            reverse_iterator result = *this;
            ++*this;
            return result;
        }
        reverse_iterator operator--(int)
        {
            reverse_iterator result = *this;
            --*this;
            return result;
        }

        bool operator==(const reverse_iterator& rhs) const
        {
            D3D12MA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem == rhs.m_pItem;
        }
        bool operator!=(const reverse_iterator& rhs) const
        {
            D3D12MA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem != rhs.m_pItem;
        }

    private:
        List<T>* m_pList;
        Item* m_pItem;

        reverse_iterator(List<T>* pList, Item* pItem) :
            m_pList(pList),
            m_pItem(pItem)
        {
        }

        friend class List<T>;
        friend class const_reverse_iterator;
    };
    
    class const_iterator
    {
    public:
        const_iterator() :
            m_pList(NULL),
            m_pItem(NULL)
        {
        }

        const_iterator(const iterator& src) :
            m_pList(src.m_pList),
            m_pItem(src.m_pItem)
        {
        }

        const_iterator(const reverse_iterator& src) :
            m_pList(src.m_pList),
            m_pItem(src.m_pItem)
        {
        }

        const_iterator(const const_reverse_iterator& src) :
            m_pList(src.m_pList),
            m_pItem(src.m_pItem)
        {
        }

        iterator dropConst() const
        {
            return iterator(const_cast<List<T>*>(m_pList), const_cast<Item*>(m_pItem));
        }

        const T& operator*() const
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            return m_pItem->Value;
        }
        const T* operator->() const
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            return &m_pItem->Value;
        }

        const_iterator& operator++()
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            m_pItem = m_pItem->pNext;
            return *this;
        }
        const_iterator& operator--()
        {
            if(m_pItem != NULL)
            {
                m_pItem = m_pItem->pPrev;
            }
            else
            {
                D3D12MA_HEAVY_ASSERT(!m_pList->IsEmpty());
                m_pItem = m_pList->Back();
            }
            return *this;
        }

        const_iterator operator++(int)
        {
            const_iterator result = *this;
            ++*this;
            return result;
        }
        const_iterator operator--(int)
        {
            const_iterator result = *this;
            --*this;
            return result;
        }

        bool operator==(const const_iterator& rhs) const
        {
            D3D12MA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem == rhs.m_pItem;
        }
        bool operator!=(const const_iterator& rhs) const
        {
            D3D12MA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem != rhs.m_pItem;
        }

    private:
        const_iterator(const List<T>* pList, const Item* pItem) :
            m_pList(pList),
            m_pItem(pItem)
        {
        }

        const List<T>* m_pList;
        const Item* m_pItem;

        friend class List<T>;
    };

    class const_reverse_iterator
    {
    public:
        const_reverse_iterator() :
            m_pList(NULL),
            m_pItem(NULL)
        {
        }

        const_reverse_iterator(const iterator& src) :
            m_pList(src.m_pList),
            m_pItem(src.m_pItem)
        {
        }

        const_reverse_iterator(const reverse_iterator& src) :
            m_pList(src.m_pList),
            m_pItem(src.m_pItem)
        {
        }

        const_reverse_iterator(const const_iterator& src) :
            m_pList(src.m_pList),
            m_pItem(src.m_pItem)
        {
        }

        reverse_iterator dropConst() const
        {
            return reverse_iterator(const_cast<List<T>*>(m_pList), const_cast<Item*>(m_pItem));
        }

        const T& operator*() const
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            return m_pItem->Value;
        }
        const T* operator->() const
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            return &m_pItem->Value;
        }

        const_reverse_iterator& operator++()
        {
            D3D12MA_HEAVY_ASSERT(m_pItem != NULL);
            m_pItem = m_pItem->pPrev;
            return *this;
        }
        const_reverse_iterator& operator--()
        {
            if(m_pItem != NULL)
            {
                m_pItem = m_pItem->pNext;
            }
            else
            {
                D3D12MA_HEAVY_ASSERT(!m_pList->IsEmpty());
                m_pItem = m_pList->Front();
            }
            return *this;
        }

        const_reverse_iterator operator++(int)
        {
            const_reverse_iterator result = *this;
            ++*this;
            return result;
        }
        const_reverse_iterator operator--(int)
        {
            const_reverse_iterator result = *this;
            --*this;
            return result;
        }

        bool operator==(const const_reverse_iterator& rhs) const
        {
            D3D12MA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem == rhs.m_pItem;
        }
        bool operator!=(const const_reverse_iterator& rhs) const
        {
            D3D12MA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem != rhs.m_pItem;
        }

    private:
        const_reverse_iterator(const List<T>* pList, const Item* pItem) :
            m_pList(pList),
            m_pItem(pItem)
        {
        }

        const List<T>* m_pList;
        const Item* m_pItem;

        friend class List<T>;
    };

    bool empty() const { return IsEmpty(); }
    size_t size() const { return GetCount(); }

    iterator begin() { return iterator(this, Front()); }
    iterator end() { return iterator(this, NULL); }

    const_iterator cbegin() const { return const_iterator(this, Front()); }
    const_iterator cend() const { return const_iterator(this, NULL); }

    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }

    reverse_iterator rbegin() { return reverse_iterator(this, Back()); }
    reverse_iterator rend() { return reverse_iterator(this, NULL); }

    const_reverse_iterator crbegin() const { return const_reverse_iterator(this, Back()); }
    const_reverse_iterator crend() const { return const_reverse_iterator(this, NULL); }

    const_reverse_iterator rbegin() const { return crbegin(); }
    const_reverse_iterator rend() const { return crend(); }

    void clear() { Clear(); }
    void push_back(const T& value) { PushBack(value); }
    void erase(iterator it) { Remove(it.m_pItem); }
    iterator insert(iterator it, const T& value) { return iterator(this, InsertBefore(it.m_pItem, value)); }

private:
    const ALLOCATION_CALLBACKS& m_AllocationCallbacks;
    PoolAllocator<Item> m_ItemAllocator;
    Item* m_pFront;
    Item* m_pBack;
    size_t m_Count;
};

template<typename T>
List<T>::List(const ALLOCATION_CALLBACKS& allocationCallbacks) :
    m_AllocationCallbacks(allocationCallbacks),
    m_ItemAllocator(allocationCallbacks, 128),
    m_pFront(NULL),
    m_pBack(NULL),
    m_Count(0)
{
}

template<typename T>
void List<T>::Clear()
{
    if(!IsEmpty())
    {
        Item* pItem = m_pBack;
        while(pItem != NULL)
        {
            Item* const pPrevItem = pItem->pPrev;
            m_ItemAllocator.Free(pItem);
            pItem = pPrevItem;
        }
        m_pFront = NULL;
        m_pBack = NULL;
        m_Count = 0;
    }
}

template<typename T>
typename List<T>::Item* List<T>::PushBack()
{
    Item* const pNewItem = m_ItemAllocator.Alloc();
    pNewItem->pNext = NULL;
    if(IsEmpty())
    {
        pNewItem->pPrev = NULL;
        m_pFront = pNewItem;
        m_pBack = pNewItem;
        m_Count = 1;
    }
    else
    {
        pNewItem->pPrev = m_pBack;
        m_pBack->pNext = pNewItem;
        m_pBack = pNewItem;
        ++m_Count;
    }
    return pNewItem;
}

template<typename T>
typename List<T>::Item* List<T>::PushFront()
{
    Item* const pNewItem = m_ItemAllocator.Alloc();
    pNewItem->pPrev = NULL;
    if(IsEmpty())
    {
        pNewItem->pNext = NULL;
        m_pFront = pNewItem;
        m_pBack = pNewItem;
        m_Count = 1;
    }
    else
    {
        pNewItem->pNext = m_pFront;
        m_pFront->pPrev = pNewItem;
        m_pFront = pNewItem;
        ++m_Count;
    }
    return pNewItem;
}

template<typename T>
typename List<T>::Item* List<T>::PushBack(const T& value)
{
    Item* const pNewItem = PushBack();
    pNewItem->Value = value;
    return pNewItem;
}

template<typename T>
typename List<T>::Item* List<T>::PushFront(const T& value)
{
    Item* const pNewItem = PushFront();
    pNewItem->Value = value;
    return pNewItem;
}

template<typename T>
void List<T>::PopBack()
{
    D3D12MA_HEAVY_ASSERT(m_Count > 0);
    Item* const pBackItem = m_pBack;
    Item* const pPrevItem = pBackItem->pPrev;
    if(pPrevItem != NULL)
    {
        pPrevItem->pNext = NULL;
    }
    m_pBack = pPrevItem;
    m_ItemAllocator.Free(pBackItem);
    --m_Count;
}

template<typename T>
void List<T>::PopFront()
{
    D3D12MA_HEAVY_ASSERT(m_Count > 0);
    Item* const pFrontItem = m_pFront;
    Item* const pNextItem = pFrontItem->pNext;
    if(pNextItem != NULL)
    {
        pNextItem->pPrev = NULL;
    }
    m_pFront = pNextItem;
    m_ItemAllocator.Free(pFrontItem);
    --m_Count;
}

template<typename T>
void List<T>::Remove(Item* pItem)
{
    D3D12MA_HEAVY_ASSERT(pItem != NULL);
    D3D12MA_HEAVY_ASSERT(m_Count > 0);

    if(pItem->pPrev != NULL)
    {
        pItem->pPrev->pNext = pItem->pNext;
    }
    else
    {
        D3D12MA_HEAVY_ASSERT(m_pFront == pItem);
        m_pFront = pItem->pNext;
    }

    if(pItem->pNext != NULL)
    {
        pItem->pNext->pPrev = pItem->pPrev;
    }
    else
    {
        D3D12MA_HEAVY_ASSERT(m_pBack == pItem);
        m_pBack = pItem->pPrev;
    }

    m_ItemAllocator.Free(pItem);
    --m_Count;
}

template<typename T>
typename List<T>::Item* List<T>::InsertBefore(Item* pItem)
{
    if(pItem != NULL)
    {
        Item* const prevItem = pItem->pPrev;
        Item* const newItem = m_ItemAllocator.Alloc();
        newItem->pPrev = prevItem;
        newItem->pNext = pItem;
        pItem->pPrev = newItem;
        if(prevItem != NULL)
        {
            prevItem->pNext = newItem;
        }
        else
        {
            D3D12MA_HEAVY_ASSERT(m_pFront == pItem);
            m_pFront = newItem;
        }
        ++m_Count;
        return newItem;
    }
    else
    {
        return PushBack();
    }
}

template<typename T>
typename List<T>::Item* List<T>::InsertAfter(Item* pItem)
{
    if(pItem != NULL)
    {
        Item* const nextItem = pItem->pNext;
        Item* const newItem = m_ItemAllocator.Alloc();
        newItem->pNext = nextItem;
        newItem->pPrev = pItem;
        pItem->pNext = newItem;
        if(nextItem != NULL)
        {
            nextItem->pPrev = newItem;
        }
        else
        {
            D3D12MA_HEAVY_ASSERT(m_pBack == pItem);
            m_pBack = newItem;
        }
        ++m_Count;
        return newItem;
    }
    else
        return PushFront();
}

template<typename T>
typename List<T>::Item* List<T>::InsertBefore(Item* pItem, const T& value)
{
    Item* const newItem = InsertBefore(pItem);
    newItem->Value = value;
    return newItem;
}

template<typename T>
typename List<T>::Item* List<T>::InsertAfter(Item* pItem, const T& value)
{
    Item* const newItem = InsertAfter(pItem);
    newItem->Value = value;
    return newItem;
}

////////////////////////////////////////////////////////////////////////////////
// Private class IntrusiveLinkedList

/*
Expected interface of ItemTypeTraits:
struct MyItemTypeTraits
{
    using ItemType = MyItem;
    static ItemType* GetPrev(const ItemType* item) { return item->myPrevPtr; }
    static ItemType* GetNext(const ItemType* item) { return item->myNextPtr; }
    static ItemType*& AccessPrev(ItemType* item) { return item->myPrevPtr; }
    static ItemType*& AccessNext(ItemType* item) { return item->myNextPtr; }
};
*/
template<typename ItemTypeTraits>
class IntrusiveLinkedList
{
public:
    using ItemType = typename ItemTypeTraits::ItemType;
    static ItemType* GetPrev(const ItemType* item) { return ItemTypeTraits::GetPrev(item); }
    static ItemType* GetNext(const ItemType* item) { return ItemTypeTraits::GetNext(item); }
    // Movable, not copyable.
    IntrusiveLinkedList() { }
    IntrusiveLinkedList(const IntrusiveLinkedList<ItemTypeTraits>& src) = delete;
    IntrusiveLinkedList(IntrusiveLinkedList<ItemTypeTraits>&& src) :
        m_Front(src.m_Front), m_Back(src.m_Back), m_Count(src.m_Count)
    {
        src.m_Front = src.m_Back = NULL;
        src.m_Count = 0;
    }
    ~IntrusiveLinkedList()
    {
        D3D12MA_HEAVY_ASSERT(IsEmpty());
    }
    IntrusiveLinkedList<ItemTypeTraits>& operator=(const IntrusiveLinkedList<ItemTypeTraits>& src) = delete;
    IntrusiveLinkedList<ItemTypeTraits>& operator=(IntrusiveLinkedList<ItemTypeTraits>&& src)
    {
        if(&src != this)
        {
            D3D12MA_HEAVY_ASSERT(IsEmpty());
            m_Front = src.m_Front;
            m_Back = src.m_Back;
            m_Count = src.m_Count;
            src.m_Front = src.m_Back = NULL;
            src.m_Count = 0;
        }
        return *this;
    }
    void RemoveAll()
    {
        if(!IsEmpty())
        {
            ItemType* item = m_Back;
            while(item != NULL)
            {
                ItemType* const prevItem = ItemTypeTraits::AccessPrev(item);
                ItemTypeTraits::AccessPrev(item) = NULL;
                ItemTypeTraits::AccessNext(item) = NULL;
                item = prevItem;
            }
            m_Front = NULL;
            m_Back = NULL;
            m_Count = 0;
        }
    }
    size_t GetCount() const { return m_Count; }
    bool IsEmpty() const { return m_Count == 0; }
    ItemType* Front() { return m_Front; }
    const ItemType* Front() const { return m_Front; }
    ItemType* Back() { return m_Back; }
    const ItemType* Back() const { return m_Back; }
    void PushBack(ItemType* item)
    {
        D3D12MA_HEAVY_ASSERT(ItemTypeTraits::GetPrev(item) == NULL && ItemTypeTraits::GetNext(item) == NULL);
        if(IsEmpty())
        {
            m_Front = item;
            m_Back = item;
            m_Count = 1;
        }
        else
        {
            ItemTypeTraits::AccessPrev(item) = m_Back;
            ItemTypeTraits::AccessNext(m_Back) = item;
            m_Back = item;
            ++m_Count;
        }
    }
    void PushFront(ItemType* item)
    {
        D3D12MA_HEAVY_ASSERT(ItemTypeTraits::GetPrev(item) == NULL && ItemTypeTraits::GetNext(item) == NULL);
        if(IsEmpty())
        {
            m_Front = item;
            m_Back = item;
            m_Count = 1;
        }
        else
        {
            ItemTypeTraits::AccessNext(item) = m_Front;
            ItemTypeTraits::AccessPrev(m_Front) = item;
            m_Front = item;
            ++m_Count;
        }
    }
    ItemType* PopBack()
    {
        D3D12MA_HEAVY_ASSERT(m_Count > 0);
        ItemType* const backItem = m_Back;
        ItemType* const prevItem = ItemTypeTraits::GetPrev(backItem);
        if(prevItem != NULL)
        {
            ItemTypeTraits::AccessNext(prevItem) = NULL;
        }
        m_Back = prevItem;
        --m_Count;
        ItemTypeTraits::AccessPrev(backItem) = NULL;
        ItemTypeTraits::AccessNext(backItem) = NULL;
        return backItem;
    }
    ItemType* PopFront()
    {
        D3D12MA_HEAVY_ASSERT(m_Count > 0);
        ItemType* const frontItem = m_Front;
        ItemType* const nextItem = ItemTypeTraits::GetNext(frontItem);
        if(nextItem != NULL)
        {
            ItemTypeTraits::AccessPrev(nextItem) = NULL;
        }
        m_Front = nextItem;
        --m_Count;
        ItemTypeTraits::AccessPrev(frontItem) = NULL;
        ItemTypeTraits::AccessNext(frontItem) = NULL;
        return frontItem;
    }

    // MyItem can be null - it means PushBack.
    void InsertBefore(ItemType* existingItem, ItemType* newItem)
    {
        D3D12MA_HEAVY_ASSERT(newItem != NULL && ItemTypeTraits::GetPrev(newItem) == NULL && ItemTypeTraits::GetNext(newItem) == NULL);
        if(existingItem != NULL)
        {
            ItemType* const prevItem = ItemTypeTraits::GetPrev(existingItem);
            ItemTypeTraits::AccessPrev(newItem) = prevItem;
            ItemTypeTraits::AccessNext(newItem) = existingItem;
            ItemTypeTraits::AccessPrev(existingItem) = newItem;
            if(prevItem != NULL)
            {
                ItemTypeTraits::AccessNext(prevItem) = newItem;
            }
            else
            {
                D3D12MA_HEAVY_ASSERT(m_Front == existingItem);
                m_Front = newItem;
            }
            ++m_Count;
        }
        else
            PushBack(newItem);
    }
    // MyItem can be null - it means PushFront.
    void InsertAfter(ItemType* existingItem, ItemType* newItem)
    {
        D3D12MA_HEAVY_ASSERT(newItem != NULL && ItemTypeTraits::GetPrev(newItem) == NULL && ItemTypeTraits::GetNext(newItem) == NULL);
        if(existingItem != NULL)
        {
            ItemType* const nextItem = ItemTypeTraits::GetNext(existingItem);
            ItemTypeTraits::AccessNext(newItem) = nextItem;
            ItemTypeTraits::AccessPrev(newItem) = existingItem;
            ItemTypeTraits::AccessNext(existingItem) = newItem;
            if(nextItem != NULL)
            {
                ItemTypeTraits::AccessPrev(nextItem) = newItem;
            }
            else
            {
                D3D12MA_HEAVY_ASSERT(m_Back == existingItem);
                m_Back = newItem;
            }
            ++m_Count;
        }
        else
            return PushFront(newItem);
    }
    void Remove(ItemType* item)
    {
        D3D12MA_HEAVY_ASSERT(item != NULL && m_Count > 0);
        if(ItemTypeTraits::GetPrev(item) != NULL)
        {
            ItemTypeTraits::AccessNext(ItemTypeTraits::AccessPrev(item)) = ItemTypeTraits::GetNext(item);
        }
        else
        {
            D3D12MA_HEAVY_ASSERT(m_Front == item);
            m_Front = ItemTypeTraits::GetNext(item);
        }

        if(ItemTypeTraits::GetNext(item) != NULL)
        {
            ItemTypeTraits::AccessPrev(ItemTypeTraits::AccessNext(item)) = ItemTypeTraits::GetPrev(item);
        }
        else
        {
            D3D12MA_HEAVY_ASSERT(m_Back == item);
            m_Back = ItemTypeTraits::GetPrev(item);
        }
        ItemTypeTraits::AccessPrev(item) = NULL;
        ItemTypeTraits::AccessNext(item) = NULL;
        --m_Count;
    }
private:
    ItemType* m_Front = NULL;
    ItemType* m_Back = NULL;
    size_t m_Count = 0;
};

////////////////////////////////////////////////////////////////////////////////
// Private class AllocationObjectAllocator definition

/*
Thread-safe wrapper over PoolAllocator free list, for allocation of Allocation objects.
*/
class AllocationObjectAllocator
{
    D3D12MA_CLASS_NO_COPY(AllocationObjectAllocator);
public:
    AllocationObjectAllocator(const ALLOCATION_CALLBACKS& allocationCallbacks);

    template<typename... Types> Allocation* Allocate(Types... args);
    void Free(Allocation* alloc);

private:
    D3D12MA_MUTEX m_Mutex;
    PoolAllocator<Allocation> m_Allocator;
};

////////////////////////////////////////////////////////////////////////////////
// Private class BlockMetadata and derived classes - declarations

enum SuballocationType
{
    SUBALLOCATION_TYPE_FREE = 0,
    SUBALLOCATION_TYPE_ALLOCATION = 1,
};

/*
Represents a region of NormalBlock that is either assigned and returned as
allocated memory block or free.
*/
struct Suballocation
{
    UINT64 offset;
    UINT64 size;
    void* userData;
    SuballocationType type;
};

// Comparator for offsets.
struct SuballocationOffsetLess
{
    bool operator()(const Suballocation& lhs, const Suballocation& rhs) const
    {
        return lhs.offset < rhs.offset;
    }
};
struct SuballocationOffsetGreater
{
    bool operator()(const Suballocation& lhs, const Suballocation& rhs) const
    {
        return lhs.offset > rhs.offset;
    }
};

using SuballocationList = List<Suballocation>;

struct SuballocationItemSizeLess
{
    bool operator()(const SuballocationList::iterator lhs, const SuballocationList::iterator rhs) const
    {
        return lhs->size < rhs->size;
    }
    bool operator()(const SuballocationList::iterator lhs, UINT64 rhsSize) const
    {
        return lhs->size < rhsSize;
    }
};

/*
Parameters of planned allocation inside a NormalBlock.
*/
struct AllocationRequest
{
    AllocHandle allocHandle;
    UINT64 size;
    UINT64 algorithmData;
    UINT64 sumFreeSize; // Sum size of free items that overlap with proposed allocation.
    UINT64 sumItemSize; // Sum size of items to make lost that overlap with proposed allocation.
    SuballocationList::iterator item;
    BOOL zeroInitialized;
};

/*
Keeps track of the range of bytes that are surely initialized with zeros.
Everything outside of it is considered uninitialized memory that may contain
garbage data.

The range is left-inclusive.
*/
class ZeroInitializedRange
{
public:
    void Reset(UINT64 size)
    {
        D3D12MA_ASSERT(size > 0);
        m_ZeroBeg = 0;
        m_ZeroEnd = size;
    }

    BOOL IsRangeZeroInitialized(UINT64 beg, UINT64 end) const
    {
        D3D12MA_ASSERT(beg < end);
        return m_ZeroBeg <= beg && end <= m_ZeroEnd;
    }

    void MarkRangeAsUsed(UINT64 usedBeg, UINT64 usedEnd)
    {
        D3D12MA_ASSERT(usedBeg < usedEnd);
        // No new bytes marked.
        if(usedEnd <= m_ZeroBeg || m_ZeroEnd <= usedBeg)
        {
            return;
        }
        // All bytes marked.
        if(usedBeg <= m_ZeroBeg && m_ZeroEnd <= usedEnd)
        {
            m_ZeroBeg = m_ZeroEnd = 0;
        }
        // Some bytes marked.
        else
        {
            const UINT64 remainingZeroBefore = usedBeg > m_ZeroBeg ? usedBeg - m_ZeroBeg : 0;
            const UINT64 remainingZeroAfter  = usedEnd < m_ZeroEnd ? m_ZeroEnd - usedEnd : 0;
            D3D12MA_ASSERT(remainingZeroBefore > 0 || remainingZeroAfter > 0);
            if(remainingZeroBefore > remainingZeroAfter)
            {
                m_ZeroEnd = usedBeg;
            }
            else
            {
                m_ZeroBeg = usedEnd;
            }
        }
    }

private:
    UINT64 m_ZeroBeg = 0, m_ZeroEnd = 0;
};

/*
Data structure used for bookkeeping of allocations and unused ranges of memory
in a single ID3D12Heap memory block.
*/
class BlockMetadata
{
public:
    BlockMetadata(const ALLOCATION_CALLBACKS* allocationCallbacks, bool isVirtual);
    virtual ~BlockMetadata() = default;
    virtual void Init(UINT64 size) { m_Size = size; }

    // Validates all data structures inside this object. If not valid, returns false.
    virtual bool Validate() const = 0;
    UINT64 GetSize() const { return m_Size; }
    bool IsVirtual() const { return m_IsVirtual; }
    virtual size_t GetAllocationCount() const = 0;
    virtual UINT64 GetSumFreeSize() const = 0;
    virtual UINT64 GetAllocationOffset(AllocHandle allocHandle) const = 0;
    // Returns true if this block is empty - contains only single free suballocation.
    virtual bool IsEmpty() const = 0;

    virtual void GetAllocationInfo(AllocHandle allocHandle, VIRTUAL_ALLOCATION_INFO& outInfo) const = 0;

    // Tries to find a place for suballocation with given parameters inside this block.
    // If succeeded, fills pAllocationRequest and returns true.
    // If failed, returns false.
    virtual bool CreateAllocationRequest(
        UINT64 allocSize,
        UINT64 allocAlignment,
        bool upperAddress,
        AllocationRequest* pAllocationRequest) = 0;

    // Makes actual allocation based on request. Request must already be checked and valid.
    virtual void Alloc(
        const AllocationRequest& request,
        UINT64 allocSize,
        void* userData) = 0;

    virtual void Free(AllocHandle allocHandle) = 0;
    // Frees all allocations.
    // Careful! Don't call it if there are Allocation objects owned by pUserData of of cleared allocations!
    virtual void Clear() = 0;

    virtual void SetAllocationUserData(AllocHandle allocHandle, void* userData) = 0;

    virtual void CalcAllocationStatInfo(StatInfo& outInfo) const = 0;
    virtual void WriteAllocationInfoToJson(JsonWriter& json) const = 0;

protected:
    const ALLOCATION_CALLBACKS* GetAllocs() const { return m_pAllocationCallbacks; }
    UINT64 GetDebugMargin() const { return IsVirtual() ? 0 : D3D12MA_DEBUG_MARGIN; }

    void PrintDetailedMap_Begin(JsonWriter& json,
        UINT64 unusedBytes,
        size_t allocationCount,
        size_t unusedRangeCount) const;
    void PrintDetailedMap_Allocation(JsonWriter& json,
        UINT64 offset, UINT64 size, void* userData) const;
    void PrintDetailedMap_UnusedRange(JsonWriter& json,
        UINT64 offset, UINT64 size) const;
    void PrintDetailedMap_End(JsonWriter& json) const;

private:
    UINT64 m_Size;
    bool m_IsVirtual;
    const ALLOCATION_CALLBACKS* m_pAllocationCallbacks;

    D3D12MA_CLASS_NO_COPY(BlockMetadata);
};

class BlockMetadata_Generic : public BlockMetadata
{
public:
    BlockMetadata_Generic(const ALLOCATION_CALLBACKS* allocationCallbacks, bool isVirtual);
    virtual ~BlockMetadata_Generic();

    size_t GetAllocationCount() const override { return m_Suballocations.size() - m_FreeCount; }
    UINT64 GetSumFreeSize() const override { return m_SumFreeSize; }
    UINT64 GetAllocationOffset(AllocHandle allocHandle) const override { return (UINT64)allocHandle - 1; }

    void Init(UINT64 size) override;
    bool Validate() const override;
    bool IsEmpty() const override;
    void GetAllocationInfo(AllocHandle allocHandle, VIRTUAL_ALLOCATION_INFO& outInfo) const override;

    bool CreateAllocationRequest(
        UINT64 allocSize,
        UINT64 allocAlignment,
        bool upperAddress,
        AllocationRequest* pAllocationRequest) override;

    void Alloc(
        const AllocationRequest& request,
        UINT64 allocSize,
        void* userData) override;

    void Free(AllocHandle allocHandle) override;
    void Clear() override;

    void SetAllocationUserData(AllocHandle allocHandle, void* userData) override;

    void CalcAllocationStatInfo(StatInfo& outInfo) const override;
    void WriteAllocationInfoToJson(JsonWriter& json) const override;

private:
    UINT m_FreeCount;
    UINT64 m_SumFreeSize;
    SuballocationList m_Suballocations;
    // Suballocations that are free and have size greater than certain threshold.
    // Sorted by size, ascending.
    Vector<SuballocationList::iterator> m_FreeSuballocationsBySize;
    ZeroInitializedRange m_ZeroInitializedRange;

    SuballocationList::const_iterator FindAtOffset(UINT64 offset) const;
    bool ValidateFreeSuballocationList() const;

    // Checks if requested suballocation with given parameters can be placed in given pFreeSuballocItem.
    // If yes, fills pOffset and returns true. If no, returns false.
    bool CheckAllocation(
        UINT64 allocSize,
        UINT64 allocAlignment,
        SuballocationList::const_iterator suballocItem,
        AllocHandle* pAllocHandle,
        UINT64* pSumFreeSize,
        UINT64* pSumItemSize,
        BOOL *pZeroInitialized) const;
    // Given free suballocation, it merges it with following one, which must also be free.
    void MergeFreeWithNext(SuballocationList::iterator item);
    // Releases given suballocation, making it free.
    // Merges it with adjacent free suballocations if applicable.
    // Returns iterator to new free suballocation at this place.
    SuballocationList::iterator FreeSuballocation(SuballocationList::iterator suballocItem);
    // Given free suballocation, it inserts it into sorted list of
    // m_FreeSuballocationsBySize if it's suitable.
    void RegisterFreeSuballocation(SuballocationList::iterator item);
    // Given free suballocation, it removes it from sorted list of
    // m_FreeSuballocationsBySize if it's suitable.
    void UnregisterFreeSuballocation(SuballocationList::iterator item);

    D3D12MA_CLASS_NO_COPY(BlockMetadata_Generic)
};

class BlockMetadata_Linear : public BlockMetadata
{
public:
    BlockMetadata_Linear(const ALLOCATION_CALLBACKS* allocationCallbacks, bool isVirtual);
    virtual ~BlockMetadata_Linear() = default;

    UINT64 GetSumFreeSize() const override { return m_SumFreeSize; }
    bool IsEmpty() const override { return GetAllocationCount() == 0; }
    UINT64 GetAllocationOffset(AllocHandle allocHandle) const override { return (UINT64)allocHandle - 1; };

    void Init(UINT64 size) override;
    bool Validate() const override;
    size_t GetAllocationCount() const override;
    void GetAllocationInfo(AllocHandle allocHandle, VIRTUAL_ALLOCATION_INFO& outInfo) const override;

    bool CreateAllocationRequest(
        UINT64 allocSize,
        UINT64 allocAlignment,
        bool upperAddress,
        AllocationRequest* pAllocationRequest) override;

    void Alloc(
        const AllocationRequest& request,
        UINT64 allocSize,
        void* userData) override;

    void Free(AllocHandle allocHandle) override;
    void Clear() override;

    void SetAllocationUserData(AllocHandle allocHandle, void* userData) override;

    void CalcAllocationStatInfo(StatInfo& outInfo) const override;
    void WriteAllocationInfoToJson(JsonWriter& json) const override;

private:
    /*
    There are two suballocation vectors, used in ping-pong way.
    The one with index m_1stVectorIndex is called 1st.
    The one with index (m_1stVectorIndex ^ 1) is called 2nd.
    2nd can be non-empty only when 1st is not empty.
    When 2nd is not empty, m_2ndVectorMode indicates its mode of operation.
    */
    typedef Vector<Suballocation> SuballocationVectorType;

    enum ALLOC_REQUEST_TYPE
    {
        ALLOC_REQUEST_UPPER_ADDRESS,
        ALLOC_REQUEST_END_OF_1ST,
        ALLOC_REQUEST_END_OF_2ND,
    };

    enum SECOND_VECTOR_MODE
    {
        SECOND_VECTOR_EMPTY,
        /*
        Suballocations in 2nd vector are created later than the ones in 1st, but they
        all have smaller offset.
        */
        SECOND_VECTOR_RING_BUFFER,
        /*
        Suballocations in 2nd vector are upper side of double stack.
        They all have offsets higher than those in 1st vector.
        Top of this stack means smaller offsets, but higher indices in this vector.
        */
        SECOND_VECTOR_DOUBLE_STACK,
    };

    UINT64 m_SumFreeSize;
    SuballocationVectorType m_Suballocations0, m_Suballocations1;
    UINT32 m_1stVectorIndex;
    SECOND_VECTOR_MODE m_2ndVectorMode;
    // Number of items in 1st vector with hAllocation = null at the beginning.
    size_t m_1stNullItemsBeginCount;
    // Number of other items in 1st vector with hAllocation = null somewhere in the middle.
    size_t m_1stNullItemsMiddleCount;
    // Number of items in 2nd vector with hAllocation = null.
    size_t m_2ndNullItemsCount;

    SuballocationVectorType& AccessSuballocations1st() { return m_1stVectorIndex ? m_Suballocations1 : m_Suballocations0; }
    SuballocationVectorType& AccessSuballocations2nd() { return m_1stVectorIndex ? m_Suballocations0 : m_Suballocations1; }
    const SuballocationVectorType& AccessSuballocations1st() const { return m_1stVectorIndex ? m_Suballocations1 : m_Suballocations0; }
    const SuballocationVectorType& AccessSuballocations2nd() const { return m_1stVectorIndex ? m_Suballocations0 : m_Suballocations1; }

    Suballocation& FindSuballocation(UINT64 offset) const;
    bool ShouldCompact1st() const;
    void CleanupAfterFree();

    bool CreateAllocationRequest_LowerAddress(
        UINT64 allocSize,
        UINT64 allocAlignment,
        AllocationRequest* pAllocationRequest);
    bool CreateAllocationRequest_UpperAddress(
        UINT64 allocSize,
        UINT64 allocAlignment,
        AllocationRequest* pAllocationRequest);

    D3D12MA_CLASS_NO_COPY(BlockMetadata_Linear)
};

class BlockMetadata_TLSF : public BlockMetadata
{
public:
    BlockMetadata_TLSF(const ALLOCATION_CALLBACKS* allocationCallbacks, bool isVirtual);
    virtual ~BlockMetadata_TLSF();

    size_t GetAllocationCount() const override { return m_AllocCount; }
    UINT64 GetSumFreeSize() const override { return m_BlocksFreeSize + m_NullBlock->size; }
    bool IsEmpty() const override { return m_NullBlock->offset == 0; }
    UINT64 GetAllocationOffset(AllocHandle allocHandle) const override { return ((Block*)allocHandle)->offset; };

    void Init(UINT64 size) override;
    bool Validate() const override;
    void GetAllocationInfo(AllocHandle allocHandle, VIRTUAL_ALLOCATION_INFO& outInfo) const override;

    bool CreateAllocationRequest(
        UINT64 allocSize,
        UINT64 allocAlignment,
        bool upperAddress,
        AllocationRequest* pAllocationRequest) override;

    void Alloc(
        const AllocationRequest& request,
        UINT64 allocSize,
        void* userData) override;

    void Free(AllocHandle allocHandle) override;
    void Clear() override;

    void SetAllocationUserData(AllocHandle allocHandle, void* userData) override;

    void CalcAllocationStatInfo(StatInfo& outInfo) const override;
    void WriteAllocationInfoToJson(JsonWriter& json) const override;

private:
    // According to original paper it should be preferable 4 or 5:
    // M. Masmano, I. Ripoll, A. Crespo, and J. Real "TLSF: a New Dynamic Memory Allocator for Real-Time Systems"
    // http://www.gii.upv.es/tlsf/files/ecrts04_tlsf.pdf
    static const UINT8 SECOND_LEVEL_INDEX = 5;
    static const UINT16 SMALL_BUFFER_SIZE = 256;
    static const UINT INITIAL_BLOCK_ALLOC_COUNT = 16;
    static const UINT8 MEMORY_CLASS_SHIFT = 7;
    static const UINT8 MAX_MEMORY_CLASSES = 65 - MEMORY_CLASS_SHIFT;

    class Block
    {
    public:
        UINT64 offset;
        UINT64 size;
        Block* prevPhysical;
        Block* nextPhysical;

        void MarkFree() { prevFree = NULL; }
        void MarkTaken() { prevFree = this; }
        bool IsFree() const { return prevFree != this; }
        void*& UserData() { D3D12MA_HEAVY_ASSERT(!IsFree()); return userData; }
        Block*& PrevFree() { return prevFree; }
        Block*& NextFree() { D3D12MA_HEAVY_ASSERT(IsFree()); return nextFree; }

    private:
        Block* prevFree; // Address of the same block here indicates that block is taken
        union
        {
            Block* nextFree;
            void* userData;
        };
    };
    
    size_t m_AllocCount;
    // Total number of free blocks besides null block
    size_t m_BlocksFreeCount;
    // Total size of free blocks excluding null block
    UINT64 m_BlocksFreeSize;
    UINT32 m_IsFreeBitmap;
    UINT8 m_MemoryClasses;
    UINT32 m_InnerIsFreeBitmap[MAX_MEMORY_CLASSES];
    UINT32 m_ListsCount;
    /*
    * 0: 0-3 lists for small buffers
    * 1+: 0-(2^SLI-1) lists for normal buffers
    */
    Block** m_FreeList;
    PoolAllocator<Block> m_BlockAllocator;
    Block* m_NullBlock;

    UINT8 SizeToMemoryClass(UINT64 size) const;
    UINT16 SizeToSecondIndex(UINT64 size, UINT8 memoryClass) const;
    UINT32 GetListIndex(UINT8 memoryClass, UINT16 secondIndex) const;
    UINT32 GetListIndex(UINT64 size) const;

    void RemoveFreeBlock(Block* block);
    void InsertFreeBlock(Block* block);
    void MergeBlock(Block* block, Block* prev);

    Block* FindFreeBlock(UINT64 size, UINT32& listIndex) const;
    bool CheckBlock(
        Block& block,
        UINT32 listIndex,
        UINT64 allocSize,
        UINT64 allocAlignment,
        AllocationRequest* pAllocationRequest);

    D3D12MA_CLASS_NO_COPY(BlockMetadata_TLSF)
};

////////////////////////////////////////////////////////////////////////////////
// Private class MemoryBlock definition

/*
Represents a single block of device memory (heap).
Base class for inheritance.
Thread-safety: This class must be externally synchronized.
*/
class MemoryBlock
{
public:
    MemoryBlock(
        AllocatorPimpl* allocator,
        const D3D12_HEAP_PROPERTIES& heapProps,
        D3D12_HEAP_FLAGS heapFlags,
        UINT64 size,
        UINT id);
    virtual ~MemoryBlock();
    // Creates the ID3D12Heap.

    const D3D12_HEAP_PROPERTIES& GetHeapProperties() const { return m_HeapProps; }
    D3D12_HEAP_FLAGS GetHeapFlags() const { return m_HeapFlags; }
    UINT64 GetSize() const { return m_Size; }
    UINT GetId() const { return m_Id; }
    ID3D12Heap* GetHeap() const { return m_Heap; }

protected:
    AllocatorPimpl* const m_Allocator;
    const D3D12_HEAP_PROPERTIES m_HeapProps;
    const D3D12_HEAP_FLAGS m_HeapFlags;
    const UINT64 m_Size;
    const UINT m_Id;

    HRESULT Init(ID3D12ProtectedResourceSession* pProtectedSession);

private:
    ID3D12Heap* m_Heap = NULL;

    D3D12MA_CLASS_NO_COPY(MemoryBlock)
};

////////////////////////////////////////////////////////////////////////////////
// Private class NormalBlock definition

/*
Represents a single block of device memory (heap) with all the data about its
regions (aka suballocations, Allocation), assigned and free.
Thread-safety: This class must be externally synchronized.
*/
class NormalBlock : public MemoryBlock
{
public:
    BlockMetadata* m_pMetadata;

    NormalBlock(
        AllocatorPimpl* allocator,
        BlockVector* blockVector,
        const D3D12_HEAP_PROPERTIES& heapProps,
        D3D12_HEAP_FLAGS heapFlags,
        UINT64 size,
        UINT id);
    virtual ~NormalBlock();

    BlockVector* GetBlockVector() const { return m_BlockVector; }

    // 'algorithm' should be one of the *_ALGORITHM_* flags in enums POOL_FLAGS or VIRTUAL_BLOCK_FLAGS
    HRESULT Init(UINT32 algorithm, ID3D12ProtectedResourceSession* pProtectedSession);

    // Validates all data structures inside this object. If not valid, returns false.
    bool Validate() const;

private:
    BlockVector* m_BlockVector;

    D3D12MA_CLASS_NO_COPY(NormalBlock)
};

////////////////////////////////////////////////////////////////////////////////
// Private class CommittedAllocationList definition

struct CommittedAllocationListItemTraits
{
    using ItemType = Allocation;
    static ItemType* GetPrev(const ItemType* item)
    {
        D3D12MA_ASSERT(item->m_PackedData.GetType() == Allocation::TYPE_COMMITTED || item->m_PackedData.GetType() == Allocation::TYPE_HEAP);
        return item->m_Committed.prev;
    }
    static ItemType* GetNext(const ItemType* item)
    {
        D3D12MA_ASSERT(item->m_PackedData.GetType() == Allocation::TYPE_COMMITTED || item->m_PackedData.GetType() == Allocation::TYPE_HEAP);
        return item->m_Committed.next;
    }
    static ItemType*& AccessPrev(ItemType* item)
    {
        D3D12MA_ASSERT(item->m_PackedData.GetType() == Allocation::TYPE_COMMITTED || item->m_PackedData.GetType() == Allocation::TYPE_HEAP);
        return item->m_Committed.prev;
    }
    static ItemType*& AccessNext(ItemType* item)
    {
        D3D12MA_ASSERT(item->m_PackedData.GetType() == Allocation::TYPE_COMMITTED || item->m_PackedData.GetType() == Allocation::TYPE_HEAP);
        return item->m_Committed.next;
    }
};

/*
Stores linked list of Allocation objects that are of TYPE_COMMITTED or TYPE_HEAP.
Thread-safe, synchronized internally.
*/
class CommittedAllocationList
{
public:
    CommittedAllocationList();
    void Init(bool useMutex, D3D12_HEAP_TYPE heapType, PoolPimpl* pool);
    ~CommittedAllocationList();

    D3D12_HEAP_TYPE GetHeapType() const { return m_HeapType; }
    
    void CalculateStats(StatInfo& outStats);
    // Writes JSON array with the list of allocations.
    void BuildStatsString(JsonWriter& json);

    void Register(Allocation* alloc);
    void Unregister(Allocation* alloc);

private:
    bool m_UseMutex = true;
    D3D12_HEAP_TYPE m_HeapType = D3D12_HEAP_TYPE_CUSTOM;
    PoolPimpl* m_Pool = NULL;

    D3D12MA_RW_MUTEX m_Mutex;
    using CommittedAllocationLinkedList = IntrusiveLinkedList<CommittedAllocationListItemTraits>;
    CommittedAllocationLinkedList m_AllocationList;
};

////////////////////////////////////////////////////////////////////////////////
// Private class BlockVector definition

/*
Sequence of NormalBlock. Represents memory blocks allocated for a specific
heap type and possibly resource type (if only Tier 1 is supported).

Synchronized internally with a mutex.
*/
class BlockVector
{
    D3D12MA_CLASS_NO_COPY(BlockVector)
public:
    BlockVector(
        AllocatorPimpl* hAllocator,
        const D3D12_HEAP_PROPERTIES& heapProps,
        D3D12_HEAP_FLAGS heapFlags,
        UINT64 preferredBlockSize,
        size_t minBlockCount,
        size_t maxBlockCount,
        bool explicitBlockSize,
        UINT64 minAllocationAlignment,
        UINT32 algorithm,
        ID3D12ProtectedResourceSession* pProtectedSession);
    ~BlockVector();

    HRESULT CreateMinBlocks();

    const D3D12_HEAP_PROPERTIES& GetHeapProperties() const { return m_HeapProps; }
    UINT64 GetPreferredBlockSize() const { return m_PreferredBlockSize; }

    bool IsEmpty();

    HRESULT Allocate(
        UINT64 size,
        UINT64 alignment,
        const ALLOCATION_DESC& allocDesc,
        size_t allocationCount,
        Allocation** pAllocations);

    void Free(
        Allocation* hAllocation);

    HRESULT CreateResource(
        UINT64 size,
        UINT64 alignment,
        const ALLOCATION_DESC& allocDesc,
        const D3D12_RESOURCE_DESC& resourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState,
        const D3D12_CLEAR_VALUE *pOptimizedClearValue,
        Allocation** ppAllocation,
        REFIID riidResource,
        void** ppvResource);

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
    HRESULT CreateResource2(
        UINT64 size,
        UINT64 alignment,
        const ALLOCATION_DESC& allocDesc,
        const D3D12_RESOURCE_DESC1& resourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState,
        const D3D12_CLEAR_VALUE *pOptimizedClearValue,
        Allocation** ppAllocation,
        REFIID riidResource,
        void** ppvResource);
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

    void AddStats(StatInfo& outStats);
    void AddStats(Stats& outStats);

    void WriteBlockInfoToJson(JsonWriter& json);

private:
    AllocatorPimpl* const m_hAllocator;
    const D3D12_HEAP_PROPERTIES m_HeapProps;
    const D3D12_HEAP_FLAGS m_HeapFlags;
    const UINT64 m_PreferredBlockSize;
    const size_t m_MinBlockCount;
    const size_t m_MaxBlockCount;
    const bool m_ExplicitBlockSize;
    const UINT64 m_MinAllocationAlignment;
    const UINT32 m_Algorithm;
    ID3D12ProtectedResourceSession* const m_ProtectedSession;
    /* There can be at most one allocation that is completely empty - a
    hysteresis to avoid pessimistic case of alternating creation and destruction
    of a VkDeviceMemory. */
    bool m_HasEmptyBlock;
    D3D12MA_RW_MUTEX m_Mutex;
    // Incrementally sorted by sumFreeSize, ascending.
    Vector<NormalBlock*> m_Blocks;
    UINT m_NextBlockId;

    UINT64 CalcSumBlockSize() const;
    UINT64 CalcMaxBlockSize() const;

    // Finds and removes given block from vector.
    void Remove(NormalBlock* pBlock);

    // Performs single step in sorting m_Blocks. They may not be fully sorted
    // after this call.
    void IncrementallySortBlocks();

    HRESULT AllocatePage(
        UINT64 size,
        UINT64 alignment,
        const ALLOCATION_DESC& allocDesc,
        Allocation** pAllocation);

    HRESULT AllocateFromBlock(
        NormalBlock* pBlock,
        UINT64 size,
        UINT64 alignment,
        ALLOCATION_FLAGS allocFlags,
        Allocation** pAllocation);

    HRESULT CreateBlock(
        UINT64 blockSize,
        size_t* pNewBlockIndex);
};

////////////////////////////////////////////////////////////////////////////////
// Private class AllocatorPimpl definition

static const UINT STANDARD_HEAP_TYPE_COUNT = 3; // Only DEFAULT, UPLOAD, READBACK.
static const UINT DEFAULT_POOL_MAX_COUNT = 9;

struct CurrentBudgetData
{
    D3D12MA_ATOMIC_UINT64 m_BlockBytes[HEAP_TYPE_COUNT];
    D3D12MA_ATOMIC_UINT64 m_AllocationBytes[HEAP_TYPE_COUNT];

    D3D12MA_ATOMIC_UINT32 m_OperationsSinceBudgetFetch;
    D3D12MA_RW_MUTEX m_BudgetMutex;
    UINT64 m_D3D12UsageLocal, m_D3D12UsageNonLocal;
    UINT64 m_D3D12BudgetLocal, m_D3D12BudgetNonLocal;
    UINT64 m_BlockBytesAtBudgetFetch[HEAP_TYPE_COUNT];

    CurrentBudgetData()
    {
        for(UINT i = 0; i < HEAP_TYPE_COUNT; ++i)
        {
            m_BlockBytes[i] = 0;
            m_AllocationBytes[i] = 0;
            m_BlockBytesAtBudgetFetch[i] = 0;
        }

        m_D3D12UsageLocal = 0;
        m_D3D12UsageNonLocal = 0;
        m_D3D12BudgetLocal = 0;
        m_D3D12BudgetNonLocal = 0;
        m_OperationsSinceBudgetFetch = 0;
    }

    void AddAllocation(UINT heapTypeIndex, UINT64 allocationSize)
    {
        m_AllocationBytes[heapTypeIndex] += allocationSize;
        ++m_OperationsSinceBudgetFetch;
    }
    void RemoveAllocation(UINT heapTypeIndex, UINT64 allocationSize)
    {
        m_AllocationBytes[heapTypeIndex] -= allocationSize;
        ++m_OperationsSinceBudgetFetch;
    }
    void AddCommittedAllocation(UINT heapTypeIndex, UINT64 allocationSize)
    {
        AddAllocation(heapTypeIndex, allocationSize);
        m_BlockBytes[heapTypeIndex] += allocationSize;
    }
    void RemoveCommittedAllocation(UINT heapTypeIndex, UINT64 allocationSize)
    {
        m_BlockBytes[heapTypeIndex] -= allocationSize;
        RemoveAllocation(heapTypeIndex, allocationSize);
    }
};

class PoolPimpl
{
public:
    PoolPimpl(AllocatorPimpl* allocator, const POOL_DESC& desc);
    HRESULT Init();
    ~PoolPimpl();

    AllocatorPimpl* GetAllocator() const { return m_Allocator; }
    const POOL_DESC& GetDesc() const { return m_Desc; }
    bool SupportsCommittedAllocations() const { return m_Desc.BlockSize == 0; }

    BlockVector* GetBlockVector() { return m_BlockVector; }
    CommittedAllocationList* GetCommittedAllocationList() { return SupportsCommittedAllocations() ? &m_CommittedAllocations : NULL; }

    void CalculateStats(StatInfo& outStats);
    void AddStats(Stats& inoutStats);

    void SetName(LPCWSTR Name);
    LPCWSTR GetName() const { return m_Name; }

private:
    friend class Allocator;
    friend struct PoolListItemTraits;

    AllocatorPimpl* m_Allocator; // Externally owned object.
    POOL_DESC m_Desc;
    BlockVector* m_BlockVector; // Owned object.
    CommittedAllocationList m_CommittedAllocations;
    wchar_t* m_Name;
    PoolPimpl* m_PrevPool = NULL;
    PoolPimpl* m_NextPool = NULL;

    void FreeName();
};

struct PoolListItemTraits
{
    using ItemType = PoolPimpl;
    static ItemType* GetPrev(const ItemType* item) { return item->m_PrevPool; }
    static ItemType* GetNext(const ItemType* item) { return item->m_NextPool; }
    static ItemType*& AccessPrev(ItemType* item) { return item->m_PrevPool; }
    static ItemType*& AccessNext(ItemType* item) { return item->m_NextPool; }
};

struct CommittedAllocationParameters
{
    CommittedAllocationList* m_List = NULL;
    D3D12_HEAP_PROPERTIES m_HeapProperties = {};
    D3D12_HEAP_FLAGS m_HeapFlags = D3D12_HEAP_FLAG_NONE;
    ID3D12ProtectedResourceSession* m_ProtectedSession = NULL;

    bool IsValid() const { return m_List != NULL; }
};

class AllocatorPimpl
{
public:
    std::atomic_uint32_t m_RefCount = 1;
    CurrentBudgetData m_Budget;

    AllocatorPimpl(const ALLOCATION_CALLBACKS& allocationCallbacks, const ALLOCATOR_DESC& desc);
    HRESULT Init(const ALLOCATOR_DESC& desc);
    ~AllocatorPimpl();

    ID3D12Device* GetDevice() const { return m_Device; }
#ifdef __ID3D12Device4_INTERFACE_DEFINED__
    ID3D12Device4* GetDevice4() const { return m_Device4; }
#endif
#ifdef __ID3D12Device8_INTERFACE_DEFINED__
    ID3D12Device8* GetDevice8() const { return m_Device8; }
#endif
    // Shortcut for "Allocation Callbacks", because this function is called so often.
    const ALLOCATION_CALLBACKS& GetAllocs() const { return m_AllocationCallbacks; }
    const D3D12_FEATURE_DATA_D3D12_OPTIONS& GetD3D12Options() const { return m_D3D12Options; }
    BOOL IsUMA() const { return m_D3D12Architecture.UMA; }
    BOOL IsCacheCoherentUMA() const { return m_D3D12Architecture.CacheCoherentUMA; }
    bool SupportsResourceHeapTier2() const { return m_D3D12Options.ResourceHeapTier >= D3D12_RESOURCE_HEAP_TIER_2; }
    bool UseMutex() const { return m_UseMutex; }
    AllocationObjectAllocator& GetAllocationObjectAllocator() { return m_AllocationObjectAllocator; }
    bool HeapFlagsFulfillResourceHeapTier(D3D12_HEAP_FLAGS flags) const;

    HRESULT CreateResource(
        const ALLOCATION_DESC* pAllocDesc,
        const D3D12_RESOURCE_DESC* pResourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState,
        const D3D12_CLEAR_VALUE *pOptimizedClearValue,
        Allocation** ppAllocation,
        REFIID riidResource,
        void** ppvResource);

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
    HRESULT CreateResource2(
        const ALLOCATION_DESC* pAllocDesc,
        const D3D12_RESOURCE_DESC1* pResourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState,
        const D3D12_CLEAR_VALUE *pOptimizedClearValue,
        Allocation** ppAllocation,
        REFIID riidResource,
        void** ppvResource);
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

    HRESULT AllocateMemory(
        const ALLOCATION_DESC* pAllocDesc,
        const D3D12_RESOURCE_ALLOCATION_INFO* pAllocInfo,
        Allocation** ppAllocation);

    HRESULT CreateAliasingResource(
        Allocation* pAllocation,
        UINT64 AllocationLocalOffset,
        const D3D12_RESOURCE_DESC* pResourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState,
        const D3D12_CLEAR_VALUE *pOptimizedClearValue,
        REFIID riidResource,
        void** ppvResource);

    // Unregisters allocation from the collection of dedicated allocations.
    // Allocation object must be deleted externally afterwards.
    void FreeCommittedMemory(Allocation* allocation);
    // Unregisters allocation from the collection of placed allocations.
    // Allocation object must be deleted externally afterwards.
    void FreePlacedMemory(Allocation* allocation);
    // Unregisters allocation from the collection of dedicated allocations and destroys associated heap.
    // Allocation object must be deleted externally afterwards.
    void FreeHeapMemory(Allocation* allocation);

    void SetCurrentFrameIndex(UINT frameIndex);

    UINT GetCurrentFrameIndex() const { return m_CurrentFrameIndex.load(); }

    void CalculateStats(Stats& outStats);

    void GetBudget(Budget* outGpuBudget, Budget* outCpuBudget);
    void GetBudgetForHeapType(Budget& outBudget, D3D12_HEAP_TYPE heapType);

    void BuildStatsString(WCHAR** ppStatsString, BOOL DetailedMap);

    void FreeStatsString(WCHAR* pStatsString);

private:
    friend class Allocator;
    friend class Pool;

    /*
    Heuristics that decides whether a resource should better be placed in its own,
    dedicated allocation (committed resource rather than placed resource).
    */
    template<typename D3D12_RESOURCE_DESC_T>
    static bool PrefersCommittedAllocation(const D3D12_RESOURCE_DESC_T& resourceDesc);

    const bool m_UseMutex;
    const bool m_AlwaysCommitted;
    ID3D12Device* m_Device; // AddRef
#ifdef __ID3D12Device4_INTERFACE_DEFINED__
    ID3D12Device4* m_Device4 = NULL; // AddRef, optional
#endif
#ifdef __ID3D12Device8_INTERFACE_DEFINED__
    ID3D12Device8* m_Device8 = NULL; // AddRef, optional
#endif
    IDXGIAdapter* m_Adapter; // AddRef
#if D3D12MA_DXGI_1_4
    IDXGIAdapter3* m_Adapter3 = NULL; // AddRef, optional
#endif
    UINT64 m_PreferredBlockSize;
    ALLOCATION_CALLBACKS m_AllocationCallbacks;
    D3D12MA_ATOMIC_UINT32 m_CurrentFrameIndex;
    DXGI_ADAPTER_DESC m_AdapterDesc;
    D3D12_FEATURE_DATA_D3D12_OPTIONS m_D3D12Options;
    D3D12_FEATURE_DATA_ARCHITECTURE m_D3D12Architecture;
    AllocationObjectAllocator m_AllocationObjectAllocator;

    using PoolList = IntrusiveLinkedList<PoolListItemTraits>;
    PoolList m_Pools[HEAP_TYPE_COUNT];
    D3D12MA_RW_MUTEX m_PoolsMutex[HEAP_TYPE_COUNT];

    // Default pools.
    BlockVector* m_BlockVectors[DEFAULT_POOL_MAX_COUNT];

    CommittedAllocationList m_CommittedAllocations[STANDARD_HEAP_TYPE_COUNT];

    // Allocates and registers new committed resource with implicit heap, as dedicated allocation.
    // Creates and returns Allocation object and optionally D3D12 resource.
    HRESULT AllocateCommittedResource(
        const CommittedAllocationParameters& committedAllocParams,
        UINT64 resourceSize, bool withinBudget,
        const D3D12_RESOURCE_DESC* pResourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState, const D3D12_CLEAR_VALUE *pOptimizedClearValue,
        Allocation** ppAllocation, REFIID riidResource, void** ppvResource);

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
    HRESULT AllocateCommittedResource2(
        const CommittedAllocationParameters& committedAllocParams,
        UINT64 resourceSize, bool withinBudget,
        const D3D12_RESOURCE_DESC1* pResourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState, const D3D12_CLEAR_VALUE *pOptimizedClearValue,
        Allocation** ppAllocation, REFIID riidResource, void** ppvResource);
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

    // Allocates and registers new heap without any resources placed in it, as dedicated allocation.
    // Creates and returns Allocation object.
    HRESULT AllocateHeap(
        const CommittedAllocationParameters& committedAllocParams,
        const D3D12_RESOURCE_ALLOCATION_INFO& allocInfo, bool withinBudget,
        Allocation** ppAllocation);

    template<typename D3D12_RESOURCE_DESC_T>
    HRESULT CalcAllocationParams(const ALLOCATION_DESC& allocDesc, UINT64 allocSize,
        const D3D12_RESOURCE_DESC_T* resDesc, // Optional
        BlockVector*& outBlockVector, CommittedAllocationParameters& outCommittedAllocationParams, bool& outPreferCommitted);

    /*
    If SupportsResourceHeapTier2():
        0: D3D12_HEAP_TYPE_DEFAULT
        1: D3D12_HEAP_TYPE_UPLOAD
        2: D3D12_HEAP_TYPE_READBACK
    else:
        0: D3D12_HEAP_TYPE_DEFAULT + buffer
        1: D3D12_HEAP_TYPE_DEFAULT + texture
        2: D3D12_HEAP_TYPE_DEFAULT + texture RT or DS
        3: D3D12_HEAP_TYPE_UPLOAD + buffer
        4: D3D12_HEAP_TYPE_UPLOAD + texture
        5: D3D12_HEAP_TYPE_UPLOAD + texture RT or DS
        6: D3D12_HEAP_TYPE_READBACK + buffer
        7: D3D12_HEAP_TYPE_READBACK + texture
        8: D3D12_HEAP_TYPE_READBACK + texture RT or DS
    */
    UINT CalcDefaultPoolCount() const;
    // Returns UINT32_MAX if index cannot be calculcated.
    UINT CalcDefaultPoolIndex(const ALLOCATION_DESC& allocDesc, ResourceClass resourceClass) const;
    void CalcDefaultPoolParams(D3D12_HEAP_TYPE& outHeapType, D3D12_HEAP_FLAGS& outHeapFlags, UINT index) const;

    // Registers Pool object in m_Pools.
    void RegisterPool(Pool* pool, D3D12_HEAP_TYPE heapType);
    // Unregisters Pool object from m_Pools.
    void UnregisterPool(Pool* pool, D3D12_HEAP_TYPE heapType);

    HRESULT UpdateD3D12Budget();
    
    D3D12_RESOURCE_ALLOCATION_INFO GetResourceAllocationInfoNative(const D3D12_RESOURCE_DESC& resourceDesc) const;
#ifdef __ID3D12Device8_INTERFACE_DEFINED__
    D3D12_RESOURCE_ALLOCATION_INFO GetResourceAllocationInfoNative(const D3D12_RESOURCE_DESC1& resourceDesc) const;
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

    template<typename D3D12_RESOURCE_DESC_T>
    D3D12_RESOURCE_ALLOCATION_INFO GetResourceAllocationInfo(D3D12_RESOURCE_DESC_T& inOutResourceDesc) const;

    bool NewAllocationWithinBudget(D3D12_HEAP_TYPE heapType, UINT64 size);

    // Writes object { } with data of given budget.
    static void WriteBudgetToJson(JsonWriter& json, const Budget& budget);
};

////////////////////////////////////////////////////////////////////////////////
// Private class BlockMetadata implementation

BlockMetadata::BlockMetadata(const ALLOCATION_CALLBACKS* allocationCallbacks, bool isVirtual) :
    m_Size(0),
    m_IsVirtual(isVirtual),
    m_pAllocationCallbacks(allocationCallbacks)
{
    D3D12MA_ASSERT(allocationCallbacks);
}

void BlockMetadata::PrintDetailedMap_Begin(JsonWriter& json,
    UINT64 unusedBytes, size_t allocationCount, size_t unusedRangeCount) const
{
    json.BeginObject();

    json.WriteString(L"TotalBytes");
    json.WriteNumber(GetSize());

    json.WriteString(L"UnusedBytes");
    json.WriteNumber(unusedBytes);

    json.WriteString(L"Allocations");
    json.WriteNumber(allocationCount);

    json.WriteString(L"UnusedRanges");
    json.WriteNumber(unusedRangeCount);

    json.WriteString(L"Suballocations");
    json.BeginArray();
}

void BlockMetadata::PrintDetailedMap_Allocation(JsonWriter& json,
    UINT64 offset, UINT64 size, void* userData) const
{
    json.BeginObject(true);

    json.WriteString(L"Offset");
    json.WriteNumber(offset);
    
    if (IsVirtual())
    {
        json.WriteString(L"Type");
        json.WriteString(L"ALLOCATION");
        json.WriteString(L"Size");
        json.WriteNumber(size);
        if (userData)
        {
            json.WriteString(L"UserData");
            json.WriteNumber((uintptr_t)userData);
        }
    }
    else
    {
        const Allocation* const alloc = (const Allocation*)userData;
        D3D12MA_ASSERT(alloc);
        json.AddAllocationToObject(*alloc);
    }
    json.EndObject();
}

void BlockMetadata::PrintDetailedMap_UnusedRange(JsonWriter& json,
    UINT64 offset, UINT64 size) const
{
    json.BeginObject(true);

    json.WriteString(L"Offset");
    json.WriteNumber(offset);

    json.WriteString(L"Type");
    json.WriteString(L"FREE");

    json.WriteString(L"Size");
    json.WriteNumber(size);

    json.EndObject();
}

void BlockMetadata::PrintDetailedMap_End(JsonWriter& json) const
{
    json.EndArray();
    json.EndObject();
}

////////////////////////////////////////////////////////////////////////////////
// Private class BlockMetadata_Generic implementation

BlockMetadata_Generic::BlockMetadata_Generic(const ALLOCATION_CALLBACKS* allocationCallbacks, bool isVirtual) :
    BlockMetadata(allocationCallbacks, isVirtual),
    m_FreeCount(0),
    m_SumFreeSize(0),
    m_Suballocations(*allocationCallbacks),
    m_FreeSuballocationsBySize(*allocationCallbacks)
{
    D3D12MA_ASSERT(allocationCallbacks);
}

BlockMetadata_Generic::~BlockMetadata_Generic()
{
}

void BlockMetadata_Generic::Init(UINT64 size)
{
    BlockMetadata::Init(size);
    m_ZeroInitializedRange.Reset(size);

    m_FreeCount = 1;
    m_SumFreeSize = size;

    Suballocation suballoc = {};
    suballoc.offset = 0;
    suballoc.size = size;
    suballoc.type = SUBALLOCATION_TYPE_FREE;
    suballoc.userData = NULL;

    D3D12MA_ASSERT(size > MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER);
    m_Suballocations.push_back(suballoc);
    SuballocationList::iterator suballocItem = m_Suballocations.end();
    --suballocItem;
    m_FreeSuballocationsBySize.push_back(suballocItem);
}

bool BlockMetadata_Generic::Validate() const
{
    D3D12MA_VALIDATE(!m_Suballocations.empty());

    // Expected offset of new suballocation as calculated from previous ones.
    UINT64 calculatedOffset = 0;
    // Expected number of free suballocations as calculated from traversing their list.
    UINT calculatedFreeCount = 0;
    // Expected sum size of free suballocations as calculated from traversing their list.
    UINT64 calculatedSumFreeSize = 0;
    // Expected number of free suballocations that should be registered in
    // m_FreeSuballocationsBySize calculated from traversing their list.
    size_t freeSuballocationsToRegister = 0;
    // True if previous visited suballocation was free.
    bool prevFree = false;

    for(const auto& subAlloc : m_Suballocations)
    {
        // Actual offset of this suballocation doesn't match expected one.
        D3D12MA_VALIDATE(subAlloc.offset == calculatedOffset);

        const bool currFree = (subAlloc.type == SUBALLOCATION_TYPE_FREE);
        // Two adjacent free suballocations are invalid. They should be merged.
        D3D12MA_VALIDATE(!prevFree || !currFree);

        const Allocation* const alloc = (Allocation*)subAlloc.userData;
        if(!IsVirtual())
        {
            D3D12MA_VALIDATE(currFree == (alloc == NULL));
        }

        if(currFree)
        {
            calculatedSumFreeSize += subAlloc.size;
            ++calculatedFreeCount;
            if(subAlloc.size >= MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER)
            {
                ++freeSuballocationsToRegister;
            }

            // Margin required between allocations - every free space must be at least that large.
            D3D12MA_VALIDATE(subAlloc.size >= GetDebugMargin());
        }
        else
        {
            if(!IsVirtual())
            {
                D3D12MA_VALIDATE(alloc->GetOffset() == subAlloc.offset);
                D3D12MA_VALIDATE(alloc->GetSize() == subAlloc.size);
            }

            // Margin required between allocations - previous allocation must be free.
            D3D12MA_VALIDATE(GetDebugMargin() == 0 || prevFree);
        }

        calculatedOffset += subAlloc.size;
        prevFree = currFree;
    }

    // Number of free suballocations registered in m_FreeSuballocationsBySize doesn't
    // match expected one.
    D3D12MA_VALIDATE(m_FreeSuballocationsBySize.size() == freeSuballocationsToRegister);

    UINT64 lastSize = 0;
    for(size_t i = 0; i < m_FreeSuballocationsBySize.size(); ++i)
    {
        SuballocationList::iterator suballocItem = m_FreeSuballocationsBySize[i];

        // Only free suballocations can be registered in m_FreeSuballocationsBySize.
        D3D12MA_VALIDATE(suballocItem->type == SUBALLOCATION_TYPE_FREE);
        // They must be sorted by size ascending.
        D3D12MA_VALIDATE(suballocItem->size >= lastSize);

        lastSize = suballocItem->size;
    }

    // Check if totals match calculacted values.
    D3D12MA_VALIDATE(ValidateFreeSuballocationList());
    D3D12MA_VALIDATE(calculatedOffset == GetSize());
    D3D12MA_VALIDATE(calculatedSumFreeSize == m_SumFreeSize);
    D3D12MA_VALIDATE(calculatedFreeCount == m_FreeCount);

    return true;
}

bool BlockMetadata_Generic::IsEmpty() const
{
    return (m_Suballocations.size() == 1) && (m_FreeCount == 1);
}

void BlockMetadata_Generic::GetAllocationInfo(AllocHandle allocHandle, VIRTUAL_ALLOCATION_INFO& outInfo) const
{
    Suballocation& suballoc = *FindAtOffset((UINT64)allocHandle - 1).dropConst();
    outInfo.Offset = suballoc.offset;
    outInfo.Size = suballoc.size;
    outInfo.pUserData = suballoc.userData;
}

bool BlockMetadata_Generic::CreateAllocationRequest(
    UINT64 allocSize,
    UINT64 allocAlignment,
    bool upperAddress,
    AllocationRequest* pAllocationRequest)
{
    D3D12MA_ASSERT(allocSize > 0);
    D3D12MA_ASSERT(!upperAddress && "ALLOCATION_FLAG_UPPER_ADDRESS can be used only with linear algorithm.");
    D3D12MA_ASSERT(pAllocationRequest != NULL);
    D3D12MA_HEAVY_ASSERT(Validate());

    // There is not enough total free space in this block to fullfill the request: Early return.
    if(m_SumFreeSize < allocSize + GetDebugMargin())
    {
        return false;
    }

    // New algorithm, efficiently searching freeSuballocationsBySize.
    const size_t freeSuballocCount = m_FreeSuballocationsBySize.size();
    if(freeSuballocCount > 0)
    {
        // Find first free suballocation with size not less than allocSize + GetDebugMargin().
        SuballocationList::iterator* const it = BinaryFindFirstNotLess(
            m_FreeSuballocationsBySize.data(),
            m_FreeSuballocationsBySize.data() + freeSuballocCount,
            allocSize + GetDebugMargin(),
            SuballocationItemSizeLess());
        size_t index = it - m_FreeSuballocationsBySize.data();
        for(; index < freeSuballocCount; ++index)
        {
            if(CheckAllocation(
                allocSize,
                allocAlignment,
                m_FreeSuballocationsBySize[index],
                &pAllocationRequest->allocHandle,
                &pAllocationRequest->sumFreeSize,
                &pAllocationRequest->sumItemSize,
                &pAllocationRequest->zeroInitialized))
            {
                pAllocationRequest->item = m_FreeSuballocationsBySize[index];
                return true;
            }
        }
    }

    return false;
}

void BlockMetadata_Generic::Alloc(
    const AllocationRequest& request,
    UINT64 allocSize,
    void* userData)
{
    D3D12MA_ASSERT(request.item != m_Suballocations.end());
    Suballocation& suballoc = *request.item;
    // Given suballocation is a free block.
    D3D12MA_ASSERT(suballoc.type == SUBALLOCATION_TYPE_FREE);
    // Given offset is inside this suballocation.
    UINT64 offset = (UINT64)request.allocHandle - 1;
    D3D12MA_ASSERT(offset >= suballoc.offset);
    const UINT64 paddingBegin = offset - suballoc.offset;
    D3D12MA_ASSERT(suballoc.size >= paddingBegin + allocSize);
    const UINT64 paddingEnd = suballoc.size - paddingBegin - allocSize;

    // Unregister this free suballocation from m_FreeSuballocationsBySize and update
    // it to become used.
    UnregisterFreeSuballocation(request.item);

    suballoc.offset = offset;
    suballoc.size = allocSize;
    suballoc.type = SUBALLOCATION_TYPE_ALLOCATION;
    suballoc.userData = userData;

    // If there are any free bytes remaining at the end, insert new free suballocation after current one.
    if(paddingEnd)
    {
        Suballocation paddingSuballoc = {};
        paddingSuballoc.offset = offset + allocSize;
        paddingSuballoc.size = paddingEnd;
        paddingSuballoc.type = SUBALLOCATION_TYPE_FREE;
        SuballocationList::iterator next = request.item;
        ++next;
        const SuballocationList::iterator paddingEndItem =
            m_Suballocations.insert(next, paddingSuballoc);
        RegisterFreeSuballocation(paddingEndItem);
    }

    // If there are any free bytes remaining at the beginning, insert new free suballocation before current one.
    if(paddingBegin)
    {
        Suballocation paddingSuballoc = {};
        paddingSuballoc.offset = offset - paddingBegin;
        paddingSuballoc.size = paddingBegin;
        paddingSuballoc.type = SUBALLOCATION_TYPE_FREE;
        const SuballocationList::iterator paddingBeginItem =
            m_Suballocations.insert(request.item, paddingSuballoc);
        RegisterFreeSuballocation(paddingBeginItem);
    }

    // Update totals.
    m_FreeCount = m_FreeCount - 1;
    if(paddingBegin > 0)
    {
        ++m_FreeCount;
    }
    if(paddingEnd > 0)
    {
        ++m_FreeCount;
    }
    m_SumFreeSize -= allocSize;

    m_ZeroInitializedRange.MarkRangeAsUsed(offset, offset + allocSize);
}

void BlockMetadata_Generic::Free(AllocHandle allocHandle)
{
    FreeSuballocation(FindAtOffset((UINT64)allocHandle - 1).dropConst());
}

void BlockMetadata_Generic::Clear()
{
    m_FreeCount = 1;
    m_SumFreeSize = GetSize();

    m_Suballocations.clear();
    Suballocation suballoc = {};
    suballoc.offset = 0;
    suballoc.size = GetSize();
    suballoc.type = SUBALLOCATION_TYPE_FREE;
    m_Suballocations.push_back(suballoc);

    m_FreeSuballocationsBySize.clear();
    m_FreeSuballocationsBySize.push_back(m_Suballocations.begin());
}

SuballocationList::const_iterator BlockMetadata_Generic::FindAtOffset(UINT64 offset) const
{
    const UINT64 last = m_Suballocations.crbegin()->offset;
    if (last == offset)
        return m_Suballocations.crbegin();
    const UINT64 first = m_Suballocations.cbegin()->offset;
    if (first == offset)
        return m_Suballocations.cbegin();

    const size_t suballocCount = m_Suballocations.size();
    const UINT64 step = (last - first + m_Suballocations.cbegin()->size) / suballocCount;
    auto findSuballocation = [&](auto begin, auto end) -> SuballocationList::const_iterator
    {
        for (auto suballocItem = begin;
            suballocItem != end;
            ++suballocItem)
        {
            const Suballocation& suballoc = *suballocItem;
            if (suballoc.offset == offset)
                return suballocItem;
        }
        D3D12MA_ASSERT(false && "Not found!");
        return m_Suballocations.end();
    };
    // If requested offset is closer to the end of range, search from the end
    if ((offset - first) > suballocCount * step / 2)
    {
        return findSuballocation(m_Suballocations.crbegin(), m_Suballocations.crend());
    }
    return findSuballocation(m_Suballocations.cbegin(), m_Suballocations.cend());
}

bool BlockMetadata_Generic::ValidateFreeSuballocationList() const
{
    UINT64 lastSize = 0;
    for(size_t i = 0, count = m_FreeSuballocationsBySize.size(); i < count; ++i)
    {
        const SuballocationList::iterator it = m_FreeSuballocationsBySize[i];

        D3D12MA_VALIDATE(it->type == SUBALLOCATION_TYPE_FREE);
        D3D12MA_VALIDATE(it->size >= MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER);
        D3D12MA_VALIDATE(it->size >= lastSize);
        lastSize = it->size;
    }
    return true;
}

bool BlockMetadata_Generic::CheckAllocation(
    UINT64 allocSize,
    UINT64 allocAlignment,
    SuballocationList::const_iterator suballocItem,
    AllocHandle* pAllocHandle,
    UINT64* pSumFreeSize,
    UINT64* pSumItemSize,
    BOOL *pZeroInitialized) const
{
    D3D12MA_ASSERT(allocSize > 0);
    D3D12MA_ASSERT(suballocItem != m_Suballocations.cend());
    D3D12MA_ASSERT(pAllocHandle != NULL && pZeroInitialized != NULL);

    *pSumFreeSize = 0;
    *pSumItemSize = 0;
    *pZeroInitialized = FALSE;

    const Suballocation& suballoc = *suballocItem;
    D3D12MA_ASSERT(suballoc.type == SUBALLOCATION_TYPE_FREE);

    *pSumFreeSize = suballoc.size;

    // Size of this suballocation is too small for this request: Early return.
    if(suballoc.size < allocSize)
    {
        return false;
    }

    // Start from offset equal to beginning of this suballocation and debug margin of previous allocation if present.
    UINT64 offset = suballoc.offset + (suballocItem == m_Suballocations.cbegin() ? 0 : GetDebugMargin());

    // Apply alignment.
    offset = AlignUp(offset, allocAlignment);

    // Calculate padding at the beginning based on current offset.
    const UINT64 paddingBegin = offset - suballoc.offset;

    // Fail if requested size plus margin after is bigger than size of this suballocation.
    if(paddingBegin + allocSize + GetDebugMargin() > suballoc.size)
    {
        return false;
    }

    // All tests passed: Success. Offset is already filled.
    *pZeroInitialized = m_ZeroInitializedRange.IsRangeZeroInitialized(offset, offset + allocSize);
    *pAllocHandle = (AllocHandle)(offset + 1);
    return true;
}

void BlockMetadata_Generic::MergeFreeWithNext(SuballocationList::iterator item)
{
    D3D12MA_ASSERT(item != m_Suballocations.end());
    D3D12MA_ASSERT(item->type == SUBALLOCATION_TYPE_FREE);

    SuballocationList::iterator nextItem = item;
    ++nextItem;
    D3D12MA_ASSERT(nextItem != m_Suballocations.end());
    D3D12MA_ASSERT(nextItem->type == SUBALLOCATION_TYPE_FREE);

    item->size += nextItem->size;
    --m_FreeCount;
    m_Suballocations.erase(nextItem);
}

SuballocationList::iterator BlockMetadata_Generic::FreeSuballocation(SuballocationList::iterator suballocItem)
{
    // Change this suballocation to be marked as free.
    Suballocation& suballoc = *suballocItem;
    suballoc.type = SUBALLOCATION_TYPE_FREE;
    suballoc.userData = NULL;

    // Update totals.
    ++m_FreeCount;
    m_SumFreeSize += suballoc.size;

    // Merge with previous and/or next suballocation if it's also free.
    bool mergeWithNext = false;
    bool mergeWithPrev = false;

    SuballocationList::iterator nextItem = suballocItem;
    ++nextItem;
    if((nextItem != m_Suballocations.end()) && (nextItem->type == SUBALLOCATION_TYPE_FREE))
    {
        mergeWithNext = true;
    }

    SuballocationList::iterator prevItem = suballocItem;
    if(suballocItem != m_Suballocations.begin())
    {
        --prevItem;
        if(prevItem->type == SUBALLOCATION_TYPE_FREE)
        {
            mergeWithPrev = true;
        }
    }

    if(mergeWithNext)
    {
        UnregisterFreeSuballocation(nextItem);
        MergeFreeWithNext(suballocItem);
    }

    if(mergeWithPrev)
    {
        UnregisterFreeSuballocation(prevItem);
        MergeFreeWithNext(prevItem);
        RegisterFreeSuballocation(prevItem);
        return prevItem;
    }
    else
    {
        RegisterFreeSuballocation(suballocItem);
        return suballocItem;
    }
}

void BlockMetadata_Generic::RegisterFreeSuballocation(SuballocationList::iterator item)
{
    D3D12MA_ASSERT(item->type == SUBALLOCATION_TYPE_FREE);
    D3D12MA_ASSERT(item->size > 0);

    // You may want to enable this validation at the beginning or at the end of
    // this function, depending on what do you want to check.
    D3D12MA_HEAVY_ASSERT(ValidateFreeSuballocationList());

    if(item->size >= MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER)
    {
        if(m_FreeSuballocationsBySize.empty())
        {
            m_FreeSuballocationsBySize.push_back(item);
        }
        else
        {
            m_FreeSuballocationsBySize.InsertSorted(item, SuballocationItemSizeLess());
        }
    }

    //D3D12MA_HEAVY_ASSERT(ValidateFreeSuballocationList());
}

void BlockMetadata_Generic::UnregisterFreeSuballocation(SuballocationList::iterator item)
{
    D3D12MA_ASSERT(item->type == SUBALLOCATION_TYPE_FREE);
    D3D12MA_ASSERT(item->size > 0);

    // You may want to enable this validation at the beginning or at the end of
    // this function, depending on what do you want to check.
    D3D12MA_HEAVY_ASSERT(ValidateFreeSuballocationList());

    if(item->size >= MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER)
    {
        SuballocationList::iterator* const it = BinaryFindFirstNotLess(
            m_FreeSuballocationsBySize.data(),
            m_FreeSuballocationsBySize.data() + m_FreeSuballocationsBySize.size(),
            item,
            SuballocationItemSizeLess());
        for(size_t index = it - m_FreeSuballocationsBySize.data();
            index < m_FreeSuballocationsBySize.size();
            ++index)
        {
            if(m_FreeSuballocationsBySize[index] == item)
            {
                m_FreeSuballocationsBySize.remove(index);
                return;
            }
            D3D12MA_ASSERT((m_FreeSuballocationsBySize[index]->size == item->size) && "Not found.");
        }
        D3D12MA_ASSERT(0 && "Not found.");
    }

    //D3D12MA_HEAVY_ASSERT(ValidateFreeSuballocationList());
}

void BlockMetadata_Generic::SetAllocationUserData(AllocHandle allocHandle, void* userData)
{
    Suballocation& suballoc = *FindAtOffset((UINT64)allocHandle - 1).dropConst();
    suballoc.userData = userData;
}

void BlockMetadata_Generic::CalcAllocationStatInfo(StatInfo& outInfo) const
{
    outInfo.BlockCount = 1;

    const UINT rangeCount = (UINT)m_Suballocations.size();
    outInfo.AllocationCount = rangeCount - m_FreeCount;
    outInfo.UnusedRangeCount = m_FreeCount;

    outInfo.UsedBytes = GetSize() - m_SumFreeSize;
    outInfo.UnusedBytes = m_SumFreeSize;

    outInfo.AllocationSizeMin = UINT64_MAX;
    outInfo.AllocationSizeMax = 0;
    outInfo.UnusedRangeSizeMin = UINT64_MAX;
    outInfo.UnusedRangeSizeMax = 0;

    for(const auto& suballoc : m_Suballocations)
    {
        if(suballoc.type == SUBALLOCATION_TYPE_FREE)
        {
            outInfo.UnusedRangeSizeMin = D3D12MA_MIN(suballoc.size, outInfo.UnusedRangeSizeMin);
            outInfo.UnusedRangeSizeMax = D3D12MA_MAX(suballoc.size, outInfo.UnusedRangeSizeMax);
        }
        else
        {
            outInfo.AllocationSizeMin = D3D12MA_MIN(suballoc.size, outInfo.AllocationSizeMin);
            outInfo.AllocationSizeMax = D3D12MA_MAX(suballoc.size, outInfo.AllocationSizeMax);
        }
    }
}

void BlockMetadata_Generic::WriteAllocationInfoToJson(JsonWriter& json) const
{
    PrintDetailedMap_Begin(json, GetSumFreeSize(), GetAllocationCount(), m_FreeCount);
    for(const auto& suballoc : m_Suballocations)
    {
        if (suballoc.type == SUBALLOCATION_TYPE_FREE)
            PrintDetailedMap_UnusedRange(json, suballoc.offset, suballoc.size);
        else
            PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);
    }
    PrintDetailedMap_End(json);
}

////////////////////////////////////////////////////////////////////////////////
// Private class BlockMetadata_Linear implementation

BlockMetadata_Linear::BlockMetadata_Linear(const ALLOCATION_CALLBACKS* allocationCallbacks, bool isVirtual)
    : BlockMetadata(allocationCallbacks, isVirtual),
    m_SumFreeSize(0),
    m_Suballocations0(*allocationCallbacks),
    m_Suballocations1(*allocationCallbacks),
    m_1stVectorIndex(0),
    m_2ndVectorMode(SECOND_VECTOR_EMPTY),
    m_1stNullItemsBeginCount(0),
    m_1stNullItemsMiddleCount(0),
    m_2ndNullItemsCount(0)
{
    D3D12MA_ASSERT(allocationCallbacks);
}

void BlockMetadata_Linear::Init(UINT64 size)
{
    BlockMetadata::Init(size);
    m_SumFreeSize = size;
}

bool BlockMetadata_Linear::Validate() const
{
    D3D12MA_VALIDATE(GetSumFreeSize() <= GetSize());
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    D3D12MA_VALIDATE(suballocations2nd.empty() == (m_2ndVectorMode == SECOND_VECTOR_EMPTY));
    D3D12MA_VALIDATE(!suballocations1st.empty() ||
        suballocations2nd.empty() ||
        m_2ndVectorMode != SECOND_VECTOR_RING_BUFFER);

    if (!suballocations1st.empty())
    {
        // Null item at the beginning should be accounted into m_1stNullItemsBeginCount.
        D3D12MA_VALIDATE(suballocations1st[m_1stNullItemsBeginCount].type != SUBALLOCATION_TYPE_FREE);
        // Null item at the end should be just pop_back().
        D3D12MA_VALIDATE(suballocations1st.back().type != SUBALLOCATION_TYPE_FREE);
    }
    if (!suballocations2nd.empty())
    {
        // Null item at the end should be just pop_back().
        D3D12MA_VALIDATE(suballocations2nd.back().type != SUBALLOCATION_TYPE_FREE);
    }

    D3D12MA_VALIDATE(m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount <= suballocations1st.size());
    D3D12MA_VALIDATE(m_2ndNullItemsCount <= suballocations2nd.size());

    UINT64 sumUsedSize = 0;
    const size_t suballoc1stCount = suballocations1st.size();
    UINT64 offset = 0;

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const size_t suballoc2ndCount = suballocations2nd.size();
        size_t nullItem2ndCount = 0;
        for (size_t i = 0; i < suballoc2ndCount; ++i)
        {
            const Suballocation& suballoc = suballocations2nd[i];
            const bool currFree = (suballoc.type == SUBALLOCATION_TYPE_FREE);

            const Allocation* alloc = (Allocation*)suballoc.userData;
            if (!IsVirtual())
            {
                D3D12MA_VALIDATE(currFree == (alloc == NULL));
            }
            D3D12MA_VALIDATE(suballoc.offset >= offset);

            if (!currFree)
            {
                if (!IsVirtual())
                {
                    D3D12MA_VALIDATE((UINT64)alloc->GetAllocHandle() == suballoc.offset);
                    D3D12MA_VALIDATE(alloc->GetSize() == suballoc.size);
                }
                sumUsedSize += suballoc.size;
            }
            else
            {
                ++nullItem2ndCount;
            }

            offset = suballoc.offset + suballoc.size + GetDebugMargin();
        }

        D3D12MA_VALIDATE(nullItem2ndCount == m_2ndNullItemsCount);
    }

    for (size_t i = 0; i < m_1stNullItemsBeginCount; ++i)
    {
        const Suballocation& suballoc = suballocations1st[i];
        D3D12MA_VALIDATE(suballoc.type == SUBALLOCATION_TYPE_FREE &&
            suballoc.userData == NULL);
    }

    size_t nullItem1stCount = m_1stNullItemsBeginCount;

    for (size_t i = m_1stNullItemsBeginCount; i < suballoc1stCount; ++i)
    {
        const Suballocation& suballoc = suballocations1st[i];
        const bool currFree = (suballoc.type == SUBALLOCATION_TYPE_FREE);

        const Allocation* alloc = (Allocation*)suballoc.userData;
        if (!IsVirtual())
        {
            D3D12MA_VALIDATE(currFree == (alloc == NULL));
        }
        D3D12MA_VALIDATE(suballoc.offset >= offset);
        D3D12MA_VALIDATE(i >= m_1stNullItemsBeginCount || currFree);

        if (!currFree)
        {
            if (!IsVirtual())
            {
                D3D12MA_VALIDATE((UINT64)alloc->GetAllocHandle() == suballoc.offset);
                D3D12MA_VALIDATE(alloc->GetSize() == suballoc.size);
            }
            sumUsedSize += suballoc.size;
        }
        else
        {
            ++nullItem1stCount;
        }

        offset = suballoc.offset + suballoc.size + GetDebugMargin();
    }
    D3D12MA_VALIDATE(nullItem1stCount == m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount);

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        const size_t suballoc2ndCount = suballocations2nd.size();
        size_t nullItem2ndCount = 0;
        for (size_t i = suballoc2ndCount; i--; )
        {
            const Suballocation& suballoc = suballocations2nd[i];
            const bool currFree = (suballoc.type == SUBALLOCATION_TYPE_FREE);

            const Allocation* alloc = (Allocation*)suballoc.userData;
            if (!IsVirtual())
            {
                D3D12MA_VALIDATE(currFree == (alloc == NULL));
            }
            D3D12MA_VALIDATE(suballoc.offset >= offset);

            if (!currFree)
            {
                if (!IsVirtual())
                {
                    D3D12MA_VALIDATE((UINT64)alloc->GetAllocHandle() == suballoc.offset);
                    D3D12MA_VALIDATE(alloc->GetSize() == suballoc.size);
                }
                sumUsedSize += suballoc.size;
            }
            else
            {
                ++nullItem2ndCount;
            }

            offset = suballoc.offset + suballoc.size + GetDebugMargin();
        }

        D3D12MA_VALIDATE(nullItem2ndCount == m_2ndNullItemsCount);
    }

    D3D12MA_VALIDATE(offset <= GetSize());
    D3D12MA_VALIDATE(m_SumFreeSize == GetSize() - sumUsedSize);

    return true;
}

size_t BlockMetadata_Linear::GetAllocationCount() const
{
    return AccessSuballocations1st().size() - m_1stNullItemsBeginCount - m_1stNullItemsMiddleCount +
        AccessSuballocations2nd().size() - m_2ndNullItemsCount;
}

void BlockMetadata_Linear::GetAllocationInfo(AllocHandle allocHandle, VIRTUAL_ALLOCATION_INFO& outInfo) const
{
    const Suballocation& suballoc = FindSuballocation((UINT64)allocHandle - 1);
    outInfo.Offset = suballoc.offset;
    outInfo.Size = suballoc.size;
    outInfo.pUserData = suballoc.userData;
}

bool BlockMetadata_Linear::CreateAllocationRequest(
    UINT64 allocSize,
    UINT64 allocAlignment,
    bool upperAddress,
    AllocationRequest* pAllocationRequest)
{
    D3D12MA_ASSERT(allocSize > 0 && "Cannot allocate empty block!");
    D3D12MA_ASSERT(pAllocationRequest != NULL);
    D3D12MA_HEAVY_ASSERT(Validate());
    pAllocationRequest->size = allocSize;
    return upperAddress ?
        CreateAllocationRequest_UpperAddress(
            allocSize, allocAlignment, pAllocationRequest) :
        CreateAllocationRequest_LowerAddress(
            allocSize, allocAlignment, pAllocationRequest);
}

void BlockMetadata_Linear::Alloc(
    const AllocationRequest& request,
    UINT64 allocSize,
    void* userData)
{
    UINT64 offset = (UINT64)request.allocHandle - 1;
    const Suballocation newSuballoc = { offset, request.size, userData, SUBALLOCATION_TYPE_ALLOCATION };

    switch (request.algorithmData)
    {
    case ALLOC_REQUEST_UPPER_ADDRESS:
    {
        D3D12MA_ASSERT(m_2ndVectorMode != SECOND_VECTOR_RING_BUFFER &&
            "CRITICAL ERROR: Trying to use linear allocator as double stack while it was already used as ring buffer.");
        SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
        suballocations2nd.push_back(newSuballoc);
        m_2ndVectorMode = SECOND_VECTOR_DOUBLE_STACK;
        break;
    }
    case ALLOC_REQUEST_END_OF_1ST:
    {
        SuballocationVectorType& suballocations1st = AccessSuballocations1st();

        D3D12MA_ASSERT(suballocations1st.empty() ||
            offset >= suballocations1st.back().offset + suballocations1st.back().size);
        // Check if it fits before the end of the block.
        D3D12MA_ASSERT(offset + request.size <= GetSize());

        suballocations1st.push_back(newSuballoc);
        break;
    }
    case ALLOC_REQUEST_END_OF_2ND:
    {
        SuballocationVectorType& suballocations1st = AccessSuballocations1st();
        // New allocation at the end of 2-part ring buffer, so before first allocation from 1st vector.
        D3D12MA_ASSERT(!suballocations1st.empty() &&
            offset + request.size <= suballocations1st[m_1stNullItemsBeginCount].offset);
        SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

        switch (m_2ndVectorMode)
        {
        case SECOND_VECTOR_EMPTY:
            // First allocation from second part ring buffer.
            D3D12MA_ASSERT(suballocations2nd.empty());
            m_2ndVectorMode = SECOND_VECTOR_RING_BUFFER;
            break;
        case SECOND_VECTOR_RING_BUFFER:
            // 2-part ring buffer is already started.
            D3D12MA_ASSERT(!suballocations2nd.empty());
            break;
        case SECOND_VECTOR_DOUBLE_STACK:
            D3D12MA_ASSERT(0 && "CRITICAL ERROR: Trying to use linear allocator as ring buffer while it was already used as double stack.");
            break;
        default:
            D3D12MA_ASSERT(0);
        }

        suballocations2nd.push_back(newSuballoc);
        break;
    }
    default:
        D3D12MA_ASSERT(0 && "CRITICAL INTERNAL ERROR.");
    }
    m_SumFreeSize -= newSuballoc.size;
}

void BlockMetadata_Linear::Free(AllocHandle allocHandle)
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    UINT64 offset = (UINT64)allocHandle - 1;

    if (!suballocations1st.empty())
    {
        // First allocation: Mark it as next empty at the beginning.
        Suballocation& firstSuballoc = suballocations1st[m_1stNullItemsBeginCount];
        if (firstSuballoc.offset == offset)
        {
            firstSuballoc.type = SUBALLOCATION_TYPE_FREE;
            firstSuballoc.userData = NULL;
            m_SumFreeSize += firstSuballoc.size;
            ++m_1stNullItemsBeginCount;
            CleanupAfterFree();
            return;
        }
    }

    // Last allocation in 2-part ring buffer or top of upper stack (same logic).
    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ||
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        Suballocation& lastSuballoc = suballocations2nd.back();
        if (lastSuballoc.offset == offset)
        {
            m_SumFreeSize += lastSuballoc.size;
            suballocations2nd.pop_back();
            CleanupAfterFree();
            return;
        }
    }
    // Last allocation in 1st vector.
    else if (m_2ndVectorMode == SECOND_VECTOR_EMPTY)
    {
        Suballocation& lastSuballoc = suballocations1st.back();
        if (lastSuballoc.offset == offset)
        {
            m_SumFreeSize += lastSuballoc.size;
            suballocations1st.pop_back();
            CleanupAfterFree();
            return;
        }
    }

    Suballocation refSuballoc;
    refSuballoc.offset = offset;
    // Rest of members stays uninitialized intentionally for better performance.

    // Item from the middle of 1st vector.
    {
        const SuballocationVectorType::iterator it = BinaryFindSorted(
            suballocations1st.begin() + m_1stNullItemsBeginCount,
            suballocations1st.end(),
            refSuballoc,
            SuballocationOffsetLess());
        if (it != suballocations1st.end())
        {
            it->type = SUBALLOCATION_TYPE_FREE;
            it->userData = NULL;
            ++m_1stNullItemsMiddleCount;
            m_SumFreeSize += it->size;
            CleanupAfterFree();
            return;
        }
    }

    if (m_2ndVectorMode != SECOND_VECTOR_EMPTY)
    {
        // Item from the middle of 2nd vector.
        const SuballocationVectorType::iterator it = m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ?
            BinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, SuballocationOffsetLess()) :
            BinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, SuballocationOffsetGreater());
        if (it != suballocations2nd.end())
        {
            it->type = SUBALLOCATION_TYPE_FREE;
            it->userData = NULL;
            ++m_2ndNullItemsCount;
            m_SumFreeSize += it->size;
            CleanupAfterFree();
            return;
        }
    }

    D3D12MA_ASSERT(0 && "Allocation to free not found in linear allocator!");
}

void BlockMetadata_Linear::Clear()
{
    m_SumFreeSize = GetSize();
    m_Suballocations0.clear();
    m_Suballocations1.clear();
    // Leaving m_1stVectorIndex unchanged - it doesn't matter.
    m_2ndVectorMode = SECOND_VECTOR_EMPTY;
    m_1stNullItemsBeginCount = 0;
    m_1stNullItemsMiddleCount = 0;
    m_2ndNullItemsCount = 0;
}

void BlockMetadata_Linear::SetAllocationUserData(AllocHandle allocHandle, void* userData)
{
    Suballocation& suballoc = FindSuballocation((UINT64)allocHandle - 1);
    suballoc.userData = userData;
}

void BlockMetadata_Linear::CalcAllocationStatInfo(StatInfo& outInfo) const
{
    outInfo.BlockCount = 1;
    outInfo.AllocationCount = 0;
    outInfo.UnusedRangeCount = 0;
    outInfo.UsedBytes = 0;

    outInfo.AllocationSizeMin = UINT64_MAX;
    outInfo.AllocationSizeMax = 0;
    outInfo.UnusedRangeSizeMin = UINT64_MAX;
    outInfo.UnusedRangeSizeMax = 0;

    const UINT64 size = GetSize();
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    UINT64 lastOffset = 0;
    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const UINT64 freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAllocIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const Suballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const UINT64 unusedRangeSize = suballoc.offset - lastOffset;
                    ++outInfo.UnusedRangeCount;
                    outInfo.UnusedRangeSizeMin = D3D12MA_MIN(unusedRangeSize, outInfo.UnusedRangeSizeMin);
                    outInfo.UnusedRangeSizeMax = D3D12MA_MAX(unusedRangeSize, outInfo.UnusedRangeSizeMax);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                outInfo.UsedBytes += suballoc.size;
                ++outInfo.AllocationCount;
                outInfo.AllocationSizeMin = D3D12MA_MIN(suballoc.size, outInfo.AllocationSizeMin);
                outInfo.AllocationSizeMax = D3D12MA_MAX(suballoc.size, outInfo.AllocationSizeMax);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    const UINT64 unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    ++outInfo.UnusedRangeCount;
                    outInfo.UnusedRangeSizeMin = D3D12MA_MIN(unusedRangeSize, outInfo.UnusedRangeSizeMin);
                    outInfo.UnusedRangeSizeMax = D3D12MA_MAX(unusedRangeSize, outInfo.UnusedRangeSizeMax);
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    const UINT64 freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const Suballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const UINT64 unusedRangeSize = suballoc.offset - lastOffset;
                ++outInfo.UnusedRangeCount;
                outInfo.UnusedRangeSizeMin = D3D12MA_MIN(unusedRangeSize, outInfo.UnusedRangeSizeMin);
                outInfo.UnusedRangeSizeMax = D3D12MA_MAX(unusedRangeSize, outInfo.UnusedRangeSizeMax);
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            outInfo.UsedBytes += suballoc.size;
            ++outInfo.AllocationCount;
            outInfo.AllocationSizeMin = D3D12MA_MIN(suballoc.size, outInfo.AllocationSizeMin);
            outInfo.AllocationSizeMax = D3D12MA_MAX(suballoc.size, outInfo.AllocationSizeMax);

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            // There is free space from lastOffset to freeSpace1stTo2ndEnd.
            if (lastOffset < freeSpace1stTo2ndEnd)
            {
                const UINT64 unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                ++outInfo.UnusedRangeCount;
                outInfo.UnusedRangeSizeMin = D3D12MA_MIN(unusedRangeSize, outInfo.UnusedRangeSizeMin);
                outInfo.UnusedRangeSizeMax = D3D12MA_MAX(unusedRangeSize, outInfo.UnusedRangeSizeMax);
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAllocIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const Suballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const UINT64 unusedRangeSize = suballoc.offset - lastOffset;
                    ++outInfo.UnusedRangeCount;
                    outInfo.UnusedRangeSizeMin = D3D12MA_MIN(unusedRangeSize, outInfo.UnusedRangeSizeMin);
                    outInfo.UnusedRangeSizeMax = D3D12MA_MAX(unusedRangeSize, outInfo.UnusedRangeSizeMax);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                outInfo.UsedBytes += suballoc.size;
                ++outInfo.AllocationCount;
                outInfo.AllocationSizeMin = D3D12MA_MIN(suballoc.size, outInfo.AllocationSizeMin);
                outInfo.AllocationSizeMax = D3D12MA_MAX(suballoc.size, outInfo.AllocationSizeMax);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // There is free space from lastOffset to size.
                if (lastOffset < size)
                {
                    const UINT64 unusedRangeSize = size - lastOffset;
                    ++outInfo.UnusedRangeCount;
                    outInfo.UnusedRangeSizeMin = D3D12MA_MIN(unusedRangeSize, outInfo.UnusedRangeSizeMin);
                    outInfo.UnusedRangeSizeMax = D3D12MA_MAX(unusedRangeSize, outInfo.UnusedRangeSizeMax);
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }
    outInfo.UnusedBytes = size - outInfo.UsedBytes;
}

void BlockMetadata_Linear::WriteAllocationInfoToJson(JsonWriter& json) const
{
    const UINT64 size = GetSize();
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    // FIRST PASS

    size_t unusedRangeCount = 0;
    UINT64 usedBytes = 0;

    UINT64 lastOffset = 0;

    size_t alloc2ndCount = 0;
    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const UINT64 freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const Suballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    ++unusedRangeCount;
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++alloc2ndCount;
                usedBytes += suballoc.size;

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    ++unusedRangeCount;
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    size_t alloc1stCount = 0;
    const UINT64 freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const Suballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                ++unusedRangeCount;
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            ++alloc1stCount;
            usedBytes += suballoc.size;

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if (lastOffset < size)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                ++unusedRangeCount;
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const Suballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    ++unusedRangeCount;
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++alloc2ndCount;
                usedBytes += suballoc.size;

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    ++unusedRangeCount;
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    const UINT64 unusedBytes = size - usedBytes;
    PrintDetailedMap_Begin(json, unusedBytes, alloc1stCount + alloc2ndCount, unusedRangeCount);

    // SECOND PASS
    lastOffset = 0;
    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const UINT64 freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const Suballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const UINT64 unusedRangeSize = suballoc.offset - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    const UINT64 unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    nextAlloc1stIndex = m_1stNullItemsBeginCount;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const Suballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const UINT64 unusedRangeSize = suballoc.offset - lastOffset;
                PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if (lastOffset < freeSpace1stTo2ndEnd)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                const UINT64 unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const Suballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const UINT64 unusedRangeSize = suballoc.offset - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    const UINT64 unusedRangeSize = size - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    PrintDetailedMap_End(json);
}

Suballocation& BlockMetadata_Linear::FindSuballocation(UINT64 offset) const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    Suballocation refSuballoc;
    refSuballoc.offset = offset;
    // Rest of members stays uninitialized intentionally for better performance.

    // Item from the 1st vector.
    {
        const SuballocationVectorType::iterator it = BinaryFindSorted(
            suballocations1st.cbegin() + m_1stNullItemsBeginCount,
            suballocations1st.cend(),
            refSuballoc,
            SuballocationOffsetLess());
        if (it != suballocations1st.cend())
        {
            return *it;
        }
    }

    if (m_2ndVectorMode != SECOND_VECTOR_EMPTY)
    {
        // Rest of members stays uninitialized intentionally for better performance.
        const SuballocationVectorType::iterator it = m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ?
            BinaryFindSorted(suballocations2nd.cbegin(), suballocations2nd.cend(), refSuballoc, SuballocationOffsetLess()) :
            BinaryFindSorted(suballocations2nd.cbegin(), suballocations2nd.cend(), refSuballoc, SuballocationOffsetGreater());
        if (it != suballocations2nd.cend())
        {
            return *it;
        }
    }

    D3D12MA_ASSERT(0 && "Allocation not found in linear allocator!");
    return *suballocations1st.crbegin(); // Should never occur.
}

bool BlockMetadata_Linear::ShouldCompact1st() const
{
    const size_t nullItemCount = m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount;
    const size_t suballocCount = AccessSuballocations1st().size();
    return suballocCount > 32 && nullItemCount * 2 >= (suballocCount - nullItemCount) * 3;
}

void BlockMetadata_Linear::CleanupAfterFree()
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (IsEmpty())
    {
        suballocations1st.clear();
        suballocations2nd.clear();
        m_1stNullItemsBeginCount = 0;
        m_1stNullItemsMiddleCount = 0;
        m_2ndNullItemsCount = 0;
        m_2ndVectorMode = SECOND_VECTOR_EMPTY;
    }
    else
    {
        const size_t suballoc1stCount = suballocations1st.size();
        const size_t nullItem1stCount = m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount;
        D3D12MA_ASSERT(nullItem1stCount <= suballoc1stCount);

        // Find more null items at the beginning of 1st vector.
        while (m_1stNullItemsBeginCount < suballoc1stCount &&
            suballocations1st[m_1stNullItemsBeginCount].type == SUBALLOCATION_TYPE_FREE)
        {
            ++m_1stNullItemsBeginCount;
            --m_1stNullItemsMiddleCount;
        }

        // Find more null items at the end of 1st vector.
        while (m_1stNullItemsMiddleCount > 0 &&
            suballocations1st.back().type == SUBALLOCATION_TYPE_FREE)
        {
            --m_1stNullItemsMiddleCount;
            suballocations1st.pop_back();
        }

        // Find more null items at the end of 2nd vector.
        while (m_2ndNullItemsCount > 0 &&
            suballocations2nd.back().type == SUBALLOCATION_TYPE_FREE)
        {
            --m_2ndNullItemsCount;
            suballocations2nd.pop_back();
        }

        // Find more null items at the beginning of 2nd vector.
        while (m_2ndNullItemsCount > 0 &&
            suballocations2nd[0].type == SUBALLOCATION_TYPE_FREE)
        {
            --m_2ndNullItemsCount;
            suballocations2nd.remove(0);
        }

        if (ShouldCompact1st())
        {
            const size_t nonNullItemCount = suballoc1stCount - nullItem1stCount;
            size_t srcIndex = m_1stNullItemsBeginCount;
            for (size_t dstIndex = 0; dstIndex < nonNullItemCount; ++dstIndex)
            {
                while (suballocations1st[srcIndex].type == SUBALLOCATION_TYPE_FREE)
                {
                    ++srcIndex;
                }
                if (dstIndex != srcIndex)
                {
                    suballocations1st[dstIndex] = suballocations1st[srcIndex];
                }
                ++srcIndex;
            }
            suballocations1st.resize(nonNullItemCount);
            m_1stNullItemsBeginCount = 0;
            m_1stNullItemsMiddleCount = 0;
        }

        // 2nd vector became empty.
        if (suballocations2nd.empty())
        {
            m_2ndVectorMode = SECOND_VECTOR_EMPTY;
        }

        // 1st vector became empty.
        if (suballocations1st.size() - m_1stNullItemsBeginCount == 0)
        {
            suballocations1st.clear();
            m_1stNullItemsBeginCount = 0;

            if (!suballocations2nd.empty() && m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
            {
                // Swap 1st with 2nd. Now 2nd is empty.
                m_2ndVectorMode = SECOND_VECTOR_EMPTY;
                m_1stNullItemsMiddleCount = m_2ndNullItemsCount;
                while (m_1stNullItemsBeginCount < suballocations2nd.size() &&
                    suballocations2nd[m_1stNullItemsBeginCount].type == SUBALLOCATION_TYPE_FREE)
                {
                    ++m_1stNullItemsBeginCount;
                    --m_1stNullItemsMiddleCount;
                }
                m_2ndNullItemsCount = 0;
                m_1stVectorIndex ^= 1;
            }
        }
    }

    D3D12MA_HEAVY_ASSERT(Validate());
}

bool BlockMetadata_Linear::CreateAllocationRequest_LowerAddress(
    UINT64 allocSize,
    UINT64 allocAlignment,
    AllocationRequest* pAllocationRequest)
{
    const UINT64 blockSize = GetSize();
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (m_2ndVectorMode == SECOND_VECTOR_EMPTY || m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        // Try to allocate at the end of 1st vector.

        UINT64 resultBaseOffset = 0;
        if (!suballocations1st.empty())
        {
            const Suballocation& lastSuballoc = suballocations1st.back();
            resultBaseOffset = lastSuballoc.offset + lastSuballoc.size + GetDebugMargin();
        }

        // Start from offset equal to beginning of free space.
        UINT64 resultOffset = resultBaseOffset;
        // Apply alignment.
        resultOffset = AlignUp(resultOffset, allocAlignment);

        const UINT64 freeSpaceEnd = m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ?
            suballocations2nd.back().offset : blockSize;

        // There is enough free space at the end after alignment.
        if (resultOffset + allocSize + GetDebugMargin() <= freeSpaceEnd)
        {
            // All tests passed: Success.
            pAllocationRequest->allocHandle = (AllocHandle)(resultOffset + 1);
            // pAllocationRequest->item, customData unused.
            pAllocationRequest->algorithmData = ALLOC_REQUEST_END_OF_1ST;
            return true;
        }
    }

    // Wrap-around to end of 2nd vector. Try to allocate there, watching for the
    // beginning of 1st vector as the end of free space.
    if (m_2ndVectorMode == SECOND_VECTOR_EMPTY || m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        D3D12MA_ASSERT(!suballocations1st.empty());

        UINT64 resultBaseOffset = 0;
        if (!suballocations2nd.empty())
        {
            const Suballocation& lastSuballoc = suballocations2nd.back();
            resultBaseOffset = lastSuballoc.offset + lastSuballoc.size + GetDebugMargin();
        }

        // Start from offset equal to beginning of free space.
        UINT64 resultOffset = resultBaseOffset;

        // Apply alignment.
        resultOffset = AlignUp(resultOffset, allocAlignment);

        size_t index1st = m_1stNullItemsBeginCount;
        // There is enough free space at the end after alignment.
        if ((index1st == suballocations1st.size() && resultOffset + allocSize + GetDebugMargin() <= blockSize) ||
            (index1st < suballocations1st.size() && resultOffset + allocSize + GetDebugMargin() <= suballocations1st[index1st].offset))
        {
            // All tests passed: Success.
            pAllocationRequest->allocHandle = (AllocHandle)(resultOffset + 1);
            pAllocationRequest->algorithmData = ALLOC_REQUEST_END_OF_2ND;
            // pAllocationRequest->item, customData unused.
            return true;
        }
    }
    return false;
}

bool BlockMetadata_Linear::CreateAllocationRequest_UpperAddress(
    UINT64 allocSize,
    UINT64 allocAlignment,
    AllocationRequest* pAllocationRequest)
{
    const UINT64 blockSize = GetSize();
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        D3D12MA_ASSERT(0 && "Trying to use pool with linear algorithm as double stack, while it is already being used as ring buffer.");
        return false;
    }

    // Try to allocate before 2nd.back(), or end of block if 2nd.empty().
    if (allocSize > blockSize)
    {
        return false;
    }
    UINT64 resultBaseOffset = blockSize - allocSize;
    if (!suballocations2nd.empty())
    {
        const Suballocation& lastSuballoc = suballocations2nd.back();
        resultBaseOffset = lastSuballoc.offset - allocSize;
        if (allocSize > lastSuballoc.offset)
        {
            return false;
        }
    }

    // Start from offset equal to end of free space.
    UINT64 resultOffset = resultBaseOffset;
    // Apply debugMargin at the end.
    if (GetDebugMargin() > 0)
    {
        if (resultOffset < GetDebugMargin())
        {
            return false;
        }
        resultOffset -= GetDebugMargin();
    }

    // Apply alignment.
    resultOffset = AlignDown(resultOffset, allocAlignment);
    // There is enough free space.
    const UINT64 endOf1st = !suballocations1st.empty() ?
        suballocations1st.back().offset + suballocations1st.back().size : 0;

    if (endOf1st + GetDebugMargin() <= resultOffset)
    {
        // All tests passed: Success.
        pAllocationRequest->allocHandle = (AllocHandle)(resultOffset + 1);
        // pAllocationRequest->item unused.
        pAllocationRequest->algorithmData = ALLOC_REQUEST_UPPER_ADDRESS;
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////
// Private class BlockMetadata_TLSF implementation

BlockMetadata_TLSF::BlockMetadata_TLSF(const ALLOCATION_CALLBACKS* allocationCallbacks, bool isVirtual)
    : BlockMetadata(allocationCallbacks, isVirtual),
    m_AllocCount(0),
    m_BlocksFreeCount(0),
    m_BlocksFreeSize(0),
    m_IsFreeBitmap(0),
    m_MemoryClasses(0),
    m_ListsCount(0),
    m_FreeList(NULL),
    m_BlockAllocator(*allocationCallbacks, INITIAL_BLOCK_ALLOC_COUNT),
    m_NullBlock(NULL)
{
    D3D12MA_ASSERT(allocationCallbacks);
}

BlockMetadata_TLSF::~BlockMetadata_TLSF()
{
    if (m_FreeList)
        D3D12MA_DELETE_ARRAY(*GetAllocs(), m_FreeList, m_ListsCount);
}

void BlockMetadata_TLSF::Init(UINT64 size)
{
    BlockMetadata::Init(size);

    m_NullBlock = m_BlockAllocator.Alloc();
    m_NullBlock->size = size;
    m_NullBlock->offset = 0;
    m_NullBlock->prevPhysical = NULL;
    m_NullBlock->nextPhysical = NULL;
    m_NullBlock->MarkFree();
    m_NullBlock->NextFree() = NULL;
    m_NullBlock->PrevFree() = NULL;
    UINT8 memoryClass = SizeToMemoryClass(size);
    UINT16 sli = SizeToSecondIndex(size, memoryClass);
    m_ListsCount = (memoryClass == 0 ? 0 : (memoryClass - 1) * (1UL << SECOND_LEVEL_INDEX) + sli) + 1;
    if (IsVirtual())
        m_ListsCount += 1UL << SECOND_LEVEL_INDEX;
    else
        m_ListsCount += 4;

    m_MemoryClasses = memoryClass + 2;
    memset(m_InnerIsFreeBitmap, 0, MAX_MEMORY_CLASSES * sizeof(UINT32));

    m_FreeList = D3D12MA_NEW_ARRAY(*GetAllocs(), Block*, m_ListsCount);
    memset(m_FreeList, 0, m_ListsCount * sizeof(Block*));
}

bool BlockMetadata_TLSF::Validate() const
{
    D3D12MA_VALIDATE(GetSumFreeSize() <= GetSize());

    UINT64 calculatedSize = m_NullBlock->size;
    UINT64 calculatedFreeSize = m_NullBlock->size;
    size_t allocCount = 0;
    size_t freeCount = 0;

    // Check integrity of free lists
    for (UINT32 list = 0; list < m_ListsCount; ++list)
    {
        Block* block = m_FreeList[list];
        if (block != NULL)
        {
            D3D12MA_VALIDATE(block->IsFree());
            D3D12MA_VALIDATE(block->PrevFree() == NULL);
            while (block->NextFree())
            {
                D3D12MA_VALIDATE(block->NextFree()->IsFree());
                D3D12MA_VALIDATE(block->NextFree()->PrevFree() == block);
                block = block->NextFree();
            }
        }
    }

    D3D12MA_VALIDATE(m_NullBlock->nextPhysical == NULL);
    if (m_NullBlock->prevPhysical)
    {
        D3D12MA_VALIDATE(m_NullBlock->prevPhysical->nextPhysical == m_NullBlock);
    }

    // Check all blocks
    UINT64 nextOffset = m_NullBlock->offset;
    for (Block* prev = m_NullBlock->prevPhysical; prev != NULL; prev = prev->prevPhysical)
    {
        D3D12MA_VALIDATE(prev->offset + prev->size == nextOffset);
        nextOffset = prev->offset;
        calculatedSize += prev->size;

        UINT32 listIndex = GetListIndex(prev->size);
        if (prev->IsFree())
        {
            ++freeCount;
            // Check if free block belongs to free list
            Block* freeBlock = m_FreeList[listIndex];
            D3D12MA_VALIDATE(freeBlock != NULL);

            bool found = false;
            do
            {
                if (freeBlock == prev)
                    found = true;

                freeBlock = freeBlock->NextFree();
            } while (!found && freeBlock != NULL);

            D3D12MA_VALIDATE(found);
            calculatedFreeSize += prev->size;
        }
        else
        {
            ++allocCount;
            // Check if taken block is not on a free list
            Block* freeBlock = m_FreeList[listIndex];
            while (freeBlock)
            {
                D3D12MA_VALIDATE(freeBlock != prev);
                freeBlock = freeBlock->NextFree();
            }
        }

        if (prev->prevPhysical)
        {
            D3D12MA_VALIDATE(prev->prevPhysical->nextPhysical == prev);
        }
    }

    D3D12MA_VALIDATE(nextOffset == 0);
    D3D12MA_VALIDATE(calculatedSize == GetSize());
    D3D12MA_VALIDATE(calculatedFreeSize == GetSumFreeSize());
    D3D12MA_VALIDATE(allocCount == m_AllocCount);
    D3D12MA_VALIDATE(freeCount == m_BlocksFreeCount);

    return true;
}

void BlockMetadata_TLSF::GetAllocationInfo(AllocHandle allocHandle, VIRTUAL_ALLOCATION_INFO& outInfo) const
{
    Block* block = (Block*)allocHandle;
    D3D12MA_ASSERT(!block->IsFree() && "Cannot get allocation info for free block!");
    outInfo.Offset = block->offset;
    outInfo.Size = block->size;
    outInfo.pUserData = block->UserData();
}

bool BlockMetadata_TLSF::CreateAllocationRequest(
    UINT64 allocSize,
    UINT64 allocAlignment,
    bool upperAddress,
    AllocationRequest* pAllocationRequest)
{
    D3D12MA_ASSERT(allocSize > 0 && "Cannot allocate empty block!");
    D3D12MA_ASSERT(!upperAddress && "ALLOCATION_FLAG_UPPER_ADDRESS can be used only with linear algorithm.");
    D3D12MA_ASSERT(pAllocationRequest != NULL); 
    D3D12MA_HEAVY_ASSERT(Validate());

    allocSize += GetDebugMargin();
    // Quick check for too small pool
    if (allocSize > GetSumFreeSize())
        return false;

    // If no free blocks in pool then check only null block
    if (m_BlocksFreeCount == 0)
        return CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, pAllocationRequest);

    // Round up to the next block
    UINT64 sizeForNextList = allocSize;
    UINT64 smallSizeStep = SMALL_BUFFER_SIZE / (IsVirtual() ? 1 << SECOND_LEVEL_INDEX : 4);
    if (allocSize > SMALL_BUFFER_SIZE)
    {
        sizeForNextList += (1ULL << (BitScanMSB(allocSize) - SECOND_LEVEL_INDEX));
    }
    else if (allocSize > SMALL_BUFFER_SIZE - smallSizeStep)
        sizeForNextList = SMALL_BUFFER_SIZE + 1;
    else
        sizeForNextList += smallSizeStep;

    // Check larger bucket
    UINT32 nextListIndex = 0;
    Block* nextListBlock = FindFreeBlock(sizeForNextList, nextListIndex);
    while (nextListBlock)
    {
        if (CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, pAllocationRequest))
            return true;
        nextListBlock = nextListBlock->NextFree();
    }

    // If failed check null block
    if (CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, pAllocationRequest))
        return true;

    // Check best fit bucket
    UINT32 prevListIndex = 0;
    Block* prevListBlock = FindFreeBlock(allocSize, prevListIndex);
    while (prevListBlock)
    {
        if (CheckBlock(*prevListBlock, prevListIndex, allocSize, allocAlignment, pAllocationRequest))
            return true;
        prevListBlock = prevListBlock->NextFree();
    }

    // Worst case, full search has to be done
    while (++nextListIndex < m_ListsCount)
    {
        nextListBlock = m_FreeList[nextListIndex];
        while (nextListBlock)
        {
            if (CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->NextFree();
        }
    }

    // No more memory sadly
    return false;
}

void BlockMetadata_TLSF::Alloc(
    const AllocationRequest& request,
    UINT64 allocSize,
    void* userData)
{
    // Get block and pop it from the free list
    Block* currentBlock = (Block*)request.allocHandle;
    UINT64 offset = request.algorithmData;
    D3D12MA_ASSERT(currentBlock != NULL);
    D3D12MA_ASSERT(currentBlock->offset <= offset);

    if (currentBlock != m_NullBlock)
        RemoveFreeBlock(currentBlock);

    // Append missing alignment to prev block or create new one
    UINT64 misssingAlignment = offset - currentBlock->offset;
    if (misssingAlignment)
    {
        Block* prevBlock = currentBlock->prevPhysical;
        D3D12MA_ASSERT(prevBlock != NULL && "There should be no missing alignment at offset 0!");

        if (prevBlock->IsFree() && prevBlock->size != GetDebugMargin())
        {
            UINT32 oldList = GetListIndex(prevBlock->size);
            prevBlock->size += misssingAlignment;
            // Check if new size crosses list bucket
            if (oldList != GetListIndex(prevBlock->size))
            {
                prevBlock->size -= misssingAlignment;
                RemoveFreeBlock(prevBlock);
                prevBlock->size += misssingAlignment;
                InsertFreeBlock(prevBlock);
            }
            else
                m_BlocksFreeSize += misssingAlignment;
        }
        else
        {
            Block* newBlock = m_BlockAllocator.Alloc();
            currentBlock->prevPhysical = newBlock;
            prevBlock->nextPhysical = newBlock;
            newBlock->prevPhysical = prevBlock;
            newBlock->nextPhysical = currentBlock;
            newBlock->size = misssingAlignment;
            newBlock->offset = currentBlock->offset;
            newBlock->MarkTaken();

            InsertFreeBlock(newBlock);
        }

        currentBlock->size -= misssingAlignment;
        currentBlock->offset += misssingAlignment;
    }

    UINT64 size = request.size + GetDebugMargin();
    if (currentBlock->size == size)
    {
        if (currentBlock == m_NullBlock)
        {
            // Setup new null block
            m_NullBlock = m_BlockAllocator.Alloc();
            m_NullBlock->size = 0;
            m_NullBlock->offset = currentBlock->offset + size;
            m_NullBlock->prevPhysical = currentBlock;
            m_NullBlock->nextPhysical = NULL;
            m_NullBlock->MarkFree();
            m_NullBlock->PrevFree() = NULL;
            m_NullBlock->NextFree() = NULL;
            currentBlock->nextPhysical = m_NullBlock;
            currentBlock->MarkTaken();
        }
    }
    else
    {
        D3D12MA_ASSERT(currentBlock->size > size && "Proper block already found, shouldn't find smaller one!");

        // Create new free block
        Block* newBlock = m_BlockAllocator.Alloc();
        newBlock->size = currentBlock->size - size;
        newBlock->offset = currentBlock->offset + size;
        newBlock->prevPhysical = currentBlock;
        newBlock->nextPhysical = currentBlock->nextPhysical;
        currentBlock->nextPhysical = newBlock;
        currentBlock->size = size;

        if (currentBlock == m_NullBlock)
        {
            m_NullBlock = newBlock;
            m_NullBlock->MarkFree();
            m_NullBlock->NextFree() = NULL;
            m_NullBlock->PrevFree() = NULL;
            currentBlock->MarkTaken();
        }
        else
        {
            newBlock->nextPhysical->prevPhysical = newBlock;
            newBlock->MarkTaken();
            InsertFreeBlock(newBlock);
        }
    }
    currentBlock->UserData() = userData;

    if (GetDebugMargin() > 0)
    {
        currentBlock->size -= GetDebugMargin();
        Block* newBlock = m_BlockAllocator.Alloc();
        newBlock->size = GetDebugMargin();
        newBlock->offset = currentBlock->offset + currentBlock->size;
        newBlock->prevPhysical = currentBlock;
        newBlock->nextPhysical = currentBlock->nextPhysical;
        newBlock->MarkTaken();
        currentBlock->nextPhysical->prevPhysical = newBlock;
        currentBlock->nextPhysical = newBlock;
        InsertFreeBlock(newBlock);
    }
    ++m_AllocCount;
}

void BlockMetadata_TLSF::Free(AllocHandle allocHandle)
{
    Block* block = (Block*)allocHandle;
    Block* next = block->nextPhysical;
    D3D12MA_ASSERT(!block->IsFree() && "Block is already free!");

    --m_AllocCount;
    if (GetDebugMargin() > 0)
    {
        RemoveFreeBlock(next);
        MergeBlock(next, block);
        block = next;
        next = next->nextPhysical;
    }

    // Try merging
    Block* prev = block->prevPhysical;
    if (prev != NULL && prev->IsFree() && prev->size != GetDebugMargin())
    {
        RemoveFreeBlock(prev);
        MergeBlock(block, prev);
    }

    if (!next->IsFree())
        InsertFreeBlock(block);
    else if (next == m_NullBlock)
        MergeBlock(m_NullBlock, block);
    else
    {
        RemoveFreeBlock(next);
        MergeBlock(next, block);
        InsertFreeBlock(next);
    }
}

void BlockMetadata_TLSF::Clear()
{
    m_AllocCount = 0;
    m_BlocksFreeCount = 0;
    m_BlocksFreeSize = 0;
    m_IsFreeBitmap = 0;
    m_NullBlock->offset = 0;
    m_NullBlock->size = GetSize();
    Block* block = m_NullBlock->prevPhysical;
    m_NullBlock->prevPhysical = NULL;
    while (block)
    {
        Block* prev = block->prevPhysical;
        m_BlockAllocator.Free(block);
        block = prev;
    }
    memset(m_FreeList, 0, m_ListsCount * sizeof(Block*));
    memset(m_InnerIsFreeBitmap, 0, m_MemoryClasses * sizeof(UINT32));
}

void BlockMetadata_TLSF::SetAllocationUserData(AllocHandle allocHandle, void* userData)
{
    Block* block = (Block*)allocHandle;
    D3D12MA_ASSERT(!block->IsFree() && "Trying to set user data for not allocated block!");
    block->UserData() = userData;
}

void BlockMetadata_TLSF::CalcAllocationStatInfo(StatInfo& outInfo) const
{
    outInfo.BlockCount = 1;
    outInfo.AllocationCount = static_cast<UINT>(m_AllocCount);
    outInfo.UnusedRangeCount = static_cast<UINT>(m_BlocksFreeCount);

    outInfo.UnusedBytes = GetSumFreeSize();
    outInfo.UsedBytes = GetSize() - outInfo.UnusedBytes;

    outInfo.AllocationSizeMin = UINT64_MAX;
    outInfo.AllocationSizeMax = 0;
    outInfo.UnusedRangeSizeMin = UINT64_MAX;
    outInfo.UnusedRangeSizeMax = 0;

    for (Block* block = m_NullBlock->prevPhysical; block != NULL; block = block->prevPhysical)
    {
        if (block->IsFree())
        {
            outInfo.UnusedRangeSizeMin = D3D12MA_MIN(block->size, outInfo.UnusedRangeSizeMin);
            outInfo.UnusedRangeSizeMax = D3D12MA_MAX(block->size, outInfo.UnusedRangeSizeMax);
        }
        else
        {
            outInfo.AllocationSizeMin = D3D12MA_MIN(block->size, outInfo.AllocationSizeMin);
            outInfo.AllocationSizeMax = D3D12MA_MAX(block->size, outInfo.AllocationSizeMax);
        }
    }

    if (m_NullBlock->size > 0)
    {
        ++outInfo.UnusedRangeCount;
        outInfo.UnusedRangeSizeMin = D3D12MA_MIN(m_NullBlock->size, outInfo.UnusedRangeSizeMin);
        outInfo.UnusedRangeSizeMax = D3D12MA_MAX(m_NullBlock->size, outInfo.UnusedRangeSizeMax);
    }
}

void BlockMetadata_TLSF::WriteAllocationInfoToJson(JsonWriter& json) const
{
    size_t blockCount = m_AllocCount + m_BlocksFreeCount;
    Vector<Block*> blockList(blockCount, *GetAllocs());

    size_t i = blockCount;
    if (m_NullBlock->size > 0)
    {
        ++blockCount;
        blockList.push_back(m_NullBlock);
    }
    for (Block* block = m_NullBlock->prevPhysical; block != NULL; block = block->prevPhysical)
    {
        blockList[--i] = block;
    }
    D3D12MA_ASSERT(i == 0);

    PrintDetailedMap_Begin(json, GetSumFreeSize(), GetAllocationCount(), m_BlocksFreeCount + static_cast<bool>(m_NullBlock->size));
    for (; i < blockCount; ++i)
    {
        Block* block = blockList[i];
        if (block->IsFree())
            PrintDetailedMap_UnusedRange(json, block->offset, block->size);
        else
            PrintDetailedMap_Allocation(json, block->size, block->offset, block->UserData());
    }
    PrintDetailedMap_End(json);
}

UINT8 BlockMetadata_TLSF::SizeToMemoryClass(UINT64 size) const
{
    if (size > SMALL_BUFFER_SIZE)
        return BitScanMSB(size) - MEMORY_CLASS_SHIFT;
    return 0;
}

UINT16 BlockMetadata_TLSF::SizeToSecondIndex(UINT64 size, UINT8 memoryClass) const
{
    if (memoryClass == 0)
    {
        if (IsVirtual())
            return static_cast<UINT16>((size - 1) / 8);
        else
            return static_cast<UINT16>((size - 1) / 64);
    }
    return static_cast<UINT16>((size >> (memoryClass + MEMORY_CLASS_SHIFT - SECOND_LEVEL_INDEX)) ^ (1U << SECOND_LEVEL_INDEX));
}

UINT32 BlockMetadata_TLSF::GetListIndex(UINT8 memoryClass, UINT16 secondIndex) const
{
    if (memoryClass == 0)
        return secondIndex;

    const UINT32 index = static_cast<UINT32>(memoryClass - 1) * (1 << SECOND_LEVEL_INDEX) + secondIndex;
    if (IsVirtual())
        return index + (1 << SECOND_LEVEL_INDEX);
    else
        return index + 4;
}

UINT32 BlockMetadata_TLSF::GetListIndex(UINT64 size) const
{
    UINT8 memoryClass = SizeToMemoryClass(size);
    return GetListIndex(memoryClass, SizeToSecondIndex(size, memoryClass));
}

void BlockMetadata_TLSF::RemoveFreeBlock(Block* block)
{
    D3D12MA_ASSERT(block != m_NullBlock);
    D3D12MA_ASSERT(block->IsFree());

    if (block->NextFree() != NULL)
        block->NextFree()->PrevFree() = block->PrevFree();
    if (block->PrevFree() != NULL)
        block->PrevFree()->NextFree() = block->NextFree();
    else
    {
        UINT8 memClass = SizeToMemoryClass(block->size);
        UINT16 secondIndex = SizeToSecondIndex(block->size, memClass);
        UINT32 index = GetListIndex(memClass, secondIndex);
        m_FreeList[index] = block->NextFree();
        if (block->NextFree() == NULL)
        {
            m_InnerIsFreeBitmap[memClass] &= ~(1U << secondIndex);
            if (m_InnerIsFreeBitmap[memClass] == 0)
                m_IsFreeBitmap &= ~(1UL << memClass);
        }
    }
    block->MarkTaken();
    block->UserData() = NULL;
    --m_BlocksFreeCount;
    m_BlocksFreeSize -= block->size;
}

void BlockMetadata_TLSF::InsertFreeBlock(Block* block)
{
    D3D12MA_ASSERT(block != m_NullBlock);
    D3D12MA_ASSERT(!block->IsFree() && "Cannot insert block twice!");

    UINT8 memClass = SizeToMemoryClass(block->size);
    UINT16 secondIndex = SizeToSecondIndex(block->size, memClass);
    UINT32 index = GetListIndex(memClass, secondIndex);
    block->PrevFree() = NULL;
    block->NextFree() = m_FreeList[index];
    m_FreeList[index] = block;
    if (block->NextFree() != NULL)
        block->NextFree()->PrevFree() = block;
    else
    {
        m_InnerIsFreeBitmap[memClass] |= 1U << secondIndex;
        m_IsFreeBitmap |= 1UL << memClass;
    }
    ++m_BlocksFreeCount;
    m_BlocksFreeSize += block->size;
}

void BlockMetadata_TLSF::MergeBlock(Block* block, Block* prev)
{
    D3D12MA_ASSERT(block->prevPhysical == prev && "Cannot merge seperate physical regions!");
    D3D12MA_ASSERT(!prev->IsFree() && "Cannot merge block that belongs to free list!");

    block->offset = prev->offset;
    block->size += prev->size;
    block->prevPhysical = prev->prevPhysical;
    if (block->prevPhysical)
        block->prevPhysical->nextPhysical = block;
    m_BlockAllocator.Free(prev);
}

BlockMetadata_TLSF::Block* BlockMetadata_TLSF::FindFreeBlock(UINT64 size, UINT32& listIndex) const
{
    UINT8 memoryClass = SizeToMemoryClass(size);
    UINT32 innerFreeMap = m_InnerIsFreeBitmap[memoryClass] & (~0U << SizeToSecondIndex(size, memoryClass));
    if (!innerFreeMap)
    {
        // Check higher levels for avaiable blocks
        UINT32 freeMap = m_IsFreeBitmap & (~0UL << (memoryClass + 1));
        if (!freeMap)
            return NULL; // No more memory avaible

        // Find lowest free region
        memoryClass = BitScanLSB(freeMap);
        innerFreeMap = m_InnerIsFreeBitmap[memoryClass];
        D3D12MA_ASSERT(innerFreeMap != 0);
    }
    // Find lowest free subregion
    listIndex = GetListIndex(memoryClass, BitScanLSB(innerFreeMap));
    return m_FreeList[listIndex];
}

bool BlockMetadata_TLSF::CheckBlock(
    Block& block,
    UINT32 listIndex,
    UINT64 allocSize,
    UINT64 allocAlignment,
    AllocationRequest* pAllocationRequest)
{
    D3D12MA_ASSERT(block.IsFree() && "Block is already taken!");

    UINT64 alignedOffset = AlignUp(block.offset, allocAlignment);
    if (block.size < allocSize + alignedOffset - block.offset)
        return false;

    // Alloc successful
    pAllocationRequest->allocHandle = (AllocHandle)&block;
    pAllocationRequest->size = allocSize - GetDebugMargin();
    pAllocationRequest->algorithmData = alignedOffset;

    // Place block at the start of list if it's normal block
    if (listIndex != m_ListsCount && block.PrevFree())
    {
        block.PrevFree()->NextFree() = block.NextFree();
        if (block.NextFree())
            block.NextFree()->PrevFree() = block.PrevFree();
        block.PrevFree() = NULL;
        block.NextFree() = m_FreeList[listIndex];
        m_FreeList[listIndex] = &block;
        if (block.NextFree())
            block.NextFree()->PrevFree() = &block;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Private class NormalBlock implementation

NormalBlock::NormalBlock(
    AllocatorPimpl* allocator,
    BlockVector* blockVector,
    const D3D12_HEAP_PROPERTIES& heapProps,
    D3D12_HEAP_FLAGS heapFlags,
    UINT64 size,
    UINT id) :
    MemoryBlock(allocator, heapProps, heapFlags, size, id),
    m_pMetadata(NULL),
    m_BlockVector(blockVector)
{
}

NormalBlock::~NormalBlock()
{
    if(m_pMetadata != NULL)
    {
        // THIS IS THE MOST IMPORTANT ASSERT IN THE ENTIRE LIBRARY!
        // Hitting it means you have some memory leak - unreleased Allocation objects.
        D3D12MA_ASSERT(m_pMetadata->IsEmpty() && "Some allocations were not freed before destruction of this memory block!");

        D3D12MA_DELETE(m_Allocator->GetAllocs(), m_pMetadata);
    }
}

HRESULT NormalBlock::Init(UINT32 algorithm, ID3D12ProtectedResourceSession* pProtectedSession)
{
    HRESULT hr = MemoryBlock::Init(pProtectedSession);
    if(FAILED(hr))
    {
        return hr;
    }
    
    switch (algorithm)
    {
    case POOL_FLAG_ALGORITHM_TLSF:
        m_pMetadata = D3D12MA_NEW(m_Allocator->GetAllocs(), BlockMetadata_TLSF)(&m_Allocator->GetAllocs(), false);
        break;
    case POOL_FLAG_ALGORITHM_LINEAR:
        m_pMetadata = D3D12MA_NEW(m_Allocator->GetAllocs(), BlockMetadata_Linear)(&m_Allocator->GetAllocs(), false);
        break;
    default:
        D3D12MA_ASSERT(0);
    case 0:
        m_pMetadata = D3D12MA_NEW(m_Allocator->GetAllocs(), BlockMetadata_Generic)(&m_Allocator->GetAllocs(), false);
        break;
    }
    m_pMetadata->Init(m_Size);

    return hr;
}

bool NormalBlock::Validate() const
{
    D3D12MA_VALIDATE(GetHeap() &&
        m_pMetadata &&
        m_pMetadata->GetSize() != 0 &&
        m_pMetadata->GetSize() == GetSize());
    return m_pMetadata->Validate();
}

////////////////////////////////////////////////////////////////////////////////
// Private class MemoryBlock definition

MemoryBlock::MemoryBlock(
    AllocatorPimpl* allocator,
    const D3D12_HEAP_PROPERTIES& heapProps,
    D3D12_HEAP_FLAGS heapFlags,
    UINT64 size,
    UINT id) :
    m_Allocator(allocator),
    m_HeapProps(heapProps),
    m_HeapFlags(heapFlags),
    m_Size(size),
    m_Id(id)
{
}

MemoryBlock::~MemoryBlock()
{
    if(m_Heap)
    {
        m_Allocator->m_Budget.m_BlockBytes[HeapTypeToIndex(m_HeapProps.Type)] -= m_Size;
        m_Heap->Release();
    }
}

HRESULT MemoryBlock::Init(ID3D12ProtectedResourceSession* pProtectedSession)
{
    D3D12MA_ASSERT(m_Heap == NULL && m_Size > 0);

    D3D12_HEAP_DESC heapDesc = {};
    heapDesc.SizeInBytes = m_Size;
    heapDesc.Properties = m_HeapProps;
    heapDesc.Alignment = HeapFlagsToAlignment(m_HeapFlags);
    heapDesc.Flags = m_HeapFlags;

#ifdef __ID3D12Device4_INTERFACE_DEFINED__
    HRESULT hr = m_Allocator->GetDevice4()->CreateHeap1(&heapDesc, pProtectedSession, D3D12MA_IID_PPV_ARGS(&m_Heap));
#else
    D3D12MA_ASSERT(pProtectedSession == NULL);
    HRESULT hr = m_Allocator->GetDevice()->CreateHeap(&heapDesc, D3D12MA_IID_PPV_ARGS(&m_Heap));
#endif
    if(SUCCEEDED(hr))
    {
        m_Allocator->m_Budget.m_BlockBytes[HeapTypeToIndex(m_HeapProps.Type)] += m_Size;
    }
    return hr;
}

////////////////////////////////////////////////////////////////////////////////
// Private class CommittedAllocationList implementation

CommittedAllocationList::CommittedAllocationList()
{
}

void CommittedAllocationList::Init(bool useMutex, D3D12_HEAP_TYPE heapType, PoolPimpl* pool)
{
    m_UseMutex = useMutex;
    m_HeapType = heapType;
    m_Pool = pool;
}

CommittedAllocationList::~CommittedAllocationList()
{
    if(!m_AllocationList.IsEmpty())
    {
        D3D12MA_ASSERT(0 && "Unfreed committed allocations found!");
    }
}

void CommittedAllocationList::CalculateStats(StatInfo& outStats)
{
    ZeroMemory(&outStats, sizeof(outStats));
    outStats.AllocationSizeMin = UINT64_MAX;
    outStats.UnusedRangeSizeMin = UINT64_MAX;

    MutexLockRead lock(m_Mutex, m_UseMutex);

    for(Allocation* alloc = m_AllocationList.Front();
        alloc != NULL; alloc = m_AllocationList.GetNext(alloc))
    {
        const UINT64 size = alloc->GetSize();
        ++outStats.BlockCount;
        ++outStats.AllocationCount;
        outStats.UsedBytes += size;
        if(size > outStats.AllocationSizeMax)
            outStats.AllocationSizeMax = size;
        if(size < outStats.AllocationSizeMin)
            outStats.AllocationSizeMin = size;
    }
}

void CommittedAllocationList::BuildStatsString(JsonWriter& json)
{
    MutexLockRead lock(m_Mutex, m_UseMutex);

    json.BeginArray();
    for(Allocation* alloc = m_AllocationList.Front();
        alloc != NULL; alloc = m_AllocationList.GetNext(alloc))
    {
        json.BeginObject(true);
        json.AddAllocationToObject(*alloc);
        json.EndObject();
    }
    json.EndArray();
}

void CommittedAllocationList::Register(Allocation* alloc)
{
    MutexLockWrite lock(m_Mutex, m_UseMutex);
    m_AllocationList.PushBack(alloc);
}

void CommittedAllocationList::Unregister(Allocation* alloc)
{
    MutexLockWrite lock(m_Mutex, m_UseMutex);
    m_AllocationList.Remove(alloc);
}

////////////////////////////////////////////////////////////////////////////////
// Private class BlockVector implementation

BlockVector::BlockVector(
    AllocatorPimpl* hAllocator,
    const D3D12_HEAP_PROPERTIES& heapProps,
    D3D12_HEAP_FLAGS heapFlags,
    UINT64 preferredBlockSize,
    size_t minBlockCount,
    size_t maxBlockCount,
    bool explicitBlockSize,
    UINT64 minAllocationAlignment,
    UINT32 algorithm,
    ID3D12ProtectedResourceSession* pProtectedSession) :
    m_hAllocator(hAllocator),
    m_HeapProps(heapProps),
    m_HeapFlags(heapFlags),
    m_PreferredBlockSize(preferredBlockSize),
    m_MinBlockCount(minBlockCount),
    m_MaxBlockCount(maxBlockCount),
    m_ExplicitBlockSize(explicitBlockSize),
    m_MinAllocationAlignment(minAllocationAlignment),
    m_Algorithm(algorithm),
    m_ProtectedSession(pProtectedSession),
    m_HasEmptyBlock(false),
    m_Blocks(hAllocator->GetAllocs()),
    m_NextBlockId(0)
{
}

BlockVector::~BlockVector()
{
    for(size_t i = m_Blocks.size(); i--; )
    {
        D3D12MA_DELETE(m_hAllocator->GetAllocs(), m_Blocks[i]);
    }
}

HRESULT BlockVector::CreateMinBlocks()
{
    for(size_t i = 0; i < m_MinBlockCount; ++i)
    {
        HRESULT hr = CreateBlock(m_PreferredBlockSize, NULL);
        if(FAILED(hr))
        {
            return hr;
        }
    }
    return S_OK;
}

bool BlockVector::IsEmpty()
{
    MutexLockRead lock(m_Mutex, m_hAllocator->UseMutex());
    return m_Blocks.empty();
}

HRESULT BlockVector::Allocate(
    UINT64 size,
    UINT64 alignment,
    const ALLOCATION_DESC& allocDesc,
    size_t allocationCount,
    Allocation** pAllocations)
{
    size_t allocIndex;
    HRESULT hr = S_OK;

    {
        MutexLockWrite lock(m_Mutex, m_hAllocator->UseMutex());
        for(allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
        {
            hr = AllocatePage(
                size,
                alignment,
                allocDesc,
                pAllocations + allocIndex);
            if(FAILED(hr))
            {
                break;
            }
        }
    }

    if(FAILED(hr))
    {
        // Free all already created allocations.
        while(allocIndex--)
        {
            Free(pAllocations[allocIndex]);
        }
        ZeroMemory(pAllocations, sizeof(Allocation*) * allocationCount);
    }

    return hr;
}

HRESULT BlockVector::AllocatePage(
    UINT64 size,
    UINT64 alignment,
    const ALLOCATION_DESC& allocDesc,
    Allocation** pAllocation)
{
    // Early reject: requested allocation size is larger that maximum block size for this block vector.
    if(size + D3D12MA_DEBUG_MARGIN > m_PreferredBlockSize)
    {
        return E_OUTOFMEMORY;
    }

    UINT64 freeMemory = UINT64_MAX;
    if(IsHeapTypeStandard(m_HeapProps.Type))
    {
        Budget budget = {};
        m_hAllocator->GetBudgetForHeapType(budget, m_HeapProps.Type);
        freeMemory = (budget.UsageBytes < budget.BudgetBytes) ? (budget.BudgetBytes - budget.UsageBytes) : 0;
    }

    const bool canCreateNewBlock =
        ((allocDesc.Flags & ALLOCATION_FLAG_NEVER_ALLOCATE) == 0) &&
        (m_Blocks.size() < m_MaxBlockCount) &&
        // Even if we don't have to stay within budget with this allocation, when the
        // budget would be exceeded, we don't want to allocate new blocks, but always
        // create resources as committed.
        freeMemory >= size;

    // 1. Search existing allocations
    {
        // Forward order in m_Blocks - prefer blocks with smallest amount of free space.
        for(size_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex )
        {
            NormalBlock* const pCurrBlock = m_Blocks[blockIndex];
            D3D12MA_ASSERT(pCurrBlock);
            HRESULT hr = AllocateFromBlock(
                pCurrBlock,
                size,
                alignment,
                allocDesc.Flags,
                pAllocation);
            if(SUCCEEDED(hr))
            {
                return hr;
            }
        }
    }

    // 2. Try to create new block.
    if(canCreateNewBlock)
    {
        // Calculate optimal size for new block.
        UINT64 newBlockSize = m_PreferredBlockSize;
        UINT newBlockSizeShift = 0;

        if(!m_ExplicitBlockSize)
        {
            // Allocate 1/8, 1/4, 1/2 as first blocks.
            const UINT64 maxExistingBlockSize = CalcMaxBlockSize();
            for(UINT i = 0; i < NEW_BLOCK_SIZE_SHIFT_MAX; ++i)
            {
                const UINT64 smallerNewBlockSize = newBlockSize / 2;
                if(smallerNewBlockSize > maxExistingBlockSize && smallerNewBlockSize >= size * 2)
                {
                    newBlockSize = smallerNewBlockSize;
                    ++newBlockSizeShift;
                }
                else
                {
                    break;
                }
            }
        }

        size_t newBlockIndex = 0;
        HRESULT hr = newBlockSize <= freeMemory ?
            CreateBlock(newBlockSize, &newBlockIndex) : E_OUTOFMEMORY;
        // Allocation of this size failed? Try 1/2, 1/4, 1/8 of m_PreferredBlockSize.
        if(!m_ExplicitBlockSize)
        {
            while(FAILED(hr) && newBlockSizeShift < NEW_BLOCK_SIZE_SHIFT_MAX)
            {
                const UINT64 smallerNewBlockSize = newBlockSize / 2;
                if(smallerNewBlockSize >= size)
                {
                    newBlockSize = smallerNewBlockSize;
                    ++newBlockSizeShift;
                    hr = newBlockSize <= freeMemory ?
                        CreateBlock(newBlockSize, &newBlockIndex) : E_OUTOFMEMORY;
                }
                else
                {
                    break;
                }
            }
        }

        if(SUCCEEDED(hr))
        {
            NormalBlock* const pBlock = m_Blocks[newBlockIndex];
            D3D12MA_ASSERT(pBlock->m_pMetadata->GetSize() >= size);

            hr = AllocateFromBlock(
                pBlock,
                size,
                alignment,
                allocDesc.Flags,
                pAllocation);
            if(SUCCEEDED(hr))
            {
                return hr;
            }
            else
            {
                // Allocation from new block failed, possibly due to D3D12MA_DEBUG_MARGIN or alignment.
                return E_OUTOFMEMORY;
            }
        }
    }

    return E_OUTOFMEMORY;
}

void BlockVector::Free(Allocation* hAllocation)
{
    NormalBlock* pBlockToDelete = NULL;

    bool budgetExceeded = false;
    if(IsHeapTypeStandard(m_HeapProps.Type))
    {
        Budget budget = {};
        m_hAllocator->GetBudgetForHeapType(budget, m_HeapProps.Type);
        budgetExceeded = budget.UsageBytes >= budget.BudgetBytes;
    }

    // Scope for lock.
    {
        MutexLockWrite lock(m_Mutex, m_hAllocator->UseMutex());

        NormalBlock* pBlock = hAllocation->m_Placed.block;

        pBlock->m_pMetadata->Free(hAllocation->GetAllocHandle());
        D3D12MA_HEAVY_ASSERT(pBlock->Validate());

        const size_t blockCount = m_Blocks.size();
        // pBlock became empty after this deallocation.
        if(pBlock->m_pMetadata->IsEmpty())
        {
            // Already has empty Allocation. We don't want to have two, so delete this one.
            if((m_HasEmptyBlock || budgetExceeded) &&
                blockCount > m_MinBlockCount)
            {
                pBlockToDelete = pBlock;
                Remove(pBlock);
            }
            // We now have first empty block.
            else
            {
                m_HasEmptyBlock = true;
            }
        }
        // pBlock didn't become empty, but we have another empty block - find and free that one.
        // (This is optional, heuristics.)
        else if(m_HasEmptyBlock && blockCount > m_MinBlockCount)
        {
            NormalBlock* pLastBlock = m_Blocks.back();
            if(pLastBlock->m_pMetadata->IsEmpty())
            {
                pBlockToDelete = pLastBlock;
                m_Blocks.pop_back();
                m_HasEmptyBlock = false;
            }
        }

        IncrementallySortBlocks();
    }

    // Destruction of a free Allocation. Deferred until this point, outside of mutex
    // lock, for performance reason.
    if(pBlockToDelete != NULL)
    {
        D3D12MA_DELETE(m_hAllocator->GetAllocs(), pBlockToDelete);
    }
}

HRESULT BlockVector::CreateResource(
    UINT64 size,
    UINT64 alignment,
    const ALLOCATION_DESC& allocDesc,
    const D3D12_RESOURCE_DESC& resourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    Allocation** ppAllocation,
    REFIID riidResource,
    void** ppvResource)
{
    HRESULT hr = Allocate(size, alignment, allocDesc, 1, ppAllocation);
    if(SUCCEEDED(hr))
    {
        ID3D12Resource* res = NULL;
        hr = m_hAllocator->GetDevice()->CreatePlacedResource(
            (*ppAllocation)->m_Placed.block->GetHeap(),
            (*ppAllocation)->GetOffset(),
            &resourceDesc,
            InitialResourceState,
            pOptimizedClearValue,
            D3D12MA_IID_PPV_ARGS(&res));
        if(SUCCEEDED(hr))
        {
            if(ppvResource != NULL)
            {
                hr = res->QueryInterface(riidResource, ppvResource);
            }
            if(SUCCEEDED(hr))
            {
                (*ppAllocation)->SetResource(res, &resourceDesc);
            }
            else
            {
                res->Release();
                SAFE_RELEASE(*ppAllocation);
            }
        }
        else
        {
            SAFE_RELEASE(*ppAllocation);
        }
    }
    return hr;
}

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
HRESULT BlockVector::CreateResource2(
    UINT64 size,
    UINT64 alignment,
    const ALLOCATION_DESC& allocDesc,
    const D3D12_RESOURCE_DESC1& resourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    Allocation** ppAllocation,
    REFIID riidResource,
    void** ppvResource)
{
    ID3D12Device8* const device8 = m_hAllocator->GetDevice8();
    if(device8 == NULL)
    {
        return E_NOINTERFACE;
    }

    HRESULT hr = Allocate(size, alignment, allocDesc, 1, ppAllocation);
    if(SUCCEEDED(hr))
    {
        ID3D12Resource* res = NULL;
        hr = device8->CreatePlacedResource1(
            (*ppAllocation)->m_Placed.block->GetHeap(),
            (*ppAllocation)->GetOffset(),
            &resourceDesc,
            InitialResourceState,
            pOptimizedClearValue,
            D3D12MA_IID_PPV_ARGS(&res));
        if(SUCCEEDED(hr))
        {
            if(ppvResource != NULL)
            {
                hr = res->QueryInterface(riidResource, ppvResource);
            }
            if(SUCCEEDED(hr))
            {
                (*ppAllocation)->SetResource(res, &resourceDesc);
            }
            else
            {
                res->Release();
                SAFE_RELEASE(*ppAllocation);
            }
        }
        else
        {
            SAFE_RELEASE(*ppAllocation);
        }
    }
    return hr;
}
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

UINT64 BlockVector::CalcSumBlockSize() const
{
    UINT64 result = 0;
    for(size_t i = m_Blocks.size(); i--; )
    {
        result += m_Blocks[i]->m_pMetadata->GetSize();
    }
    return result;
}

UINT64 BlockVector::CalcMaxBlockSize() const
{
    UINT64 result = 0;
    for(size_t i = m_Blocks.size(); i--; )
    {
        result = D3D12MA_MAX(result, m_Blocks[i]->m_pMetadata->GetSize());
        if(result >= m_PreferredBlockSize)
        {
            break;
        }
    }
    return result;
}

void BlockVector::Remove(NormalBlock* pBlock)
{
    for(size_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
    {
        if(m_Blocks[blockIndex] == pBlock)
        {
            m_Blocks.remove(blockIndex);
            return;
        }
    }
    D3D12MA_ASSERT(0);
}

void BlockVector::IncrementallySortBlocks()
{
    // Bubble sort only until first swap.
    for(size_t i = 1; i < m_Blocks.size(); ++i)
    {
        if(m_Blocks[i - 1]->m_pMetadata->GetSumFreeSize() > m_Blocks[i]->m_pMetadata->GetSumFreeSize())
        {
            D3D12MA_SWAP(m_Blocks[i - 1], m_Blocks[i]);
            return;
        }
    }
}

HRESULT BlockVector::AllocateFromBlock(
    NormalBlock* pBlock,
    UINT64 size,
    UINT64 alignment,
    ALLOCATION_FLAGS allocFlags,
    Allocation** pAllocation)
{
    alignment = D3D12MA_MAX(alignment, m_MinAllocationAlignment);

    AllocationRequest currRequest = {};
    if(pBlock->m_pMetadata->CreateAllocationRequest(
        size,
        alignment,
        allocFlags & ALLOCATION_FLAG_UPPER_ADDRESS,
        &currRequest))
    {
        // We no longer have an empty Allocation.
        if(pBlock->m_pMetadata->IsEmpty())
        {
            m_HasEmptyBlock = false;
        }

        *pAllocation = m_hAllocator->GetAllocationObjectAllocator().Allocate(m_hAllocator, size, currRequest.zeroInitialized);
        pBlock->m_pMetadata->Alloc(currRequest, size, *pAllocation);
        (*pAllocation)->InitPlaced(currRequest.allocHandle, alignment, pBlock);
        D3D12MA_HEAVY_ASSERT(pBlock->Validate());
        m_hAllocator->m_Budget.AddAllocation(HeapTypeToIndex(m_HeapProps.Type), size);
        return S_OK;
    }
    return E_OUTOFMEMORY;
}

HRESULT BlockVector::CreateBlock(
    UINT64 blockSize,
    size_t* pNewBlockIndex)
{
    NormalBlock* const pBlock = D3D12MA_NEW(m_hAllocator->GetAllocs(), NormalBlock)(
        m_hAllocator,
        this,
        m_HeapProps,
        m_HeapFlags,
        blockSize,
        m_NextBlockId++);
    HRESULT hr = pBlock->Init(m_Algorithm, m_ProtectedSession);
    if(FAILED(hr))
    {
        D3D12MA_DELETE(m_hAllocator->GetAllocs(), pBlock);
        return hr;
    }

    m_Blocks.push_back(pBlock);
    if(pNewBlockIndex != NULL)
    {
        *pNewBlockIndex = m_Blocks.size() - 1;
    }

    return hr;
}

void BlockVector::AddStats(StatInfo& outStats)
{
    MutexLockRead lock(m_Mutex, m_hAllocator->UseMutex());

    for(size_t i = 0; i < m_Blocks.size(); ++i)
    {
        const NormalBlock* const pBlock = m_Blocks[i];
        D3D12MA_ASSERT(pBlock);
        D3D12MA_HEAVY_ASSERT(pBlock->Validate());
        StatInfo blockStatInfo;
        pBlock->m_pMetadata->CalcAllocationStatInfo(blockStatInfo);
        AddStatInfo(outStats, blockStatInfo);
    }
}

void BlockVector::AddStats(Stats& outStats)
{
    const UINT heapTypeIndex = HeapTypeToIndex(m_HeapProps.Type);
    StatInfo* const pStatInfo = &outStats.HeapType[heapTypeIndex];

    MutexLockRead lock(m_Mutex, m_hAllocator->UseMutex());

    for(size_t i = 0; i < m_Blocks.size(); ++i)
    {
        const NormalBlock* const pBlock = m_Blocks[i];
        D3D12MA_ASSERT(pBlock);
        D3D12MA_HEAVY_ASSERT(pBlock->Validate());
        StatInfo blockStatInfo;
        pBlock->m_pMetadata->CalcAllocationStatInfo(blockStatInfo);
        AddStatInfo(outStats.Total, blockStatInfo);
        AddStatInfo(*pStatInfo, blockStatInfo);
    }
}

void BlockVector::WriteBlockInfoToJson(JsonWriter& json)
{
    MutexLockRead lock(m_Mutex, m_hAllocator->UseMutex());

    json.BeginObject();

    for (size_t i = 0, count = m_Blocks.size(); i < count; ++i)
    {
        const NormalBlock* const pBlock = m_Blocks[i];
        D3D12MA_ASSERT(pBlock);
        D3D12MA_HEAVY_ASSERT(pBlock->Validate());
        json.BeginString();
        json.ContinueString(pBlock->GetId());
        json.EndString();

        pBlock->m_pMetadata->WriteAllocationInfoToJson(json);
    }

    json.EndObject();
}

////////////////////////////////////////////////////////////////////////////////
// Private class PoolPimpl

PoolPimpl::PoolPimpl(AllocatorPimpl* allocator, const POOL_DESC& desc) :
    m_Allocator(allocator),
    m_Desc(desc),
    m_BlockVector(NULL),
    m_Name(NULL)
{
    const bool explicitBlockSize = desc.BlockSize != 0;
    const UINT64 preferredBlockSize = explicitBlockSize ? desc.BlockSize : D3D12MA_DEFAULT_BLOCK_SIZE;
    UINT maxBlockCount = desc.MaxBlockCount != 0 ? desc.MaxBlockCount : UINT_MAX;

#ifndef __ID3D12Device4_INTERFACE_DEFINED__
    D3D12MA_ASSERT(m_Desc.pProtectedSession == NULL);
#endif

    m_BlockVector = D3D12MA_NEW(allocator->GetAllocs(), BlockVector)(
        allocator, desc.HeapProperties, desc.HeapFlags,
        preferredBlockSize,
        desc.MinBlockCount, maxBlockCount,
        explicitBlockSize,
        D3D12MA_MAX(desc.MinAllocationAlignment, (UINT64)D3D12MA_DEBUG_ALIGNMENT),
        desc.Flags & POOL_FLAG_ALGORITHM_MASK,
        desc.pProtectedSession);
}

HRESULT PoolPimpl::Init()
{
    m_CommittedAllocations.Init(m_Allocator->UseMutex(), m_Desc.HeapProperties.Type, this);
    return m_BlockVector->CreateMinBlocks();
}

PoolPimpl::~PoolPimpl()
{
    D3D12MA_ASSERT(m_PrevPool == NULL && m_NextPool == NULL);
    FreeName();
    D3D12MA_DELETE(m_Allocator->GetAllocs(), m_BlockVector);
}

void PoolPimpl::CalculateStats(StatInfo& outStats)
{
    ZeroMemory(&outStats, sizeof(outStats));
    outStats.AllocationSizeMin = UINT64_MAX;
    outStats.UnusedRangeSizeMin = UINT64_MAX;

    m_BlockVector->AddStats(outStats);

    {
        StatInfo committedStatInfo; // Uninitialized.
        m_CommittedAllocations.CalculateStats(committedStatInfo);
        AddStatInfo(outStats, committedStatInfo);
    }

    PostProcessStatInfo(outStats);
}

void PoolPimpl::AddStats(Stats& inoutStats)
{
    StatInfo poolStatInfo = {};
    CalculateStats(poolStatInfo);

    AddStatInfo(inoutStats.Total, poolStatInfo);
    AddStatInfo(inoutStats.HeapType[HeapTypeToIndex(m_Desc.HeapProperties.Type)], poolStatInfo);
}

void PoolPimpl::SetName(LPCWSTR Name)
{
    FreeName();

    if(Name)
    {
        const size_t nameCharCount = wcslen(Name) + 1;
        m_Name = D3D12MA_NEW_ARRAY(m_Allocator->GetAllocs(), WCHAR, nameCharCount);
        memcpy(m_Name, Name, nameCharCount * sizeof(WCHAR));
    }
}

void PoolPimpl::FreeName()
{
    if(m_Name)
    {
        const size_t nameCharCount = wcslen(m_Name) + 1;
        D3D12MA_DELETE_ARRAY(m_Allocator->GetAllocs(), m_Name, nameCharCount);
        m_Name = NULL;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Public class Pool implementation

void Pool::ReleaseThis()
{
    if(this == NULL)
    {
        return;
    }

    D3D12MA_DELETE(m_Pimpl->GetAllocator()->GetAllocs(), this);
}

POOL_DESC Pool::GetDesc() const
{
    return m_Pimpl->GetDesc();
}

void Pool::CalculateStats(StatInfo* pStats)
{
    D3D12MA_ASSERT(pStats);
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    m_Pimpl->CalculateStats(*pStats);
}

void Pool::SetName(LPCWSTR Name)
{
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    m_Pimpl->SetName(Name);
}

LPCWSTR Pool::GetName() const
{
    return m_Pimpl->GetName();
}

Pool::Pool(Allocator* allocator, const POOL_DESC &desc) :
    m_Pimpl(D3D12MA_NEW(allocator->m_Pimpl->GetAllocs(), PoolPimpl)(allocator->m_Pimpl, desc))
{
}

Pool::~Pool()
{
    m_Pimpl->GetAllocator()->UnregisterPool(this, m_Pimpl->GetDesc().HeapProperties.Type);

    D3D12MA_DELETE(m_Pimpl->GetAllocator()->GetAllocs(), m_Pimpl);
}

////////////////////////////////////////////////////////////////////////////////
// Private class AllocatorPimpl implementation

AllocatorPimpl::AllocatorPimpl(const ALLOCATION_CALLBACKS& allocationCallbacks, const ALLOCATOR_DESC& desc) :
    m_UseMutex((desc.Flags & ALLOCATOR_FLAG_SINGLETHREADED) == 0),
    m_AlwaysCommitted((desc.Flags & ALLOCATOR_FLAG_ALWAYS_COMMITTED) != 0),
    m_Device(desc.pDevice),
    m_Adapter(desc.pAdapter),
    m_PreferredBlockSize(desc.PreferredBlockSize != 0 ? desc.PreferredBlockSize : D3D12MA_DEFAULT_BLOCK_SIZE),
    m_AllocationCallbacks(allocationCallbacks),
    m_CurrentFrameIndex(0),
    // Below this line don't use allocationCallbacks but m_AllocationCallbacks!!!
    m_AllocationObjectAllocator(m_AllocationCallbacks)
{
    // desc.pAllocationCallbacks intentionally ignored here, preprocessed by CreateAllocator.
    ZeroMemory(&m_D3D12Options, sizeof(m_D3D12Options));
    ZeroMemory(&m_D3D12Architecture, sizeof(m_D3D12Architecture));

    ZeroMemory(m_BlockVectors, sizeof(m_BlockVectors));

    for(UINT i = 0; i < STANDARD_HEAP_TYPE_COUNT; ++i)
    {
        m_CommittedAllocations[i].Init(
            m_UseMutex,
            (D3D12_HEAP_TYPE)(D3D12_HEAP_TYPE_DEFAULT + i),
            NULL); // pool
    }

    m_Device->AddRef();
    m_Adapter->AddRef();
}

HRESULT AllocatorPimpl::Init(const ALLOCATOR_DESC& desc)
{
#if D3D12MA_DXGI_1_4
    desc.pAdapter->QueryInterface(D3D12MA_IID_PPV_ARGS(&m_Adapter3));
#endif

#ifdef __ID3D12Device4_INTERFACE_DEFINED__
    m_Device->QueryInterface(D3D12MA_IID_PPV_ARGS(&m_Device4));
#endif

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
    m_Device->QueryInterface(D3D12MA_IID_PPV_ARGS(&m_Device8));
#endif

    HRESULT hr = m_Adapter->GetDesc(&m_AdapterDesc);
    if(FAILED(hr))
    {
        return hr;
    }

    hr = m_Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &m_D3D12Options, sizeof(m_D3D12Options));
    if(FAILED(hr))
    {
        return hr;
    }
#ifdef D3D12MA_FORCE_RESOURCE_HEAP_TIER
    m_D3D12Options.ResourceHeapTier = (D3D12MA_FORCE_RESOURCE_HEAP_TIER);
#endif

    hr = m_Device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &m_D3D12Architecture, sizeof(m_D3D12Architecture));
    if(FAILED(hr))
    {
        m_D3D12Architecture.UMA = FALSE;
        m_D3D12Architecture.CacheCoherentUMA = FALSE;
    }

    D3D12_HEAP_PROPERTIES heapProps = {};
    const UINT defaultPoolCount = CalcDefaultPoolCount();
    for(UINT i = 0; i < defaultPoolCount; ++i)
    {
        D3D12_HEAP_FLAGS heapFlags;
        CalcDefaultPoolParams(heapProps.Type, heapFlags, i);

        m_BlockVectors[i] = D3D12MA_NEW(GetAllocs(), BlockVector)(
            this, // hAllocator
            heapProps, // heapType
            heapFlags, // heapFlags
            m_PreferredBlockSize,
            0, // minBlockCount
            SIZE_MAX, // maxBlockCount
            false, // explicitBlockSize
            D3D12MA_DEBUG_ALIGNMENT, // minAllocationAlignment
            0, // Default algorithm
            NULL); // pProtectedSession
        // No need to call m_pBlockVectors[i]->CreateMinBlocks here, becase minBlockCount is 0.
    }

#if D3D12MA_DXGI_1_4
    if(m_Adapter3)
    {
        UpdateD3D12Budget();
    }
#endif

    return S_OK;
}

AllocatorPimpl::~AllocatorPimpl()
{
#ifdef __ID3D12Device8_INTERFACE_DEFINED__
    SAFE_RELEASE(m_Device8);
#endif
#ifdef __ID3D12Device4_INTERFACE_DEFINED__
    SAFE_RELEASE(m_Device4);
#endif
#if D3D12MA_DXGI_1_4
    SAFE_RELEASE(m_Adapter3);
#endif
    SAFE_RELEASE(m_Adapter);
    SAFE_RELEASE(m_Device);

    for(UINT i = DEFAULT_POOL_MAX_COUNT; i--; )
    {
        D3D12MA_DELETE(GetAllocs(), m_BlockVectors[i]);
    }

    for(UINT i = HEAP_TYPE_COUNT; i--; )
    {
        if(!m_Pools[i].IsEmpty())
        {
            D3D12MA_ASSERT(0 && "Unfreed pools found!");
        }
    }
}

bool AllocatorPimpl::HeapFlagsFulfillResourceHeapTier(D3D12_HEAP_FLAGS flags) const
{
    if(SupportsResourceHeapTier2())
    {
        return true;
    }
    else
    {
        const bool allowBuffers         = (flags & D3D12_HEAP_FLAG_DENY_BUFFERS           ) == 0;
        const bool allowRtDsTextures    = (flags & D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES    ) == 0;
        const bool allowNonRtDsTextures = (flags & D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES) == 0;
        const uint8_t allowedGroupCount = (allowBuffers ? 1 : 0) + (allowRtDsTextures ? 1 : 0) + (allowNonRtDsTextures ? 1 : 0);
        return allowedGroupCount == 1;
    }
}

HRESULT AllocatorPimpl::CreateResource(
    const ALLOCATION_DESC* pAllocDesc,
    const D3D12_RESOURCE_DESC* pResourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    Allocation** ppAllocation,
    REFIID riidResource,
    void** ppvResource)
{
    D3D12MA_ASSERT(pAllocDesc && pResourceDesc && ppAllocation);

    *ppAllocation = NULL;
    if(ppvResource)
    {
        *ppvResource = NULL;
    }

    D3D12_RESOURCE_DESC finalResourceDesc = *pResourceDesc;
    D3D12_RESOURCE_ALLOCATION_INFO resAllocInfo = GetResourceAllocationInfo(finalResourceDesc);
    D3D12MA_ASSERT(IsPow2(resAllocInfo.Alignment));
    D3D12MA_ASSERT(resAllocInfo.SizeInBytes > 0);

    BlockVector* blockVector = NULL;
    CommittedAllocationParameters committedAllocationParams = {};
    bool preferCommitted = false;
    HRESULT hr = CalcAllocationParams<D3D12_RESOURCE_DESC>(*pAllocDesc, resAllocInfo.SizeInBytes,
        pResourceDesc,
        blockVector, committedAllocationParams, preferCommitted);
    if(FAILED(hr))
        return hr;

    const bool withinBudget = (pAllocDesc->Flags & ALLOCATION_FLAG_WITHIN_BUDGET) != 0;
    hr = E_INVALIDARG;
    if(committedAllocationParams.IsValid() && preferCommitted)
    {
        hr = AllocateCommittedResource(committedAllocationParams,
            resAllocInfo.SizeInBytes, withinBudget, &finalResourceDesc,
            InitialResourceState, pOptimizedClearValue,
            ppAllocation, riidResource, ppvResource);
        if(SUCCEEDED(hr))
            return hr;
    }
    if(blockVector != NULL)
    {
        hr = blockVector->CreateResource(resAllocInfo.SizeInBytes, resAllocInfo.Alignment,
            *pAllocDesc, finalResourceDesc,
            InitialResourceState, pOptimizedClearValue,
            ppAllocation, riidResource, ppvResource);
        if(SUCCEEDED(hr))
            return hr;
    }
    if(committedAllocationParams.IsValid() && !preferCommitted)
    {
        hr = AllocateCommittedResource(committedAllocationParams,
            resAllocInfo.SizeInBytes, withinBudget, &finalResourceDesc,
            InitialResourceState, pOptimizedClearValue,
            ppAllocation, riidResource, ppvResource);
        if(SUCCEEDED(hr))
            return hr;
    }
    return hr;
}

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
HRESULT AllocatorPimpl::CreateResource2(
    const ALLOCATION_DESC* pAllocDesc,
    const D3D12_RESOURCE_DESC1* pResourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    Allocation** ppAllocation,
    REFIID riidResource,
    void** ppvResource)
{
    D3D12MA_ASSERT(pAllocDesc && pResourceDesc && ppAllocation);

    *ppAllocation = NULL;
    if(ppvResource)
    {
        *ppvResource = NULL;
    }
    if(m_Device8 == NULL)
    {
        return E_NOINTERFACE;
    }

    D3D12_RESOURCE_DESC1 finalResourceDesc = *pResourceDesc;
    D3D12_RESOURCE_ALLOCATION_INFO resAllocInfo = GetResourceAllocationInfo(finalResourceDesc);
    D3D12MA_ASSERT(IsPow2(resAllocInfo.Alignment));
    D3D12MA_ASSERT(resAllocInfo.SizeInBytes > 0);

    BlockVector* blockVector = NULL;
    CommittedAllocationParameters committedAllocationParams = {};
    bool preferCommitted = false;
    HRESULT hr = CalcAllocationParams<D3D12_RESOURCE_DESC1>(*pAllocDesc, resAllocInfo.SizeInBytes,
        pResourceDesc,
        blockVector, committedAllocationParams, preferCommitted);
    if(FAILED(hr))
        return hr;
    
    const bool withinBudget = (pAllocDesc->Flags & ALLOCATION_FLAG_WITHIN_BUDGET) != 0;
    hr = E_INVALIDARG;
    if(committedAllocationParams.IsValid() && preferCommitted)
    {
        hr = AllocateCommittedResource2(committedAllocationParams,
            resAllocInfo.SizeInBytes, withinBudget,&finalResourceDesc,
            InitialResourceState, pOptimizedClearValue,
            ppAllocation, riidResource, ppvResource);
        if(SUCCEEDED(hr))
            return hr;
    }
    if(blockVector != NULL)
    {
        hr = blockVector->CreateResource2(resAllocInfo.SizeInBytes, resAllocInfo.Alignment,
            *pAllocDesc, finalResourceDesc,
            InitialResourceState, pOptimizedClearValue,
            ppAllocation, riidResource, ppvResource);
        if(SUCCEEDED(hr))
            return hr;
    }
    if(committedAllocationParams.IsValid() && !preferCommitted)
    {
        hr = AllocateCommittedResource2(committedAllocationParams,
            resAllocInfo.SizeInBytes, withinBudget,&finalResourceDesc,
            InitialResourceState, pOptimizedClearValue,
            ppAllocation, riidResource, ppvResource);
        if(SUCCEEDED(hr))
            return hr;
    }
    return hr;
}
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

HRESULT AllocatorPimpl::AllocateMemory(
    const ALLOCATION_DESC* pAllocDesc,
    const D3D12_RESOURCE_ALLOCATION_INFO* pAllocInfo,
    Allocation** ppAllocation)
{
    *ppAllocation = NULL;

    BlockVector* blockVector = NULL;
    CommittedAllocationParameters committedAllocationParams = {};
    bool preferCommitted = false;
    HRESULT hr = CalcAllocationParams<D3D12_RESOURCE_DESC>(*pAllocDesc, pAllocInfo->SizeInBytes,
        NULL, // pResDesc
        blockVector, committedAllocationParams, preferCommitted);
    if(FAILED(hr))
        return hr;
    
    const bool withinBudget = (pAllocDesc->Flags & ALLOCATION_FLAG_WITHIN_BUDGET) != 0;
    hr = E_INVALIDARG;
    if(committedAllocationParams.IsValid() && preferCommitted)
    {
        hr = AllocateHeap(committedAllocationParams, *pAllocInfo, withinBudget, ppAllocation);
        if(SUCCEEDED(hr))
            return hr;
    }
    if(blockVector != NULL)
    {
        hr = blockVector->Allocate(pAllocInfo->SizeInBytes, pAllocInfo->Alignment,
            *pAllocDesc, 1, (Allocation**)ppAllocation);
        if(SUCCEEDED(hr))
            return hr;
    }
    if(committedAllocationParams.IsValid() && !preferCommitted)
    {
        hr = AllocateHeap(committedAllocationParams, *pAllocInfo, withinBudget, ppAllocation);
        if(SUCCEEDED(hr))
            return hr;
    }
    return hr;
}

HRESULT AllocatorPimpl::CreateAliasingResource(
    Allocation* pAllocation,
    UINT64 AllocationLocalOffset,
    const D3D12_RESOURCE_DESC* pResourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    REFIID riidResource,
    void** ppvResource)
{
    *ppvResource = NULL;

    D3D12_RESOURCE_DESC resourceDesc2 = *pResourceDesc;
    D3D12_RESOURCE_ALLOCATION_INFO resAllocInfo = GetResourceAllocationInfo(resourceDesc2);
    D3D12MA_ASSERT(IsPow2(resAllocInfo.Alignment));
    D3D12MA_ASSERT(resAllocInfo.SizeInBytes > 0);

    ID3D12Heap* const existingHeap = pAllocation->GetHeap();
    const UINT64 existingOffset = pAllocation->GetOffset();
    const UINT64 existingSize = pAllocation->GetSize();
    const UINT64 newOffset = existingOffset + AllocationLocalOffset;

    if(existingHeap == NULL ||
        AllocationLocalOffset + resAllocInfo.SizeInBytes > existingSize ||
        newOffset % resAllocInfo.Alignment != 0)
    {
        return E_INVALIDARG;
    }

    return m_Device->CreatePlacedResource(
        existingHeap,
        newOffset,
        &resourceDesc2,
        InitialResourceState,
        pOptimizedClearValue,
        riidResource,
        ppvResource);
}

template<typename D3D12_RESOURCE_DESC_T>
bool AllocatorPimpl::PrefersCommittedAllocation(const D3D12_RESOURCE_DESC_T& resourceDesc)
{
    // Intentional. It may change in the future.
    return false;
}

HRESULT AllocatorPimpl::AllocateCommittedResource(
    const CommittedAllocationParameters& committedAllocParams,
    UINT64 resourceSize, bool withinBudget,
    const D3D12_RESOURCE_DESC* pResourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState, const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    Allocation** ppAllocation, REFIID riidResource, void** ppvResource)
{
    D3D12MA_ASSERT(committedAllocParams.IsValid());

    if(withinBudget &&
        !NewAllocationWithinBudget(committedAllocParams.m_HeapProperties.Type, resourceSize))
    {
        return E_OUTOFMEMORY;
    }

    ID3D12Resource* res = NULL;
    /* D3D12 ERROR:
     * ID3D12Device::CreateCommittedResource: 
     * When creating a committed resource, D3D12_HEAP_FLAGS must not have either
     *      D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
     *      D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES,
     *      nor D3D12_HEAP_FLAG_DENY_BUFFERS set.
     * These flags will be set automatically to correspond with the committed resource type.
     * 
     * [ STATE_CREATION ERROR #640: CREATERESOURCEANDHEAP_INVALIDHEAPMISCFLAGS]
    */
#ifdef __ID3D12Device4_INTERFACE_DEFINED__
    if(m_Device4 == NULL)
        return E_NOINTERFACE;

    HRESULT hr = m_Device4->CreateCommittedResource1(
        &committedAllocParams.m_HeapProperties,
        committedAllocParams.m_HeapFlags & ~RESOURCE_CLASS_HEAP_FLAGS,
        pResourceDesc, InitialResourceState,
        pOptimizedClearValue, committedAllocParams.m_ProtectedSession, D3D12MA_IID_PPV_ARGS(&res));
#else
    D3D12MA_ASSERT(committedAllocParams.m_ProtectedSession == NULL);
    HRESULT hr = m_Device->CreateCommittedResource(
        &committedAllocParams.m_HeapProperties,
        committedAllocParams.m_HeapFlags & ~RESOURCE_CLASS_HEAP_FLAGS, 
        pResourceDesc, InitialResourceState,
        pOptimizedClearValue, D3D12MA_IID_PPV_ARGS(&res));
#endif
    if(SUCCEEDED(hr))
    {
        if(ppvResource != NULL)
        {
            hr = res->QueryInterface(riidResource, ppvResource);
        }
        if(SUCCEEDED(hr))
        {
            const BOOL wasZeroInitialized = TRUE;
            Allocation* alloc = m_AllocationObjectAllocator.Allocate(this, resourceSize, wasZeroInitialized);
            alloc->InitCommitted(committedAllocParams.m_List);
            alloc->SetResource(res, pResourceDesc);

            *ppAllocation = alloc;

            committedAllocParams.m_List->Register(alloc);

            const UINT heapTypeIndex = HeapTypeToIndex(committedAllocParams.m_HeapProperties.Type);
            m_Budget.AddCommittedAllocation(heapTypeIndex, resourceSize);
        }
        else
        {
            res->Release();
        }
    }
    return hr;
}

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
HRESULT AllocatorPimpl::AllocateCommittedResource2(
    const CommittedAllocationParameters& committedAllocParams,
    UINT64 resourceSize, bool withinBudget,
    const D3D12_RESOURCE_DESC1* pResourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState, const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    Allocation** ppAllocation, REFIID riidResource, void** ppvResource)
{
    D3D12MA_ASSERT(committedAllocParams.IsValid());
    
    if(m_Device8 == NULL)
    {
        return E_NOINTERFACE;
    }

    if(withinBudget &&
        !NewAllocationWithinBudget(committedAllocParams.m_HeapProperties.Type, resourceSize))
    {
        return E_OUTOFMEMORY;
    }

    ID3D12Resource* res = NULL;
    HRESULT hr = m_Device8->CreateCommittedResource2(
        &committedAllocParams.m_HeapProperties,
        committedAllocParams.m_HeapFlags & ~RESOURCE_CLASS_HEAP_FLAGS, // D3D12 ERROR: ID3D12Device::CreateCommittedResource: When creating a committed resource, D3D12_HEAP_FLAGS must not have either D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES, D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES, nor D3D12_HEAP_FLAG_DENY_BUFFERS set. These flags will be set automatically to correspond with the committed resource type. [ STATE_CREATION ERROR #640: CREATERESOURCEANDHEAP_INVALIDHEAPMISCFLAGS]
        pResourceDesc, InitialResourceState,
        pOptimizedClearValue, committedAllocParams.m_ProtectedSession, D3D12MA_IID_PPV_ARGS(&res));
    if(SUCCEEDED(hr))
    {
        if(ppvResource != NULL)
        {
            hr = res->QueryInterface(riidResource, ppvResource);
        }
        if(SUCCEEDED(hr))
        {
            const BOOL wasZeroInitialized = TRUE;
            Allocation* alloc = m_AllocationObjectAllocator.Allocate(this, resourceSize, wasZeroInitialized);
            alloc->InitCommitted(committedAllocParams.m_List);
            alloc->SetResource(res, pResourceDesc);

            *ppAllocation = alloc;

            committedAllocParams.m_List->Register(alloc);

            const UINT heapTypeIndex = HeapTypeToIndex(committedAllocParams.m_HeapProperties.Type);
            m_Budget.AddCommittedAllocation(heapTypeIndex, resourceSize);
        }
        else
        {
            res->Release();
        }
    }
    return hr;
}
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

HRESULT AllocatorPimpl::AllocateHeap(
    const CommittedAllocationParameters& committedAllocParams,
    const D3D12_RESOURCE_ALLOCATION_INFO& allocInfo, bool withinBudget,
    Allocation** ppAllocation)
{
    D3D12MA_ASSERT(committedAllocParams.IsValid());

    *ppAllocation = nullptr;

    if(withinBudget &&
        !NewAllocationWithinBudget(committedAllocParams.m_HeapProperties.Type, allocInfo.SizeInBytes))
    {
        return E_OUTOFMEMORY;
    }

    D3D12_HEAP_DESC heapDesc = {};
    heapDesc.SizeInBytes = allocInfo.SizeInBytes;
    heapDesc.Properties = committedAllocParams.m_HeapProperties;
    heapDesc.Alignment = allocInfo.Alignment;
    heapDesc.Flags = committedAllocParams.m_HeapFlags;

    ID3D12Heap* heap = nullptr;
#ifdef __ID3D12Device4_INTERFACE_DEFINED__
    if(m_Device4 == NULL)
        return E_NOINTERFACE;

    HRESULT hr = m_Device4->CreateHeap1(&heapDesc, committedAllocParams.m_ProtectedSession, D3D12MA_IID_PPV_ARGS(&heap));
#else
    D3D12MA_ASSERT(committedAllocParams.m_ProtectedSession == NULL);
    HRESULT hr = m_Device->CreateHeap(&heapDesc, D3D12MA_IID_PPV_ARGS(&heap));
#endif
    if(SUCCEEDED(hr))
    {
        const BOOL wasZeroInitialized = TRUE;
        (*ppAllocation) = m_AllocationObjectAllocator.Allocate(this, allocInfo.SizeInBytes, wasZeroInitialized);
        (*ppAllocation)->InitHeap(committedAllocParams.m_List, heap);
        committedAllocParams.m_List->Register(*ppAllocation);

        const UINT heapTypeIndex = HeapTypeToIndex(committedAllocParams.m_HeapProperties.Type);
        m_Budget.AddCommittedAllocation(heapTypeIndex, allocInfo.SizeInBytes);
    }
    return hr;
}

template<typename D3D12_RESOURCE_DESC_T>
HRESULT AllocatorPimpl::CalcAllocationParams(const ALLOCATION_DESC& allocDesc, UINT64 allocSize,
    const D3D12_RESOURCE_DESC_T* resDesc,
    BlockVector*& outBlockVector, CommittedAllocationParameters& outCommittedAllocationParams, bool& outPreferCommitted)
{
    outBlockVector = NULL;
    outCommittedAllocationParams = CommittedAllocationParameters();
    outPreferCommitted = false;

    if(allocDesc.CustomPool != NULL)
    {
        PoolPimpl* const pool = allocDesc.CustomPool->m_Pimpl;

        outBlockVector = pool->GetBlockVector();

        outCommittedAllocationParams.m_ProtectedSession = pool->GetDesc().pProtectedSession;
        outCommittedAllocationParams.m_HeapProperties = pool->GetDesc().HeapProperties;
        outCommittedAllocationParams.m_HeapFlags = pool->GetDesc().HeapFlags;
        outCommittedAllocationParams.m_List = pool->GetCommittedAllocationList();
    }
    else
    {
        if(!IsHeapTypeStandard(allocDesc.HeapType))
        {
            return E_INVALIDARG;
        }

        outCommittedAllocationParams.m_HeapProperties = StandardHeapTypeToHeapProperties(allocDesc.HeapType);
        outCommittedAllocationParams.m_HeapFlags = allocDesc.ExtraHeapFlags;
        outCommittedAllocationParams.m_List = &m_CommittedAllocations[HeapTypeToIndex(allocDesc.HeapType)];
        
        const ResourceClass resourceClass = (resDesc != NULL) ?
            ResourceDescToResourceClass(*resDesc) : HeapFlagsToResourceClass(allocDesc.ExtraHeapFlags);
        const UINT defaultPoolIndex = CalcDefaultPoolIndex(allocDesc, resourceClass);
        if(defaultPoolIndex != UINT32_MAX)
        {
            outBlockVector = m_BlockVectors[defaultPoolIndex];
            const UINT64 preferredBlockSize = outBlockVector->GetPreferredBlockSize();
            if(allocSize > preferredBlockSize)
            {
                outBlockVector = NULL;
            }
            else if(allocSize > preferredBlockSize / 2)
            {
                // Heuristics: Allocate committed memory if requested size if greater than half of preferred block size.
                outPreferCommitted = true;
            }
        }
        
        const D3D12_HEAP_FLAGS extraHeapFlags = allocDesc.ExtraHeapFlags & ~RESOURCE_CLASS_HEAP_FLAGS;
        if(outBlockVector != NULL && extraHeapFlags != 0)
        {
            outBlockVector = NULL;
        }
    }

    if((allocDesc.Flags & ALLOCATION_FLAG_COMMITTED) != 0 ||
        m_AlwaysCommitted)
    {
        outBlockVector = NULL;
    }
    if((allocDesc.Flags & ALLOCATION_FLAG_NEVER_ALLOCATE) != 0)
    {
        outCommittedAllocationParams.m_List = NULL;
    }

    if(resDesc != NULL && !outPreferCommitted && PrefersCommittedAllocation(*resDesc))
    {
        outPreferCommitted = true;
    }

    return (outBlockVector != NULL || outCommittedAllocationParams.m_List != NULL) ? S_OK : E_INVALIDARG;
}

UINT AllocatorPimpl::CalcDefaultPoolCount() const
{
    if(SupportsResourceHeapTier2())
    {
        return 3;
    }
    else
    {
        return 9;
    }
}

UINT AllocatorPimpl::CalcDefaultPoolIndex(const ALLOCATION_DESC& allocDesc, ResourceClass resourceClass) const
{
    const D3D12_HEAP_FLAGS extraHeapFlags = allocDesc.ExtraHeapFlags & ~RESOURCE_CLASS_HEAP_FLAGS;
    if(extraHeapFlags != 0)
    {
        return UINT32_MAX;
    }

    UINT poolIndex = UINT_MAX;
    switch(allocDesc.HeapType)
    {
    case D3D12_HEAP_TYPE_DEFAULT:  poolIndex = 0; break;
    case D3D12_HEAP_TYPE_UPLOAD:   poolIndex = 1; break;
    case D3D12_HEAP_TYPE_READBACK: poolIndex = 2; break;
    default: D3D12MA_ASSERT(0);
    }

    if(SupportsResourceHeapTier2())
        return poolIndex;
    else
    {
        switch(resourceClass)
        {
        case ResourceClass::Buffer:
            return poolIndex * 3;
        case ResourceClass::Non_RT_DS_Texture:
            return poolIndex * 3 + 1;
        case ResourceClass::RT_DS_Texture:
            return poolIndex * 3 + 2;
        default:
            return UINT32_MAX;
        }
    }
}

void AllocatorPimpl::CalcDefaultPoolParams(D3D12_HEAP_TYPE& outHeapType, D3D12_HEAP_FLAGS& outHeapFlags, UINT index) const
{
    outHeapType = D3D12_HEAP_TYPE_DEFAULT;
    outHeapFlags = D3D12_HEAP_FLAG_NONE;

    if(!SupportsResourceHeapTier2())
    {
        switch(index % 3)
        {
        case 0:
            outHeapFlags = D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES;
            break;
        case 1:
            outHeapFlags = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES;
            break;
        case 2:
            outHeapFlags = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES;
            break;
        }

        index /= 3;
    }

    switch(index)
    {
    case 0:
        outHeapType = D3D12_HEAP_TYPE_DEFAULT;
        break;
    case 1:
        outHeapType = D3D12_HEAP_TYPE_UPLOAD;
        break;
    case 2:
        outHeapType = D3D12_HEAP_TYPE_READBACK;
        break;
    default:
        D3D12MA_ASSERT(0);
    }
}

void AllocatorPimpl::RegisterPool(Pool* pool, D3D12_HEAP_TYPE heapType)
{
    const UINT heapTypeIndex = HeapTypeToIndex(heapType);

    MutexLockWrite lock(m_PoolsMutex[heapTypeIndex], m_UseMutex);
    m_Pools[heapTypeIndex].PushBack(pool->m_Pimpl);
}

void AllocatorPimpl::UnregisterPool(Pool* pool, D3D12_HEAP_TYPE heapType)
{
    const UINT heapTypeIndex = HeapTypeToIndex(heapType);

    MutexLockWrite lock(m_PoolsMutex[heapTypeIndex], m_UseMutex);
    m_Pools[heapTypeIndex].Remove(pool->m_Pimpl);
}

void AllocatorPimpl::FreeCommittedMemory(Allocation* allocation)
{
    D3D12MA_ASSERT(allocation && allocation->m_PackedData.GetType() == Allocation::TYPE_COMMITTED);

    CommittedAllocationList* const allocList = allocation->m_Committed.list;
    allocList->Unregister(allocation);

    const UINT64 allocationSize = allocation->GetSize();
    const UINT heapTypeIndex = HeapTypeToIndex(allocList->GetHeapType());
    m_Budget.RemoveCommittedAllocation(heapTypeIndex, allocationSize);
}

void AllocatorPimpl::FreePlacedMemory(Allocation* allocation)
{
    D3D12MA_ASSERT(allocation && allocation->m_PackedData.GetType() == Allocation::TYPE_PLACED);

    NormalBlock* const block = allocation->m_Placed.block;
    D3D12MA_ASSERT(block);
    BlockVector* const blockVector = block->GetBlockVector();
    D3D12MA_ASSERT(blockVector);
    m_Budget.RemoveAllocation(HeapTypeToIndex(block->GetHeapProperties().Type), allocation->GetSize());
    blockVector->Free(allocation);
}

void AllocatorPimpl::FreeHeapMemory(Allocation* allocation)
{
    D3D12MA_ASSERT(allocation && allocation->m_PackedData.GetType() == Allocation::TYPE_HEAP);

    CommittedAllocationList* const allocList = allocation->m_Committed.list;
    allocList->Unregister(allocation);
    SAFE_RELEASE(allocation->m_Heap.heap);

    const UINT heapTypeIndex = HeapTypeToIndex(allocList->GetHeapType());
    const UINT64 allocationSize = allocation->GetSize();
    m_Budget.RemoveCommittedAllocation(heapTypeIndex, allocationSize);
}

void AllocatorPimpl::SetCurrentFrameIndex(UINT frameIndex)
{
    m_CurrentFrameIndex.store(frameIndex);

#if D3D12MA_DXGI_1_4
    if(m_Adapter3)
    {
        UpdateD3D12Budget();
    }
#endif
}

void AllocatorPimpl::CalculateStats(Stats& outStats)
{
    // Init stats
    ZeroMemory(&outStats, sizeof(outStats));
    outStats.Total.AllocationSizeMin = UINT64_MAX;
    outStats.Total.UnusedRangeSizeMin = UINT64_MAX;
    for(size_t i = 0; i < HEAP_TYPE_COUNT; i++)
    {
        outStats.HeapType[i].AllocationSizeMin = UINT64_MAX;
        outStats.HeapType[i].UnusedRangeSizeMin = UINT64_MAX;
    }

    // Process deafult pools.
    if(SupportsResourceHeapTier2())
    {
        for(size_t heapTypeIndex = 0; heapTypeIndex < STANDARD_HEAP_TYPE_COUNT; ++heapTypeIndex)
        {
            BlockVector* const pBlockVector = m_BlockVectors[heapTypeIndex];
            D3D12MA_ASSERT(pBlockVector);
            pBlockVector->AddStats(outStats);
        }
    }
    else
    {
        for(size_t heapTypeIndex = 0; heapTypeIndex < STANDARD_HEAP_TYPE_COUNT; ++heapTypeIndex)
        {
            for(size_t heapSubType = 0; heapSubType < 3; ++heapSubType)
            {
                BlockVector* const pBlockVector = m_BlockVectors[heapTypeIndex * 3 + heapSubType];
                D3D12MA_ASSERT(pBlockVector);
                pBlockVector->AddStats(outStats);
            }
        }
    }

    // Process custom pools
    for(size_t heapTypeIndex = 0; heapTypeIndex < HEAP_TYPE_COUNT; ++heapTypeIndex)
    {
        MutexLockRead lock(m_PoolsMutex[heapTypeIndex], m_UseMutex);
        PoolList& poolList = m_Pools[heapTypeIndex];
        for(PoolPimpl* pool = poolList.Front(); pool != NULL; pool = poolList.GetNext(pool))
        {
            pool->AddStats(outStats);
        }
    }

    // Process committed allocations.
    for(size_t heapTypeIndex = 0; heapTypeIndex < STANDARD_HEAP_TYPE_COUNT; ++heapTypeIndex)
    {
        StatInfo statInfo; // Uninitialized.
        m_CommittedAllocations[heapTypeIndex].CalculateStats(statInfo);
        AddStatInfo(outStats.Total, statInfo);
        AddStatInfo(outStats.HeapType[heapTypeIndex], statInfo);
    }

    // Post process
    PostProcessStatInfo(outStats.Total);
    for(size_t i = 0; i < HEAP_TYPE_COUNT; ++i)
        PostProcessStatInfo(outStats.HeapType[i]);
}

void AllocatorPimpl::GetBudget(Budget* outGpuBudget, Budget* outCpuBudget)
{
    if(outGpuBudget)
    {
        // Taking DEFAULT.
        outGpuBudget->BlockBytes = m_Budget.m_BlockBytes[0];
        outGpuBudget->AllocationBytes = m_Budget.m_AllocationBytes[0];
    }
    if(outCpuBudget)
    {
        // Taking UPLOAD + READBACK.
        outCpuBudget->BlockBytes = m_Budget.m_BlockBytes[1] + m_Budget.m_BlockBytes[2];
        outCpuBudget->AllocationBytes = m_Budget.m_AllocationBytes[1] + m_Budget.m_AllocationBytes[2];
    }
    // TODO: What to do with CUSTOM?

#if D3D12MA_DXGI_1_4
    if(m_Adapter3)
    {
        if(m_Budget.m_OperationsSinceBudgetFetch < 30)
        {
            MutexLockRead lockRead(m_Budget.m_BudgetMutex, m_UseMutex);
            if(outGpuBudget)
            {

                if(m_Budget.m_D3D12UsageLocal + outGpuBudget->BlockBytes > m_Budget.m_BlockBytesAtBudgetFetch[0])
                {
                    outGpuBudget->UsageBytes = m_Budget.m_D3D12UsageLocal +
                        outGpuBudget->BlockBytes - m_Budget.m_BlockBytesAtBudgetFetch[0];
                }
                else
                {
                    outGpuBudget->UsageBytes = 0;
                }
                outGpuBudget->BudgetBytes = m_Budget.m_D3D12BudgetLocal;
            }
            if(outCpuBudget)
            {
                if(m_Budget.m_D3D12UsageNonLocal + outCpuBudget->BlockBytes > m_Budget.m_BlockBytesAtBudgetFetch[1] + m_Budget.m_BlockBytesAtBudgetFetch[2])
                {
                    outCpuBudget->UsageBytes = m_Budget.m_D3D12UsageNonLocal +
                        outCpuBudget->BlockBytes - (m_Budget.m_BlockBytesAtBudgetFetch[1] + m_Budget.m_BlockBytesAtBudgetFetch[2]);
                }
                else
                {
                    outCpuBudget->UsageBytes = 0;
                }
                outCpuBudget->BudgetBytes = m_Budget.m_D3D12BudgetNonLocal;
            }
        }
        else
        {
            UpdateD3D12Budget(); // Outside of mutex lock
            GetBudget(outGpuBudget, outCpuBudget); // Recursion
        }
    }
    else
#endif
    {
        if(outGpuBudget)
        {
            const UINT64 gpuMemorySize = m_AdapterDesc.DedicatedVideoMemory + m_AdapterDesc.DedicatedSystemMemory; // TODO: Is this right?
            outGpuBudget->UsageBytes = outGpuBudget->BlockBytes;
            outGpuBudget->BudgetBytes = gpuMemorySize * 8 / 10; // 80% heuristics.
        }
        if(outCpuBudget)
        {
            const UINT64 cpuMemorySize = m_AdapterDesc.SharedSystemMemory; // TODO: Is this right?
            outCpuBudget->UsageBytes = outCpuBudget->BlockBytes;
            outCpuBudget->BudgetBytes = cpuMemorySize * 8 / 10; // 80% heuristics.
        }
    }
}

void AllocatorPimpl::GetBudgetForHeapType(Budget& outBudget, D3D12_HEAP_TYPE heapType)
{
    switch(heapType)
    {
    case D3D12_HEAP_TYPE_DEFAULT:
        GetBudget(&outBudget, NULL);
        break;
    case D3D12_HEAP_TYPE_UPLOAD:
    case D3D12_HEAP_TYPE_READBACK:
        GetBudget(NULL, &outBudget);
        break;
    default: D3D12MA_ASSERT(0);
    }
}

static void AddStatInfoToJson(JsonWriter& json, const StatInfo& statInfo)
{
    json.BeginObject();
    json.WriteString(L"Blocks");
    json.WriteNumber(statInfo.BlockCount);
    json.WriteString(L"Allocations");
    json.WriteNumber(statInfo.AllocationCount);
    json.WriteString(L"UnusedRanges");
    json.WriteNumber(statInfo.UnusedRangeCount);
    json.WriteString(L"UsedBytes");
    json.WriteNumber(statInfo.UsedBytes);
    json.WriteString(L"UnusedBytes");
    json.WriteNumber(statInfo.UnusedBytes);

    json.WriteString(L"AllocationSize");
    json.BeginObject(true);
    json.WriteString(L"Min");
    json.WriteNumber(statInfo.AllocationSizeMin);
    json.WriteString(L"Avg");
    json.WriteNumber(statInfo.AllocationSizeAvg);
    json.WriteString(L"Max");
    json.WriteNumber(statInfo.AllocationSizeMax);
    json.EndObject();

    json.WriteString(L"UnusedRangeSize");
    json.BeginObject(true);
    json.WriteString(L"Min");
    json.WriteNumber(statInfo.UnusedRangeSizeMin);
    json.WriteString(L"Avg");
    json.WriteNumber(statInfo.UnusedRangeSizeAvg);
    json.WriteString(L"Max");
    json.WriteNumber(statInfo.UnusedRangeSizeMax);
    json.EndObject();

    json.EndObject();
}

void AllocatorPimpl::BuildStatsString(WCHAR** ppStatsString, BOOL DetailedMap)
{
    StringBuilder sb(GetAllocs());
    {
        JsonWriter json(GetAllocs(), sb);

        Budget gpuBudget = {}, cpuBudget = {};
        GetBudget(&gpuBudget, &cpuBudget);

        Stats stats;
        CalculateStats(stats);

        json.BeginObject();
        
        json.WriteString(L"Total");
        AddStatInfoToJson(json, stats.Total);
        for (size_t heapType = 0; heapType < HEAP_TYPE_COUNT; ++heapType)
        {
            json.WriteString(HeapTypeNames[heapType]);
            AddStatInfoToJson(json, stats.HeapType[heapType]);
        }

        json.WriteString(L"Budget");
        json.BeginObject();
        {
            json.WriteString(L"GPU");
            WriteBudgetToJson(json, gpuBudget);
            json.WriteString(L"CPU");
            WriteBudgetToJson(json, cpuBudget);
        }
        json.EndObject();

        if (DetailedMap)
        {
            json.WriteString(L"DetailedMap");
            json.BeginObject();

            json.WriteString(L"DefaultPools");
            json.BeginObject();

            if (SupportsResourceHeapTier2())
            {
                for (size_t heapType = 0; heapType < STANDARD_HEAP_TYPE_COUNT; ++heapType)
                {
                    json.WriteString(HeapTypeNames[heapType]);
                    json.BeginObject();

                    json.WriteString(L"Blocks");

                    BlockVector* blockVector = m_BlockVectors[heapType];
                    D3D12MA_ASSERT(blockVector);
                    blockVector->WriteBlockInfoToJson(json);

                    json.EndObject(); // heap name
                }
            }
            else
            {
                for (size_t heapType = 0; heapType < STANDARD_HEAP_TYPE_COUNT; ++heapType)
                {
                    for (size_t heapSubType = 0; heapSubType < 3; ++heapSubType)
                    {
                        static const WCHAR* const heapSubTypeName[] = {
                            L" + buffer",
                            L" + texture",
                            L" + texture RT or DS",
                        };
                        json.BeginString();
                        json.ContinueString(HeapTypeNames[heapType]);
                        json.ContinueString(heapSubTypeName[heapSubType]);
                        json.EndString();
                        json.BeginObject();

                        json.WriteString(L"Blocks");

                        BlockVector* blockVector = m_BlockVectors[heapType * 3 + heapSubType];
                        D3D12MA_ASSERT(blockVector);
                        blockVector->WriteBlockInfoToJson(json);

                        json.EndObject(); // heap name
                    }
                }
            }

            json.EndObject(); // DefaultPools

            json.WriteString(L"CommittedAllocations");
            json.BeginObject();

            for (size_t heapTypeIndex = 0; heapTypeIndex < STANDARD_HEAP_TYPE_COUNT; ++heapTypeIndex)
            {
                json.WriteString(HeapTypeNames[heapTypeIndex]);
                m_CommittedAllocations[heapTypeIndex].BuildStatsString(json);
            }

            json.EndObject(); // CommittedAllocations

            json.WriteString(L"Pools");
            json.BeginObject();

            for (size_t heapTypeIndex = 0; heapTypeIndex < HEAP_TYPE_COUNT; ++heapTypeIndex)
            {
                json.WriteString(HeapTypeNames[heapTypeIndex]);
                json.BeginArray();
                MutexLockRead mutex(m_PoolsMutex[heapTypeIndex], m_UseMutex);
                size_t index = 0;
                for (auto* item = m_Pools[heapTypeIndex].Front(); item != nullptr; item = PoolList::GetNext(item))
                {
                    json.BeginObject();
                    json.WriteString(L"Name");
                    if (item->GetName() != nullptr)
                    {
                        json.WriteString(item->GetName());
                    }
                    else
                    {
                        json.BeginString();
                        json.ContinueString(index);
                        json.EndString();
                    }
                    ++index;

                    json.WriteString(L"Blocks");
                    item->GetBlockVector()->WriteBlockInfoToJson(json);
                    json.WriteString(L"CommittedAllocations");
                    item->GetCommittedAllocationList()->BuildStatsString(json);

                    json.EndObject();
                }
                json.EndArray();
            }

            json.EndObject(); // Pools

            json.EndObject(); // DetailedMap
        }
        json.EndObject();
    }

    const size_t length = sb.GetLength();
    WCHAR* result = AllocateArray<WCHAR>(GetAllocs(), length + 1);
    memcpy(result, sb.GetData(), length * sizeof(WCHAR));
    result[length] = L'\0';
    *ppStatsString = result;
}

void AllocatorPimpl::FreeStatsString(WCHAR* pStatsString)
{
    D3D12MA_ASSERT(pStatsString);
    Free(GetAllocs(), pStatsString);
}

HRESULT AllocatorPimpl::UpdateD3D12Budget()
{
#if D3D12MA_DXGI_1_4
    D3D12MA_ASSERT(m_Adapter3);

    DXGI_QUERY_VIDEO_MEMORY_INFO infoLocal = {};
    DXGI_QUERY_VIDEO_MEMORY_INFO infoNonLocal = {};
    HRESULT hrLocal = m_Adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &infoLocal);
    HRESULT hrNonLocal = m_Adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &infoNonLocal);

    {
        MutexLockWrite lockWrite(m_Budget.m_BudgetMutex, m_UseMutex);

        if(SUCCEEDED(hrLocal))
        {
            m_Budget.m_D3D12UsageLocal = infoLocal.CurrentUsage;
            m_Budget.m_D3D12BudgetLocal = infoLocal.Budget;
        }
        if(SUCCEEDED(hrNonLocal))
        {
            m_Budget.m_D3D12UsageNonLocal = infoNonLocal.CurrentUsage;
            m_Budget.m_D3D12BudgetNonLocal = infoNonLocal.Budget;
        }

        for(UINT i = 0; i < HEAP_TYPE_COUNT; ++i)
        {
            m_Budget.m_BlockBytesAtBudgetFetch[i] = m_Budget.m_BlockBytes[i].load();
        }

        m_Budget.m_OperationsSinceBudgetFetch = 0;
    }

    return FAILED(hrLocal) ? hrLocal : hrNonLocal;
#else
    return S_OK;
#endif
}

D3D12_RESOURCE_ALLOCATION_INFO AllocatorPimpl::GetResourceAllocationInfoNative(const D3D12_RESOURCE_DESC& resourceDesc) const
{
    return m_Device->GetResourceAllocationInfo(0, 1, &resourceDesc);
}

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
D3D12_RESOURCE_ALLOCATION_INFO AllocatorPimpl::GetResourceAllocationInfoNative(const D3D12_RESOURCE_DESC1& resourceDesc) const
{
    D3D12MA_ASSERT(m_Device8 != NULL);
    D3D12_RESOURCE_ALLOCATION_INFO1 info1Unused;
    return m_Device8->GetResourceAllocationInfo2(0, 1, &resourceDesc, &info1Unused);
}
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

template<typename D3D12_RESOURCE_DESC_T>
D3D12_RESOURCE_ALLOCATION_INFO AllocatorPimpl::GetResourceAllocationInfo(D3D12_RESOURCE_DESC_T& inOutResourceDesc) const
{
    /* Optional optimization: Microsoft documentation says:
    https://docs.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-getresourceallocationinfo
    
    Your application can forgo using GetResourceAllocationInfo for buffer resources
    (D3D12_RESOURCE_DIMENSION_BUFFER). Buffers have the same size on all adapters,
    which is merely the smallest multiple of 64KB that's greater or equal to
    D3D12_RESOURCE_DESC::Width.
    */
    if(inOutResourceDesc.Alignment == 0 &&
        inOutResourceDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        return {
            AlignUp<UINT64>(inOutResourceDesc.Width, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT), // SizeInBytes
            D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT}; // Alignment
    }

#if D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT
    if(inOutResourceDesc.Alignment == 0 &&
        inOutResourceDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
        (inOutResourceDesc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) == 0
#if D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT == 1
        && CanUseSmallAlignment(inOutResourceDesc)
#endif
        )
    {
        /*
        The algorithm here is based on Microsoft sample: "Small Resources Sample"
        https://github.com/microsoft/DirectX-Graphics-Samples/tree/master/Samples/Desktop/D3D12SmallResources
        */
        const UINT64 smallAlignmentToTry = inOutResourceDesc.SampleDesc.Count > 1 ?
            D3D12_SMALL_MSAA_RESOURCE_PLACEMENT_ALIGNMENT :
            D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
        inOutResourceDesc.Alignment = smallAlignmentToTry;
        const D3D12_RESOURCE_ALLOCATION_INFO smallAllocInfo = GetResourceAllocationInfoNative(inOutResourceDesc);
        // Check if alignment requested has been granted.
        if(smallAllocInfo.Alignment == smallAlignmentToTry)
        {
            return smallAllocInfo;
        }
        inOutResourceDesc.Alignment = 0; // Restore original
    }
#endif // #if D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT

    return GetResourceAllocationInfoNative(inOutResourceDesc);
}

bool AllocatorPimpl::NewAllocationWithinBudget(D3D12_HEAP_TYPE heapType, UINT64 size)
{
    Budget budget = {};
    GetBudgetForHeapType(budget, heapType);
    return budget.UsageBytes + size <= budget.BudgetBytes;
}

void AllocatorPimpl::WriteBudgetToJson(JsonWriter& json, const Budget& budget)
{
    json.BeginObject();
    {
        json.WriteString(L"BlockBytes");
        json.WriteNumber(budget.BlockBytes);
        json.WriteString(L"AllocationBytes");
        json.WriteNumber(budget.AllocationBytes);
        json.WriteString(L"UsageBytes");
        json.WriteNumber(budget.UsageBytes);
        json.WriteString(L"BudgetBytes");
        json.WriteNumber(budget.BudgetBytes);
    }
    json.EndObject();
}

////////////////////////////////////////////////////////////////////////////////
// Public but internal class IUnknownImpl implementation

HRESULT STDMETHODCALLTYPE IUnknownImpl::QueryInterface(REFIID riid, void** ppvObject)
{
    if(ppvObject == NULL)
        return E_POINTER;
    if(riid == IID_IUnknown)
    {
        ++m_RefCount;
        *ppvObject = this;
        return S_OK;
    }
    *ppvObject = NULL;
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE IUnknownImpl::AddRef()
{
    return ++m_RefCount;
}

ULONG STDMETHODCALLTYPE IUnknownImpl::Release()
{
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK

    const uint32_t newRefCount = --m_RefCount;
    if(newRefCount == 0)
        ReleaseThis();
    return newRefCount;
}

////////////////////////////////////////////////////////////////////////////////
// Public class Allocation implementation

void Allocation::PackedData::SetType(Type type)
{
    const UINT u = (UINT)type;
    D3D12MA_ASSERT(u < (1u << 2));
    m_Type = u;
}

void Allocation::PackedData::SetResourceDimension(D3D12_RESOURCE_DIMENSION resourceDimension)
{
    const UINT u = (UINT)resourceDimension;
    D3D12MA_ASSERT(u < (1u << 3));
    m_ResourceDimension = u;
}

void Allocation::PackedData::SetResourceFlags(D3D12_RESOURCE_FLAGS resourceFlags)
{
    const UINT u = (UINT)resourceFlags;
    D3D12MA_ASSERT(u < (1u << 24));
    m_ResourceFlags = u;
}

void Allocation::PackedData::SetTextureLayout(D3D12_TEXTURE_LAYOUT textureLayout)
{
    const UINT u = (UINT)textureLayout;
    D3D12MA_ASSERT(u < (1u << 9));
    m_TextureLayout = u;
}

void Allocation::ReleaseThis()
{
    if(this == NULL)
    {
        return;
    }

    SAFE_RELEASE(m_Resource);

    switch(m_PackedData.GetType())
    {
    case TYPE_COMMITTED:
        m_Allocator->FreeCommittedMemory(this);
        break;
    case TYPE_PLACED:
        m_Allocator->FreePlacedMemory(this);
        break;
    case TYPE_HEAP:
        m_Allocator->FreeHeapMemory(this);
        break;
    }

    FreeName();

    m_Allocator->GetAllocationObjectAllocator().Free(this);
}

UINT64 Allocation::GetOffset() const
{
    switch(m_PackedData.GetType())
    {
    case TYPE_COMMITTED:
    case TYPE_HEAP:
        return 0;
    case TYPE_PLACED:
        return m_Placed.block->m_pMetadata->GetAllocationOffset(m_Placed.allocHandle);
    default:
        D3D12MA_ASSERT(0);
        return 0;
    }
}

ID3D12Heap* Allocation::GetHeap() const
{
    switch(m_PackedData.GetType())
    {
    case TYPE_COMMITTED:
        return NULL;
    case TYPE_PLACED:
        return m_Placed.block->GetHeap();
    case TYPE_HEAP:
        return m_Heap.heap;
    default:
        D3D12MA_ASSERT(0);
        return 0;
    }
}

void Allocation::SetName(LPCWSTR Name)
{
    FreeName();

    if(Name)
    {
        const size_t nameCharCount = wcslen(Name) + 1;
        m_Name = D3D12MA_NEW_ARRAY(m_Allocator->GetAllocs(), WCHAR, nameCharCount);
        memcpy(m_Name, Name, nameCharCount * sizeof(WCHAR));
    }
}

Allocation::Allocation(AllocatorPimpl* allocator, UINT64 size, BOOL wasZeroInitialized) :
    m_Allocator{allocator},
    m_Size{size},
    m_Resource{NULL},
    m_CreationFrameIndex{allocator->GetCurrentFrameIndex()},
    m_Name{NULL}
{
    D3D12MA_ASSERT(allocator);

    m_PackedData.SetType(TYPE_COUNT);
    m_PackedData.SetResourceDimension(D3D12_RESOURCE_DIMENSION_UNKNOWN);
    m_PackedData.SetResourceFlags(D3D12_RESOURCE_FLAG_NONE);
    m_PackedData.SetTextureLayout(D3D12_TEXTURE_LAYOUT_UNKNOWN);
    m_PackedData.SetWasZeroInitialized(wasZeroInitialized);
}

void Allocation::InitCommitted(CommittedAllocationList* list)
{
    m_PackedData.SetType(TYPE_COMMITTED);
    m_Committed.list = list;
    m_Committed.prev = NULL;
    m_Committed.next = NULL;
}

void Allocation::InitPlaced(AllocHandle allocHandle, UINT64 alignment, NormalBlock* block)
{
    m_PackedData.SetType(TYPE_PLACED);
    m_Placed.allocHandle = allocHandle;
    m_Placed.block = block;
}

void Allocation::InitHeap(CommittedAllocationList* list, ID3D12Heap* heap)
{
    m_PackedData.SetType(TYPE_HEAP);
    m_Heap.list = list;
    m_Committed.prev = NULL;
    m_Committed.next = NULL;
    m_Heap.heap = heap;
}

AllocHandle Allocation::GetAllocHandle() const
{
    switch (m_PackedData.GetType())
    {
    case TYPE_COMMITTED:
    case TYPE_HEAP:
        return (AllocHandle)0;
    case TYPE_PLACED:
        return m_Placed.allocHandle;
    default:
        D3D12MA_ASSERT(0);
        return (AllocHandle)0;
    }
}

template<typename D3D12_RESOURCE_DESC_T>
void Allocation::SetResource(ID3D12Resource* resource, const D3D12_RESOURCE_DESC_T* pResourceDesc)
{
    D3D12MA_ASSERT(m_Resource == NULL && pResourceDesc);
    m_Resource = resource;
    m_PackedData.SetResourceDimension(pResourceDesc->Dimension);
    m_PackedData.SetResourceFlags(pResourceDesc->Flags);
    m_PackedData.SetTextureLayout(pResourceDesc->Layout);
}

void Allocation::FreeName()
{
    if(m_Name)
    {
        const size_t nameCharCount = wcslen(m_Name) + 1;
        D3D12MA_DELETE_ARRAY(m_Allocator->GetAllocs(), m_Name, nameCharCount);
        m_Name = NULL;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Private class AllocationObjectAllocator implementation

AllocationObjectAllocator::AllocationObjectAllocator(const ALLOCATION_CALLBACKS& allocationCallbacks) :
    m_Allocator(allocationCallbacks, 1024)
{
}

template<typename... Types> Allocation* AllocationObjectAllocator::Allocate(Types... args)
{
    MutexLock mutexLock(m_Mutex);
    return m_Allocator.Alloc(std::forward<Types>(args)...);
}

void AllocationObjectAllocator::Free(Allocation* alloc)
{
    MutexLock mutexLock(m_Mutex);
    m_Allocator.Free(alloc);
}

////////////////////////////////////////////////////////////////////////////////
// Public class Allocator implementation

Allocator::Allocator(const ALLOCATION_CALLBACKS& allocationCallbacks, const ALLOCATOR_DESC& desc) :
    m_Pimpl(D3D12MA_NEW(allocationCallbacks, AllocatorPimpl)(allocationCallbacks, desc))
{
}

Allocator::~Allocator()
{
    D3D12MA_DELETE(m_Pimpl->GetAllocs(), m_Pimpl);
}

void Allocator::ReleaseThis()
{
    // Copy is needed because otherwise we would call destructor and invalidate the structure with callbacks before using it to free memory.
    const ALLOCATION_CALLBACKS allocationCallbacksCopy = m_Pimpl->GetAllocs();
    D3D12MA_DELETE(allocationCallbacksCopy, this);
}

const D3D12_FEATURE_DATA_D3D12_OPTIONS& Allocator::GetD3D12Options() const
{
    return m_Pimpl->GetD3D12Options();
}
BOOL Allocator::IsUMA() const
{
    return m_Pimpl->IsUMA();
}
BOOL Allocator::IsCacheCoherentUMA() const
{
    return m_Pimpl->IsCacheCoherentUMA();
}

HRESULT Allocator::CreateResource(
    const ALLOCATION_DESC* pAllocDesc,
    const D3D12_RESOURCE_DESC* pResourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    Allocation** ppAllocation,
    REFIID riidResource,
    void** ppvResource)
{
    if(!pAllocDesc || !pResourceDesc || !ppAllocation)
    {
        D3D12MA_ASSERT(0 && "Invalid arguments passed to Allocator::CreateResource.");
        return E_INVALIDARG;
    }
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    return m_Pimpl->CreateResource(pAllocDesc, pResourceDesc, InitialResourceState, pOptimizedClearValue, ppAllocation, riidResource, ppvResource);
}

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
HRESULT Allocator::CreateResource2(
    const ALLOCATION_DESC* pAllocDesc,
    const D3D12_RESOURCE_DESC1* pResourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    Allocation** ppAllocation,
    REFIID riidResource,
    void** ppvResource)
{
    if(!pAllocDesc || !pResourceDesc || !ppAllocation)
    {
        D3D12MA_ASSERT(0 && "Invalid arguments passed to Allocator::CreateResource2.");
        return E_INVALIDARG;
    }
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    return m_Pimpl->CreateResource2(pAllocDesc, pResourceDesc, InitialResourceState, pOptimizedClearValue, ppAllocation, riidResource, ppvResource);
}
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

static inline bool ValidateAllocateMemoryParameters(
    const ALLOCATION_DESC* pAllocDesc,
    const D3D12_RESOURCE_ALLOCATION_INFO* pAllocInfo,
    Allocation** ppAllocation)
{
    return pAllocDesc &&
        pAllocInfo &&
        ppAllocation &&
        (pAllocInfo->Alignment == 0 ||
            pAllocInfo->Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT ||
            pAllocInfo->Alignment == D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT) &&
        pAllocInfo->SizeInBytes != 0 &&
        pAllocInfo->SizeInBytes % (64ull * 1024) == 0;
}

HRESULT Allocator::AllocateMemory(
    const ALLOCATION_DESC* pAllocDesc,
    const D3D12_RESOURCE_ALLOCATION_INFO* pAllocInfo,
    Allocation** ppAllocation)
{
    if(!ValidateAllocateMemoryParameters(pAllocDesc, pAllocInfo, ppAllocation))
    {
        D3D12MA_ASSERT(0 && "Invalid arguments passed to Allocator::AllocateMemory.");
        return E_INVALIDARG;
    }
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    return m_Pimpl->AllocateMemory(pAllocDesc, pAllocInfo, ppAllocation);
}

HRESULT Allocator::CreateAliasingResource(
    Allocation* pAllocation,
    UINT64 AllocationLocalOffset,
    const D3D12_RESOURCE_DESC* pResourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    REFIID riidResource,
    void** ppvResource)
{
    if(!pAllocation || !pResourceDesc || !ppvResource)
    {
        D3D12MA_ASSERT(0 && "Invalid arguments passed to Allocator::CreateAliasingResource.");
        return E_INVALIDARG;
    }
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    return m_Pimpl->CreateAliasingResource(pAllocation, AllocationLocalOffset, pResourceDesc, InitialResourceState, pOptimizedClearValue, riidResource, ppvResource);
}

HRESULT Allocator::CreatePool(
    const POOL_DESC* pPoolDesc,
    Pool** ppPool)
{
    if(!pPoolDesc || !ppPool ||
        (pPoolDesc->MaxBlockCount > 0 && pPoolDesc->MaxBlockCount < pPoolDesc->MinBlockCount) ||
        (pPoolDesc->MinAllocationAlignment > 0 && !IsPow2(pPoolDesc->MinAllocationAlignment)))
    {
        D3D12MA_ASSERT(0 && "Invalid arguments passed to Allocator::CreatePool.");
        return E_INVALIDARG;
    }
    if(!m_Pimpl->HeapFlagsFulfillResourceHeapTier(pPoolDesc->HeapFlags))
    {
        D3D12MA_ASSERT(0 && "Invalid pPoolDesc->HeapFlags passed to Allocator::CreatePool. Did you forget to handle ResourceHeapTier=1?");
        return E_INVALIDARG;
    }
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    *ppPool = D3D12MA_NEW(m_Pimpl->GetAllocs(), Pool)(this, *pPoolDesc);
    HRESULT hr = (*ppPool)->m_Pimpl->Init();
    if(SUCCEEDED(hr))
    {
        m_Pimpl->RegisterPool(*ppPool, pPoolDesc->HeapProperties.Type);
    }
    else
    {
        D3D12MA_DELETE(m_Pimpl->GetAllocs(), *ppPool);
        *ppPool = NULL;
    }
    return hr;
}

void Allocator::SetCurrentFrameIndex(UINT frameIndex)
{
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    m_Pimpl->SetCurrentFrameIndex(frameIndex);
}

void Allocator::CalculateStats(Stats* pStats)
{
    D3D12MA_ASSERT(pStats);
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    m_Pimpl->CalculateStats(*pStats);
}

void Allocator::GetBudget(Budget* pGpuBudget, Budget* pCpuBudget)
{
    if(pGpuBudget == NULL && pCpuBudget == NULL)
    {
        return;
    }
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    m_Pimpl->GetBudget(pGpuBudget, pCpuBudget);
}

void Allocator::BuildStatsString(WCHAR** ppStatsString, BOOL DetailedMap) const
{
    D3D12MA_ASSERT(ppStatsString);
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
    m_Pimpl->BuildStatsString(ppStatsString, DetailedMap);
}

void Allocator::FreeStatsString(WCHAR* pStatsString) const
{
    if (pStatsString != NULL)
    {
        D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
        m_Pimpl->FreeStatsString(pStatsString);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Private class VirtualBlockPimpl definition

class VirtualBlockPimpl
{
public:
    const ALLOCATION_CALLBACKS m_AllocationCallbacks;
    const UINT64 m_Size;
    BlockMetadata* m_Metadata;

    VirtualBlockPimpl(const ALLOCATION_CALLBACKS& allocationCallbacks, const VIRTUAL_BLOCK_DESC& desc);
    ~VirtualBlockPimpl();
};

VirtualBlockPimpl::VirtualBlockPimpl(const ALLOCATION_CALLBACKS& allocationCallbacks, const VIRTUAL_BLOCK_DESC& desc) :
    m_AllocationCallbacks(allocationCallbacks),
    m_Size(desc.Size)
{
    switch (desc.Flags & VIRTUAL_BLOCK_FLAG_ALGORITHM_MASK)
    {
    case VIRTUAL_BLOCK_FLAG_ALGORITHM_TLSF:
        m_Metadata = D3D12MA_NEW(allocationCallbacks, BlockMetadata_TLSF)(&m_AllocationCallbacks, true);
        break;
    case VIRTUAL_BLOCK_FLAG_ALGORITHM_LINEAR:
        m_Metadata = D3D12MA_NEW(allocationCallbacks, BlockMetadata_Linear)(&m_AllocationCallbacks, true);
        break;
    default:
        D3D12MA_ASSERT(0);
    case 0:
        m_Metadata = D3D12MA_NEW(allocationCallbacks, BlockMetadata_Generic)(&m_AllocationCallbacks, true);
        break;
    }
    m_Metadata->Init(m_Size);
}

VirtualBlockPimpl::~VirtualBlockPimpl()
{
    D3D12MA_DELETE(m_AllocationCallbacks, m_Metadata);
}

////////////////////////////////////////////////////////////////////////////////
// Public class VirtualBlock implementation

VirtualBlock::VirtualBlock(const ALLOCATION_CALLBACKS& allocationCallbacks, const VIRTUAL_BLOCK_DESC& desc) :
    m_Pimpl(D3D12MA_NEW(allocationCallbacks, VirtualBlockPimpl)(allocationCallbacks, desc))
{
}

VirtualBlock::~VirtualBlock()
{
    // THIS IS AN IMPORTANT ASSERT!
    // Hitting it means you have some memory leak - unreleased allocations in this virtual block.
    D3D12MA_ASSERT(m_Pimpl->m_Metadata->IsEmpty() && "Some allocations were not freed before destruction of this virtual block!");

    D3D12MA_DELETE(m_Pimpl->m_AllocationCallbacks, m_Pimpl);
}

void VirtualBlock::ReleaseThis()
{
    // Copy is needed because otherwise we would call destructor and invalidate the structure with callbacks before using it to free memory.
    const ALLOCATION_CALLBACKS allocationCallbacksCopy = m_Pimpl->m_AllocationCallbacks;
    D3D12MA_DELETE(allocationCallbacksCopy, this);
}

BOOL VirtualBlock::IsEmpty() const
{
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK

    return m_Pimpl->m_Metadata->IsEmpty() ? TRUE : FALSE;
}

void VirtualBlock::GetAllocationInfo(VirtualAllocation allocation, VIRTUAL_ALLOCATION_INFO* pInfo) const
{
    D3D12MA_ASSERT(allocation.AllocHandle != (AllocHandle)0 && pInfo);

    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK

    m_Pimpl->m_Metadata->GetAllocationInfo(allocation.AllocHandle, *pInfo);
}

HRESULT VirtualBlock::Allocate(const VIRTUAL_ALLOCATION_DESC* pDesc, VirtualAllocation* pAllocation, UINT64* pOffset)
{
    if(!pDesc || !pAllocation || pDesc->Size == 0 || !IsPow2(pDesc->Alignment))
    {
        D3D12MA_ASSERT(0 && "Invalid arguments passed to VirtualBlock::Allocate.");
        return E_INVALIDARG;
    }

    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
        
    const UINT64 alignment = pDesc->Alignment != 0 ? pDesc->Alignment : 1;
    AllocationRequest allocRequest = {};
    if(m_Pimpl->m_Metadata->CreateAllocationRequest(
        pDesc->Size,
        alignment,
        pDesc->Flags & VIRTUAL_ALLOCATION_FLAG_UPPER_ADDRESS,
        &allocRequest))
    {
        m_Pimpl->m_Metadata->Alloc(allocRequest, pDesc->Size, pDesc->pUserData);
        D3D12MA_HEAVY_ASSERT(m_Pimpl->m_Metadata->Validate());
        pAllocation->AllocHandle = allocRequest.allocHandle;

        if (pOffset)
            *pOffset = m_Pimpl->m_Metadata->GetAllocationOffset(allocRequest.allocHandle);
        return S_OK;
    }

    pAllocation->AllocHandle = (AllocHandle)0;
    if (pOffset)
        *pOffset = UINT64_MAX;

    return E_OUTOFMEMORY;
}

void VirtualBlock::FreeAllocation(VirtualAllocation allocation)
{
    if(allocation.AllocHandle == (AllocHandle)0)
        return;

    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK

    m_Pimpl->m_Metadata->Free(allocation.AllocHandle);
    D3D12MA_HEAVY_ASSERT(m_Pimpl->m_Metadata->Validate());
}

void VirtualBlock::Clear()
{
    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK

    m_Pimpl->m_Metadata->Clear();
    D3D12MA_HEAVY_ASSERT(m_Pimpl->m_Metadata->Validate());
}

void VirtualBlock::SetAllocationUserData(VirtualAllocation allocation, void* pUserData)
{
    D3D12MA_ASSERT(allocation.AllocHandle != (AllocHandle)0);

    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK

    m_Pimpl->m_Metadata->SetAllocationUserData(allocation.AllocHandle, pUserData);
}

void VirtualBlock::CalculateStats(StatInfo* pInfo) const
{
    D3D12MA_ASSERT(pInfo);

    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK

    D3D12MA_HEAVY_ASSERT(m_Pimpl->m_Metadata->Validate());
    m_Pimpl->m_Metadata->CalcAllocationStatInfo(*pInfo);
}

void VirtualBlock::BuildStatsString(WCHAR** ppStatsString) const
{
    D3D12MA_ASSERT(ppStatsString);

    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK

    StringBuilder sb(m_Pimpl->m_AllocationCallbacks);
    {
        JsonWriter json(m_Pimpl->m_AllocationCallbacks, sb);
        D3D12MA_HEAVY_ASSERT(m_Pimpl->m_Metadata->Validate());
        m_Pimpl->m_Metadata->WriteAllocationInfoToJson(json);
    } // Scope for JsonWriter

    const size_t length = sb.GetLength();
    WCHAR* result = AllocateArray<WCHAR>(m_Pimpl->m_AllocationCallbacks, length + 1);
    memcpy(result, sb.GetData(), length * sizeof(WCHAR));
    result[length] = L'\0';
    *ppStatsString = result;
}

void VirtualBlock::FreeStatsString(WCHAR* pStatsString) const
{
    if (pStatsString != NULL)
    {
        D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK
        D3D12MA::Free(m_Pimpl->m_AllocationCallbacks, pStatsString);
    }
}


////////////////////////////////////////////////////////////////////////////////
// Public global functions

HRESULT CreateAllocator(const ALLOCATOR_DESC* pDesc, Allocator** ppAllocator)
{
    if(!pDesc || !ppAllocator || !pDesc->pDevice || !pDesc->pAdapter ||
        !(pDesc->PreferredBlockSize == 0 || (pDesc->PreferredBlockSize >= 16 && pDesc->PreferredBlockSize < 0x10000000000ull)))
    {
        D3D12MA_ASSERT(0 && "Invalid arguments passed to CreateAllocator.");
        return E_INVALIDARG;
    }

    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK

    ALLOCATION_CALLBACKS allocationCallbacks;
    SetupAllocationCallbacks(allocationCallbacks, pDesc->pAllocationCallbacks);

    *ppAllocator = D3D12MA_NEW(allocationCallbacks, Allocator)(allocationCallbacks, *pDesc);
    HRESULT hr = (*ppAllocator)->m_Pimpl->Init(*pDesc);
    if(FAILED(hr))
    {
        D3D12MA_DELETE(allocationCallbacks, *ppAllocator);
        *ppAllocator = NULL;
    }
    return hr;
}

HRESULT CreateVirtualBlock(const VIRTUAL_BLOCK_DESC* pDesc, VirtualBlock** ppVirtualBlock)
{
    if(!pDesc || !ppVirtualBlock)
    {
        D3D12MA_ASSERT(0 && "Invalid arguments passed to CreateVirtualBlock.");
        return E_INVALIDARG;
    }

    D3D12MA_DEBUG_GLOBAL_MUTEX_LOCK

    ALLOCATION_CALLBACKS allocationCallbacks;
    SetupAllocationCallbacks(allocationCallbacks, pDesc->pAllocationCallbacks);

    *ppVirtualBlock = D3D12MA_NEW(allocationCallbacks, VirtualBlock)(allocationCallbacks, *pDesc);
    return S_OK;
}

} // namespace D3D12MA
