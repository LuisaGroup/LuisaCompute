#pragma once
#include <vstl/common.h>
#include <vstl/lockfree_array_queue.h>
namespace luisa::compute {
class PyStream;
class ManagedCollector : public vstd::IOperatorNewBase{
    size_t objPerEle;
    vstd::vector<uint64> handles;
    vstd::vector<size_t> handlePool;
    vstd::unordered_map<size_t, size_t> handleMap;
    vstd::vector<uint64> deferredDisposeList;
    size_t allocCapa = 64;
    uint64 Allocate() noexcept;
    vstd::span<uint64> Sample(size_t index) noexcept {
        return {handles.data() + index * objPerEle, objPerEle};
    }

public:
    ManagedCollector(size_t objPerEle) noexcept;
    ~ManagedCollector() noexcept;
    void InRef(size_t element, size_t subElement, uint64 handle) noexcept;
    void InRef(size_t element, vstd::span<uint64> handles) noexcept;
    void DeRef(size_t element) noexcept;
    void AfterExecuteStream(PyStream &stream) noexcept;
};
}// namespace luisa::compute