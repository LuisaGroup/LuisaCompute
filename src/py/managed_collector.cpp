#include <py/managed_collector.h>
#include <py/ref_counter.h>
#include <py/py_stream.h>

namespace luisa::compute {

uint64 ManagedCollector::Allocate() noexcept {
    if (!handlePool.empty()) {
        auto ite = handlePool.back();
        handlePool.pop_back();
        return ite;
    }
    auto lastSize = handles.size();
    auto lastEleSize = lastSize / objPerEle;
    handles.push_back_uninitialized(objPerEle);
    memset(handles.data() + lastSize, -1, sizeof(decltype(handles)::value_type) * objPerEle);
    return lastEleSize;
}

ManagedCollector::ManagedCollector(size_t objPerEle) noexcept : objPerEle(objPerEle) {}

ManagedCollector::~ManagedCollector() noexcept {
    auto ref = RefCounter::current.get();
    for (auto &&i : handles) {
        if (i != invalid_resource_handle) {
            ref->DeRef(i);
        }
    }
}

void ManagedCollector::InRef(size_t element, vstd::span<uint64> handles) noexcept {
    auto ite = handleMap.try_emplace(element, vstd::lazy_eval([this] { return Allocate(); }));
    auto eleArr = Sample(ite.first->second);
    assert(eleArr.size() == handles.size());
    for (auto i : vstd::range(eleArr.size())) {
        auto &ele = eleArr[i];
        auto &handle = handles[i];
        if (ele != invalid_resource_handle) {
            deferredDisposeList.emplace_back(ele);
        }
        ele = handle;
        if (handle != invalid_resource_handle) {
            RefCounter::current->InRef(handle);
        }
    }
}

void ManagedCollector::InRef(size_t element, size_t subElement, uint64 handle) noexcept {
    auto ite = handleMap.try_emplace(element, vstd::lazy_eval([this] { return Allocate(); }));
    auto eleArr = Sample(ite.first->second);
    auto &ele = eleArr[subElement];
    if (ele != invalid_resource_handle) {
        deferredDisposeList.emplace_back(ele);
    }
    ele = handle;
    if (handle != invalid_resource_handle) {
        RefCounter::current->InRef(handle);
    }
}

void ManagedCollector::DeRef(size_t element) noexcept {
    auto ite = handleMap.find(element);
    if (ite == handleMap.end()) return;
    auto eleArr = Sample(ite->second);
    for (auto &&i : eleArr) {
        if (i != invalid_resource_handle) {
            deferredDisposeList.emplace_back(i);
            i = invalid_resource_handle;
        }
    }
    handlePool.emplace_back(ite->second);
    handleMap.erase(ite);
}

void ManagedCollector::AfterExecuteStream(PyStream &stream) noexcept {
    stream.delegates.emplace_back([lst = std::move(deferredDisposeList)] {
        for (auto &&i : lst) {
            RefCounter::current->DeRef(i);
        }
    });
}

}// namespace luisa::compute
