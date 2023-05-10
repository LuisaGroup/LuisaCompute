#include <core/binary_io.h>
#include <core/dynamic_module.h>
#include <runtime/context.h>
#include <dstorage/dstorage.h>
#include <winrt/base.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <vstl/meta_lib.h>
#include <iostream>
#include <core/logging.h>
using winrt::check_hresult;
using winrt::com_ptr;

namespace luisa {
class DStorageStream;
class DStorageImpl {
public:
    DynamicModule dstorage_core_module;
    DynamicModule dstorage_module;
    com_ptr<IDStorageFactory> factory;
    com_ptr<ID3D12Device> device_com;
    ID3D12Device *device;
    com_ptr<IDStorageQueue> queue;
    std::mutex queue_mtx;
    bool dstorage_supported{false};
    DStorageImpl(std::filesystem::path const &runtime_dir, ID3D12Device *device_ptr) noexcept
        : dstorage_core_module{DynamicModule::load(runtime_dir, "dstoragecore")},
          device{device_ptr},
          dstorage_module{DynamicModule::load(runtime_dir, "dstorage")} {
        HRESULT(WINAPI * DStorageGetFactory)
        (REFIID riid, _COM_Outptr_ void **ppv);
        if (!dstorage_module || !dstorage_core_module) {
            LUISA_WARNING("Direct-Storage DLL not found.");
            return;
        }
        DStorageGetFactory = reinterpret_cast<decltype(DStorageGetFactory)>(GetProcAddress(reinterpret_cast<HMODULE>(dstorage_module.handle()), "DStorageGetFactory"));
        if (!device) {
            uint32_t dxgiFactoryFlags = 0;
#ifndef NDEBUG
            // Enable the debug layer (requires the Graphics Tools "optional feature").
            // NOTE: Enabling the debug layer after device creation will invalidate the active device.
            {
                com_ptr<ID3D12Debug> debugController;
                if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
                    debugController->EnableDebugLayer();

                    // Enable additional debug layers.
                    dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
                }
            }
#endif
            com_ptr<IDXGIFactory4> dxgiFactory;
            com_ptr<IDXGIAdapter1> adapter;
            check_hresult(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(dxgiFactory.put())));
            for (auto adapterIndex = 0u; dxgiFactory->EnumAdapters1(adapterIndex, adapter.put()) != DXGI_ERROR_NOT_FOUND; adapterIndex++) {
                DXGI_ADAPTER_DESC1 desc;
                adapter->GetDesc1(&desc);
                if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
                    check_hresult(D3D12CreateDevice(
                        adapter.get(), D3D_FEATURE_LEVEL_12_1,
                        IID_PPV_ARGS(device_com.put())));
                    break;
                }
            }
            if (adapter == nullptr) {
                LUISA_WARNING("Direct Storage not supported on this device.");
                return;
            }
            device = device_com.get();
        }
        check_hresult(DStorageGetFactory(IID_PPV_ARGS(factory.put())));
        DSTORAGE_QUEUE_DESC queue_desc{
            .SourceType = DSTORAGE_REQUEST_SOURCE_FILE,
            .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
            .Priority = DSTORAGE_PRIORITY_NORMAL,
            .Device = device};
        check_hresult(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(queue.put())));
        dstorage_supported = true;
    }
    DStorageImpl(compute::Context const &ctx, ID3D12Device *device_ptr) noexcept
        : DStorageImpl{ctx.runtime_directory(), device_ptr} {}
    BinaryStream *create_stream(luisa::string_view path) noexcept;
};
class DStorageStream : public BinaryStream {
    com_ptr<IDStorageFile> _file;
    DStorageImpl *_impl;
    com_ptr<ID3D12Fence> fence;
    uint64_t fence_idx{1};
    size_t _length{0};
    size_t _pos{0};

public:
    DStorageStream(com_ptr<IDStorageFile> &&file, DStorageImpl *impl, size_t length) noexcept
        : _file{std::move(file)},
          _impl{impl},
          _length{length} {
        check_hresult(impl->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put())));
    }
    size_t length() const noexcept override {
        return _length;
    }
    size_t pos() const noexcept override {
        return _pos;
    }
    void read(luisa::span<std::byte> dst) noexcept override {
        size_t sz = std::min(dst.size(), _length - _pos);
        if (sz == 0) [[unlikely]] {
            return;
        }
        DSTORAGE_REQUEST request = {};
        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
        request.Source.File.Source = _file.get();
        request.Source.File.Offset = _pos;
        request.Source.File.Size = sz;
        request.UncompressedSize = sz;
        request.Destination.Memory.Buffer = dst.data();
        request.Destination.Memory.Size = sz;
        {
            std::lock_guard lck{_impl->queue_mtx};
            _impl->queue->EnqueueRequest(&request);
            _impl->queue->EnqueueSignal(fence.get(), fence_idx);
            _impl->queue->Submit();
        }
        HANDLE event_handle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
        {
            auto scope_exit = vstd::scope_exit([&] {
                CloseHandle(event_handle);
            });
            check_hresult(fence->SetEventOnCompletion(fence_idx, event_handle));
            WaitForSingleObject(event_handle, INFINITE);
        }
        fence_idx++;
        _pos += sz;
    }
};
BinaryStream *DStorageImpl::create_stream(luisa::string_view path) noexcept {
    com_ptr<IDStorageFile> file;
    luisa::vector<wchar_t> wstr;
    wstr.push_back_uninitialized(path.size() + 1);
    wstr[path.size()] = 0;
    for (size_t i = 0; i < path.size(); ++i) {
        wstr[i] = path[i];
    }
    HRESULT hr = factory->OpenFile(wstr.data(), IID_PPV_ARGS(file.put()));
    if (FAILED(hr)) {
        return nullptr;
    }
    size_t length;
    BY_HANDLE_FILE_INFORMATION info{};
    check_hresult(file->GetFileInformation(&info));
    if constexpr (sizeof(size_t) > sizeof(DWORD)) {
        length = info.nFileSizeHigh;
        length <<= (sizeof(DWORD) * 8);
        length |= info.nFileSizeLow;
    } else {
        length = info.nFileSizeLow;
    }
    if (length == 0) return nullptr;
    return new_with_allocator<DStorageStream>(std::move(file), this, length);
}
LUISA_EXPORT_API bool dstorage_supported(void *impl) {
    return reinterpret_cast<DStorageImpl *>(impl)->dstorage_supported;
}
LUISA_EXPORT_API void *create_dstorage_impl(compute::Context const &ctx, ID3D12Device *device) noexcept {
    return new_with_allocator<DStorageImpl>(ctx, device);
}
LUISA_EXPORT_API void delete_dstorage_impl(void *ptr) noexcept {
    delete_with_allocator(reinterpret_cast<DStorageImpl *>(ptr));
}
LUISA_EXPORT_API BinaryStream *create_dstorage_stream(void *impl, luisa::string_view path) noexcept {
    return reinterpret_cast<DStorageImpl *>(impl)->create_stream(path);
}

}// namespace luisa
