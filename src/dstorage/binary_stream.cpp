#include <core/binary_io.h>
#include <core/dynamic_module.h>
#include <runtime/context.h>
#include <dstorage/dstorage.h>
#include <winrt/base.h>
#include <d3d12.h>
#include <vstl/meta_lib.h>
#include <iostream>
using winrt::check_hresult;
using winrt::com_ptr;

namespace luisa {
using namespace compute;
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
    DStorageImpl(std::filesystem::path const &runtime_dir, ID3D12Device *device_ptr) noexcept
        : dstorage_core_module{DynamicModule::load(runtime_dir, "dstoragecore")},
          device{device_ptr},
          dstorage_module{DynamicModule::load(runtime_dir, "dstorage")} {
        HRESULT(WINAPI * DStorageGetFactory)
        (REFIID riid, _COM_Outptr_ void **ppv);
        DStorageGetFactory = reinterpret_cast<decltype(DStorageGetFactory)>(GetProcAddress(reinterpret_cast<HMODULE>(dstorage_module.handle()), "DStorageGetFactory"));
        if (!device) {
            check_hresult(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(device_com.put())));
            device = device_com.get();
        }
        check_hresult(DStorageGetFactory(IID_PPV_ARGS(factory.put())));
        DSTORAGE_QUEUE_DESC queue_desc{
            .SourceType = DSTORAGE_REQUEST_SOURCE_FILE,
            .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
            .Priority = DSTORAGE_PRIORITY_NORMAL,
            .Device = device};
        check_hresult(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(queue.put())));
    }
    DStorageImpl(Context const &ctx, ID3D12Device *device_ptr) noexcept
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
    DStorageStream(com_ptr<IDStorageFile> &&file, DStorageImpl *impl) noexcept
        : _file{std::move(file)},
          _impl{impl} {
        BY_HANDLE_FILE_INFORMATION info{};
        check_hresult(_file->GetFileInformation(&info));
        if constexpr (sizeof(size_t) > sizeof(DWORD)) {
            _length = info.nFileSizeHigh;
            _length <<= (sizeof(DWORD) * 8);
            _length |= info.nFileSizeLow;
        } else {
            _length = info.nFileSizeLow;
        }

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
        request.UncompressedSize = 0;
        request.Destination.Memory.Buffer = dst.data();
        request.Destination.Memory.Size = sz;
        {
            // std::lock_guard lck{_impl->queue_mtx};
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
    return new_with_allocator<DStorageStream>(std::move(file), this);
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
 // int main() {
 //     using namespace luisa;
 //     DynamicModule dstorage_core_module{DynamicModule::load("dstoragecore")};
 //     DynamicModule dstorage_module{DynamicModule::load("dstorage")};
 //     HRESULT(WINAPI * DStorageGetFactory)
 //     (REFIID riid, _COM_Outptr_ void **ppv);
 //     DStorageGetFactory = reinterpret_cast<decltype(DStorageGetFactory)>(GetProcAddress(reinterpret_cast<HMODULE>(dstorage_module.handle()), "DStorageGetFactory"));

//     com_ptr<IDStorageFactory> factory;
//     com_ptr<ID3D12Device> device;
//     com_ptr<IDStorageQueue> queue;
//     com_ptr<ID3D12Fence> fence;

//     check_hresult(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)));
//     check_hresult(DStorageGetFactory(IID_PPV_ARGS(factory.put())));
//     // DSTORAGE_QUEUE_DESC queueDesc{};
//     // queueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
//     // queueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
//     // queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
//     // queueDesc.Device = device.get();
//     DSTORAGE_QUEUE_DESC queue_desc{
//         .SourceType = DSTORAGE_REQUEST_SOURCE_FILE,
//         .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
//         .Priority = DSTORAGE_PRIORITY_NORMAL,
//         .Device = device.get()};
//     check_hresult(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(queue.put())));
//     check_hresult(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put())));

//     com_ptr<IDStorageFile> file;
//     HRESULT hr = factory->OpenFile(L"test_file.txt", IID_PPV_ARGS(file.put()));
//     if (FAILED(hr)) {
//         return 0;
//     }

//     BY_HANDLE_FILE_INFORMATION info{};
//     check_hresult(file->GetFileInformation(&info));
//     size_t fileSize;
//     fileSize = info.nFileSizeHigh;
//     fileSize <<= 32;
//     fileSize |= info.nFileSizeLow;
//     char c[32];
//     for (auto &&i : c) {
//         i = 'X';
//     }
//     DSTORAGE_REQUEST request = {};
//     request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
//     request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
//     request.Source.File.Source = file.get();
//     request.Source.File.Offset = 0;
//     request.Source.File.Size = fileSize;
//     request.UncompressedSize = fileSize;

//     request.Destination.Memory.Buffer = c;
//     request.Destination.Memory.Size = fileSize;

//     constexpr uint64_t fenceValue = 1;
//     queue->EnqueueRequest(&request);
//     queue->EnqueueSignal(fence.get(), fenceValue);
//     queue->Submit();

//     // Configure a fence to be signaled when the request is completed
// {}
//     HANDLE fenceEvent = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
//     auto scope_exit = vstd::scope_exit([&] {
//         CloseHandle(fenceEvent);
//     });
//     check_hresult(fence->SetEventOnCompletion(fenceValue, fenceEvent));
//     // {
//     //     // std::lock_guard lck{_impl->queue_mtx};
//     //     queue->EnqueueRequest(&request);
//     //     queue->EnqueueSignal(fence.get(), fence_idx);
//     //     queue->Submit();
//     // }
//     // {
//     //     HANDLE event_handle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
//     //     auto scope_exit = vstd::scope_exit([&] {
//     //         CloseHandle(event_handle);
//     //     });
//     //     check_hresult(fence->SetEventOnCompletion(fence_idx, event_handle));
//     //     WaitForSingleObject(fence.get(), INFINITE);
//     // }
//     // Tell DirectStorage to start executing all queued items.

//     // Wait for the submitted work to complete
//     WaitForSingleObject(fenceEvent, INFINITE);

//     for (auto &&i : c) {
//         std::cout << i << ' ';
//     }
//     // Check the status array for errors.
//     // If an error was detected the first failure record
//     // can be retrieved to get more details.
//     DSTORAGE_ERROR_RECORD errorRecord{};
//     queue->RetrieveErrorRecord(&errorRecord);
//     if (FAILED(errorRecord.FirstFailure.HResult)) {
//         //
//         // errorRecord.FailureCount - The number of failed requests in the queue since the last
//         //                            RetrieveErrorRecord call.
//         // errorRecord.FirstFailure - Detailed record about the first failed command in the enqueue order.
//         //
//         std::cout << "The DirectStorage request failed! HRESULT=0x" << std::hex << errorRecord.FirstFailure.HResult << std::endl;
//     } else {
//         std::cout << "The DirectStorage request completed successfully!" << std::endl;
//     }

//     return 0;

//     // com_ptr<IDStorageQueue> queue;

//     // DSTORAGE_QUEUE_DESC queue_desc{
//     //     .SourceType = DSTORAGE_REQUEST_SOURCE_FILE,
//     //     .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
//     //     .Priority = DSTORAGE_PRIORITY_NORMAL,
//     //     .Device = device.get()};

//     // BY_HANDLE_FILE_INFORMATION info{};
//     // check_hresult(file->GetFileInformation(&info));
//     // uint32_t fileSize = info.nFileSizeLow;

//     // constexpr auto fence_idx = 1;
//     // {
//     //     // std::lock_guard lck{_impl->queue_mtx};
//     //     queue->EnqueueRequest(&request);
//     //     queue->EnqueueSignal(fence.get(), fence_idx);
//     //     queue->Submit();
//     // }
//     // {
//     //     HANDLE event_handle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
//     //     auto scope_exit = vstd::scope_exit([&] {
//     //         CloseHandle(event_handle);
//     //     });
//     //     check_hresult(fence->SetEventOnCompletion(fence_idx, event_handle));
//     //     WaitForSingleObject(fence.get(), INFINITE);
//     // }
//     // for (auto &&i : c) {
//     //     std::cout << i << ' ';
//     // }
//     // std::cout << '\n';
//     // auto impl = new_with_allocator<DStorageImpl>("./");
//     // auto strm = impl->create_stream("test_file.txt");
//     // strm->read({});
//     // delete_with_allocator(strm);
//     // delete_with_allocator(impl);
// }