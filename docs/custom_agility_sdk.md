# Custom DirectX Agility SDK Configuration

The DX backend can load a custom Agility SDK (e.g., SM 6.9 / SDK 1.619+) without
modifying the source, by subclassing `DirectXDeviceConfigExt` at runtime.

## Overview

The DX backend exports `D3D12SDKVersion` and `D3D12SDKPath` symbols that the
DirectX 12 runtime looks up to locate the Agility SDK. By default, these use the
bundled headers (`LCAgilitySDK/`, `D3D12_PREVIEW_SDK_VERSION = 717`) and path
(`.\\D3D12\\`). You can override both at device creation time via the extension API.

## Prerequisites

1. Download a custom Agility SDK release from
   [Microsoft's GitHub releases](https://github.com/microsoft/DirectX-AgilitySDK/releases).
2. Extract the archive to a directory (e.g. `C:/agility-sdk-619/`).
   The directory should contain at least:
   - `D3D12Core.dll`
   - `d3d12SDKVersion.h` (or equivalent headers)

## Runtime Configuration via Extension API

Provide a custom Agility SDK version and path at device creation time by
subclassing `DirectXDeviceConfigExt`.

### Step 1: Create a custom extension class

```cpp
#include <luisa/backends/ext/dx_config_ext.h>

using namespace luisa::compute;

class MyAgilitySDKConfig : public DirectXDeviceConfigExt {
public:
    // Return a custom SDK version (0 = use default)
    [[nodiscard]] uint32_t GetSDKVersion() const noexcept override {
        return 619;  // SM 6.9 Agility SDK version
    }

    // Return a custom SDK DLL path (empty = use default ".\\D3D12\\")
    [[nodiscard]] luisa::string_view GetSDKPath() const noexcept override {
        return "C:/agility-sdk-619/";
    }
};
```

### Step 2: Pass the extension when creating the device

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/dx_config_ext.h>

int main() {
    Context context{};
    auto ext = luisa::make_unique<MyAgilitySDKConfig>();
    DeviceConfig config{};
    config.extension = std::move(ext);
    auto device = context.create_device("dx", &config);
    // ...
}
```

### How it works

1. During `Device` construction, `GetSDKVersion()` and `GetSDKPath()` are queried
   from the extension.
2. The exported `D3D12SDKVersion` and `D3D12SDKPath` symbols are updated **before**
   `CreateDXGIFactory2` / `D3D12CreateDevice` is called.
3. The DirectX runtime picks up the custom DLLs from the specified path.

The override happens only when the extension returns non-zero version / non-empty
path; default values are preserved otherwise.

## Header Compatibility

When using a newer Agility SDK (e.g., version 619), the bundled headers
(`LCAgilitySDK/d3d12.h`, SDK version 616, preview 717) may not match. Options:

1. **Use bundled headers** (default): Works if the preview version (717) contains
   the definitions you need. Most SM 6.9 definitions are already present.
2. **Provide your own headers**: Place a custom `d3d12.h` alongside the SDK
   DLLs and adjust include paths manually.

## Verification

Run your application that uses `MyAgilitySDKConfig` (see example above).
The device will be created with the custom SDK version and path.
