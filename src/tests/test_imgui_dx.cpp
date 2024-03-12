// Dear ImGui: standalone example application for DirectX 12

// Learn about Dear ImGui:
// - FAQ                  https://dearimgui.com/faq
// - Getting Started      https://dearimgui.com/getting-started
// - Documentation        https://dearimgui.com/docs (same as your local docs/ folder).
// - Introduction, links and more at the top of imgui.cpp

// Important: to compile on 32-bit systems, the DirectX12 backend requires code to be compiled with '#define ImTextureID ImU64'.
// This is because we need ImTextureID to carry a 64-bit value and by default ImTextureID is defined as void*.
// This define is set in the example .vcxproj file and need to be replicated in your app or by adding it to your imconfig.h file.

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include <tchar.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/functional.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/backends/ext/dx_config_ext.h>
#include <luisa/backends/ext/dx_custom_cmd.h>

#ifdef _DEBUG
#define DX12_ENABLE_DEBUG_LAYER
#endif

#ifdef DX12_ENABLE_DEBUG_LAYER
#include <dxgidebug.h>
#pragma comment(lib, "dxguid.lib")
#endif

// struct FrameContext
// {
//     ID3D12CommandAllocator* CommandAllocator;
//     UINT64                  FenceValue;
// };

// // Data
// static int const                    NUM_FRAMES_IN_FLIGHT = 3;
// static FrameContext                 g_frameContext[NUM_FRAMES_IN_FLIGHT] = {};
// static UINT                         g_frameIndex = 0;

// static int const                    NUM_BACK_BUFFERS = 3;
// static ID3D12Device*                g_pd3dDevice = nullptr;
// static ID3D12DescriptorHeap*        g_pd3dRtvDescHeap = nullptr;
// static ID3D12DescriptorHeap*        g_pd3dSrvDescHeap = nullptr;
// static ID3D12CommandQueue*          g_pd3dCommandQueue = nullptr;
// static ID3D12GraphicsCommandList*   g_pd3dCommandList = nullptr;
// static ID3D12Fence*                 g_fence = nullptr;
// static HANDLE                       g_fenceEvent = nullptr;
// static UINT64                       g_fenceLastSignaledValue = 0;
// static IDXGISwapChain3*             g_pSwapChain = nullptr;
// static HANDLE                       g_hSwapChainWaitableObject = nullptr;
static D3D12_CPU_DESCRIPTOR_HANDLE g_mainRenderTargetDescriptor;
struct ImguiDx12Context {
    ID3D12Device *device;
    ID3D12DescriptorHeap *pd3dRtvDescHeap = nullptr;
    ID3D12DescriptorHeap *pd3dSrvDescHeap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE mainRenderTargetDescriptor;
    inline static luisa::move_only_function<void(uint32_t width, uint32_t height)> resize_func;
};
bool CreateDeviceD3D(HWND hWnd, ImguiDx12Context *ctx);
struct ImguiDeviceConfigExt : public luisa::compute::DirectXDeviceConfigExt {
    ImguiDx12Context *ctx;
    ImguiDeviceConfigExt(ImguiDx12Context *ctx) : ctx(ctx) {}
    void ReadbackDX12Device(
        ID3D12Device *device,
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *factory,
        luisa::compute::DirectXFuncTable const *funcTable,
        luisa::BinaryIO const *shaderIo,
        IDxcCompiler3 *dxcCompiler,
        IDxcLibrary *dxcLibrary,
        IDxcUtils *dxcUtils,
        ID3D12DescriptorHeap *shaderDescriptor,
        ID3D12DescriptorHeap *samplerDescriptor) noexcept override {
        ctx->device = device;
    }
};
class ImguiDrawPass : public luisa::compute::DXCustomCmd {
public:
    luisa::vector<ResourceUsage> _usages;
    ImguiDx12Context *ctx;
    luisa::compute::ImageView<float> img;
    ImVec4 clear_color;
    bool viewport_enabled = false;
    ImguiDrawPass(ImguiDx12Context *ctx, luisa::compute::ImageView<float> img, ImVec4 clear_color)
        : ctx(ctx), img(img), clear_color(clear_color) {
        using namespace luisa;
        using namespace luisa::compute;

        _usages.emplace_back(
            Argument::Texture{
                img.handle(),
                img.level()},
            D3D12_RESOURCE_STATE_RENDER_TARGET);
    }
    luisa::compute::StreamTag stream_tag() const noexcept override { return luisa::compute::StreamTag::GRAPHICS; }
    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept override {
        const float clear_color_with_alpha[4] = {clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w};
        command_list->ClearRenderTargetView(ctx->mainRenderTargetDescriptor, clear_color_with_alpha, 0, nullptr);
        command_list->OMSetRenderTargets(1, &ctx->mainRenderTargetDescriptor, FALSE, nullptr);
        command_list->SetDescriptorHeaps(1, &ctx->pd3dSrvDescHeap);
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), command_list);
    }
    luisa::span<const ResourceUsage> get_resource_usages() const noexcept override {
        return _usages;
    }
};
struct ImGui_ImplDX12_Data {
    ID3D12Device *pd3dDevice;
    ID3D12RootSignature *pRootSignature;
    ID3D12PipelineState *pPipelineState;
    DXGI_FORMAT RTVFormat;
    ID3D12Resource *pFontTextureResource;
    D3D12_CPU_DESCRIPTOR_HANDLE hFontSrvCpuDescHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE hFontSrvGpuDescHandle;
    ID3D12DescriptorHeap *pd3dSrvDescHeap;
    UINT numFramesInFlight;
};
struct ImGui_ImplDX12_FrameContext {
    ID3D12CommandAllocator *CommandAllocator;
    ID3D12Resource *RenderTarget;
    D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetCpuDescriptors;
};
struct ImGui_ImplDX12_RenderBuffers {
    ID3D12Resource *IndexBuffer;
    ID3D12Resource *VertexBuffer;
    int IndexBufferSize;
    int VertexBufferSize;
};
struct ImGui_ImplDX12_ViewportData {
    // Window
    ID3D12CommandQueue *CommandQueue;
    ID3D12GraphicsCommandList *CommandList;
    ID3D12DescriptorHeap *RtvDescHeap;
    IDXGISwapChain3 *SwapChain;
    ID3D12Fence *Fence;
    UINT64 FenceSignaledValue;
    HANDLE FenceEvent;
    UINT NumFramesInFlight;
    ImGui_ImplDX12_FrameContext *FrameCtx;

    // Render buffers
    UINT FrameIndex;
    ImGui_ImplDX12_RenderBuffers *FrameRenderBuffers;
};

// Backend data stored in io.BackendRendererUserData to allow support for multiple Dear ImGui contexts
// It is STRONGLY preferred that you use docking branch with multi-viewports (== single Dear ImGui context + multiple windows) instead of multiple Dear ImGui contexts.
static ImGui_ImplDX12_Data *ImGui_ImplDX12_GetBackendData() {
    return ImGui::GetCurrentContext() ? (ImGui_ImplDX12_Data *)ImGui::GetIO().BackendRendererUserData : nullptr;
}

class ImguiRedrawPass : public luisa::compute::DXCustomCmd {
    ImGuiViewport *viewport;
    ImguiRedrawPass(
        ImGuiViewport *viewport) : viewport(viewport) {}
    luisa::compute::StreamTag stream_tag() const noexcept override { return luisa::compute::StreamTag::GRAPHICS; }
    luisa::span<const ResourceUsage> get_resource_usages() const noexcept override {
        return {};
    }
    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *cmd_list) const noexcept override {
        ImGui_ImplDX12_Data *bd = ImGui_ImplDX12_GetBackendData();
        ImGui_ImplDX12_ViewportData *vd = (ImGui_ImplDX12_ViewportData *)viewport->RendererUserData;

        ImGui_ImplDX12_FrameContext *frame_context = &vd->FrameCtx[vd->FrameIndex % bd->numFramesInFlight];
        UINT back_buffer_idx = vd->SwapChain->GetCurrentBackBufferIndex();

        const ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = vd->FrameCtx[back_buffer_idx].RenderTarget;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        cmd_list->ResourceBarrier(1, &barrier);
        cmd_list->OMSetRenderTargets(1, &vd->FrameCtx[back_buffer_idx].RenderTargetCpuDescriptors, FALSE, nullptr);
        if (!(viewport->Flags & ImGuiViewportFlags_NoRendererClear))
            cmd_list->ClearRenderTargetView(vd->FrameCtx[back_buffer_idx].RenderTargetCpuDescriptors, (float *)&clear_color, 0, nullptr);
        cmd_list->SetDescriptorHeaps(1, &bd->pd3dSrvDescHeap);

        ImGui_ImplDX12_RenderDrawData(viewport->DrawData, cmd_list);

        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        cmd_list->ResourceBarrier(1, &barrier);
    }
};
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Main code
int main(int argc, char *argv[]) {
    using namespace luisa;
    using namespace luisa::compute;
    Context context{argv[0]};
    ImguiDx12Context dx12_ctx;
    DeviceConfig config{
        .extension = luisa::make_unique<ImguiDeviceConfigExt>(&dx12_ctx),
        .inqueue_buffer_limit = false};
    uint2 resolution{1280, 800};
    WNDCLASSEXW wc = {sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"ImGui Example", nullptr};
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"Dear ImGui DirectX12 Example", WS_OVERLAPPEDWINDOW, 100, 100, resolution.x, resolution.y, nullptr, nullptr, wc.hInstance, nullptr);
    auto device = context.create_device("dx", &config);

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Swapchain swap_chain;
    Image<float> ldr_image;
    CreateDeviceD3D(hwnd, &dx12_ctx);
    ImguiDx12Context::resize_func = [&](uint width, uint height) {
        resolution = uint2(width, height);
        stream.synchronize();
        (&swap_chain)->~Swapchain();
        new (std::launder(&swap_chain)) Swapchain(device.create_swapchain(
            reinterpret_cast<uint64_t>(hwnd),
            stream,
            resolution,
            false, false, 3));
        ldr_image = device.create_image<float>(swap_chain.backend_storage(), resolution);
        dx12_ctx.device->CreateRenderTargetView(reinterpret_cast<ID3D12Resource *>(ldr_image.native_handle()), nullptr, dx12_ctx.mainRenderTargetDescriptor);
    };
    // {device.create_swapchain(
    //     reinterpret_cast<uint64_t>(hwnd),
    //     stream,
    //     resolution,
    //     false, false, 3)};
    //  = device.create_image<float>(swap_chain.backend_storage(), resolution);

    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);
    // init dx12
    {
        // dx12_ctx.device->CreateRenderTargetView(reinterpret_cast<ID3D12Resource *>(ldr_image.native_handle()), nullptr, dx12_ctx.mainRenderTargetDescriptor);
    }
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;// Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;    // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;  // Enable Multi-Viewport / Platform Windows
    //io.ConfigViewportsNoAutoMerge = true;
    //io.ConfigViewportsNoTaskBarIcon = true;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle &style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX12_Init(dx12_ctx.device, 3,
                        DXGI_FORMAT_R8G8B8A8_UNORM, dx12_ctx.pd3dSrvDescHeap,
                        dx12_ctx.pd3dSrvDescHeap->GetCPUDescriptorHandleForHeapStart(),
                        dx12_ctx.pd3dSrvDescHeap->GetGPUDescriptorHandleForHeapStart());

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return a nullptr. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != nullptr);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    bool done = false;
    TimelineEvent graphics_event = device.create_timeline_event();
    uint64_t frame_index = 0;

    while (!done) {
        // Poll and handle messages (inputs, window resize, etc.)
        // See the WndProc() function below for our to dispatch events to the Win32 backend.
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done)
            break;
        uint64_t this_frame = frame_index;
        frame_index += 1;
        if (this_frame >= 3) {
            graphics_event.synchronize(this_frame - (3 - 1));
        }
        // Start the Dear ImGui frame
        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGuiPlatformIO &platform_io = ImGui::GetPlatformIO();
        // TODO
        struct RenderCtx {
            Stream &stream;
            ImGuiViewport *viewport;
        };
        platform_io.Renderer_RenderWindow = [](ImGuiViewport *viewport, void *void_ptr) {
            auto ptr = reinterpret_cast<RenderCtx *>(void_ptr);
            // ptr->stream << luisa::make_unique<
        };
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");// Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");         // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window);// Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float *)&clear_color);// Edit 3 floats representing a color

            if (ImGui::Button("Button"))// Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window) {
            ImGui::Begin("Another Window", &show_another_window);// Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        if (this_frame > 0) {
            stream << graphics_event.wait(this_frame);
        }
        stream << luisa::make_unique<ImguiDrawPass>(&dx12_ctx, ldr_image, clear_color) << swap_chain.present(ldr_image);

        // FrameContext *frameCtx = WaitForNextFrameResources();
        // UINT backBufferIdx = g_pSwapChain->GetCurrentBackBufferIndex();
        // frameCtx->CommandAllocator->Reset();

        // D3D12_RESOURCE_BARRIER barrier = {};
        // barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        // barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        // barrier.Transition.pResource = g_mainRenderTargetResource[backBufferIdx];
        // barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        // barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        // barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        // g_pd3dCommandList->Reset(frameCtx->CommandAllocator, nullptr);
        // g_pd3dCommandList->ResourceBarrier(1, &barrier);

        // // Render Dear ImGui graphics
        // const float clear_color_with_alpha[4] = {clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w};
        // g_pd3dCommandList->ClearRenderTargetView(g_mainRenderTargetDescriptor[backBufferIdx], clear_color_with_alpha, 0, nullptr);
        // g_pd3dCommandList->OMSetRenderTargets(1, &g_mainRenderTargetDescriptor[backBufferIdx], FALSE, nullptr);
        // g_pd3dCommandList->SetDescriptorHeaps(1, &g_pd3dSrvDescHeap);
        // ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), g_pd3dCommandList);
        // barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        // barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        // g_pd3dCommandList->ResourceBarrier(1, &barrier);
        // g_pd3dCommandList->Close();

        // g_pd3dCommandQueue->ExecuteCommandLists(1, (ID3D12CommandList *const *)&g_pd3dCommandList);

        // Update and Render additional Platform Windows
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            // ImGui::RenderPlatformWindowsDefault(nullptr, (void *)g_pd3dCommandList);
        }
        stream << graphics_event.signal(frame_index);
        // g_pSwapChain->Present(1, 0);// Present with vsync
        // //g_pSwapChain->Present(0, 0); // Present without vsync

        // UINT64 fenceValue = g_fenceLastSignaledValue + 1;
        // g_pd3dCommandQueue->Signal(g_fence, fenceValue);
        // g_fenceLastSignaledValue = fenceValue;
        // frameCtx->FenceValue = fenceValue;
    }
    stream.synchronize();
    // Cleanup
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}

// Helper functions
bool CreateDeviceD3D(HWND hWnd, ImguiDx12Context *ctx) {
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        desc.NumDescriptors = 1;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        desc.NodeMask = 1;
        if (ctx->device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&ctx->pd3dRtvDescHeap)) != S_OK)
            return false;

        SIZE_T rtvDescriptorSize = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = ctx->pd3dRtvDescHeap->GetCPUDescriptorHandleForHeapStart();
        ctx->mainRenderTargetDescriptor = rtvHandle;
        rtvHandle.ptr += rtvDescriptorSize;
    }

    {
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = 1;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (ctx->device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&ctx->pd3dSrvDescHeap)) != S_OK)
            return false;
    }

    return true;
}

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    using namespace luisa;
    using namespace luisa::compute;
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
        case WM_SIZE:
            if (wParam != SIZE_MINIMIZED)
                ImguiDx12Context::resize_func((UINT)LOWORD(lParam), (UINT)HIWORD(lParam));
            // LUISA_ERROR("Resize not support yet");
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU)// Disable ALT application menu
                return 0;
            break;
        case WM_DESTROY:
            ::PostQuitMessage(0);
            return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}
