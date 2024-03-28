#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif
#include <vulkan/vulkan.h>
#if defined(LUISA_PLATFORM_WINDOWS)
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#elif defined(LUISA_PLATFORM_APPLE)
#include <vulkan/vulkan_macos.h>
#elif defined(LUISA_PLATFORM_UNIX)
#include <X11/Xlib.h>
#include <vulkan/vulkan_xlib.h>
#if LUISA_ENABLE_WAYLAND
#include <dlfcn.h>
#include <vulkan/vulkan_wayland.h>
#include <wayland-client.h>
#endif
#else
#error "Unsupported platform"
#endif

#include <volk.h>