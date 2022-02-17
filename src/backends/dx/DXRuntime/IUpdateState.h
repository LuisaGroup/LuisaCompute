#pragma once
namespace toolhub::directx {
class ResourceStateTracker;
class CommandBufferBuilder;
class IUpdateState {
public:
    virtual ~IUpdateState() = default;
    virtual void PreProcessStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker) const = 0;
    virtual void UpdateStates(
        CommandBufferBuilder& builder,
        ResourceStateTracker &tracker) const = 0;
};
}// namespace toolhub::directx