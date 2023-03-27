use luisa_compute_api_types as api;
use serde::*;
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ContextCreate {
    pub exe_path: String,
    pub ret: api::Context,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ContextDestroy {
    pub ctx: api::Context,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeviceCreate {
    pub ctx: api::Context,
    pub name: String,
    pub properties: String,
    pub ret: api::Device,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeviceDestroy {
    pub device: api::Device,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeviceRetain {
    pub device: api::Device,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeviceRelease {
    pub device: api::Device,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BufferDestroy {
    pub device: api::Device,
    pub buffer: api::Buffer,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TextureCreate {
    pub device: api::Device,
    pub format: api::PixelFormat,
    pub dim: u32,
    pub w: u32,
    pub h: u32,
    pub d: u32,
    pub mips: u32,
    pub ret: api::CreatedResourceInfoRemote,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TextureDestroy {
    pub device: api::Device,
    pub texture: api::Texture,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamCreate {
    pub device: api::Device,
    pub stream_tag: api::StreamTag,
    pub ret: api::CreatedResourceInfoRemote,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamDestroy {
    pub device: api::Device,
    pub stream: api::Stream,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamSynchronize {
    pub device: api::Device,
    pub stream: api::Stream,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ShaderDestroy {
    pub device: api::Device,
    pub shader: api::Shader,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EventCreate {
    pub device: api::Device,
    pub ret: api::CreatedResourceInfoRemote,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EventDestroy {
    pub device: api::Device,
    pub event: api::Event,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EventSignal {
    pub device: api::Device,
    pub event: api::Event,
    pub stream: api::Stream,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EventWait {
    pub device: api::Device,
    pub event: api::Event,
    pub stream: api::Stream,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EventSynchronize {
    pub device: api::Device,
    pub event: api::Event,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BindlessArrayCreate {
    pub device: api::Device,
    pub n: usize,
    pub ret: api::CreatedResourceInfoRemote,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BindlessArrayDestroy {
    pub device: api::Device,
    pub array: api::BindlessArray,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MeshCreate {
    pub device: api::Device,
    pub option: api::AccelOption,
    pub ret: api::CreatedResourceInfoRemote,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MeshDestroy {
    pub device: api::Device,
    pub mesh: api::Mesh,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AccelCreate {
    pub device: api::Device,
    pub option: api::AccelOption,
    pub ret: api::CreatedResourceInfoRemote,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AccelDestroy {
    pub device: api::Device,
    pub accel: api::Accel,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SwapchainCreate {
    pub device: api::Device,
    pub window_handle: u64,
    pub stream_handle: api::Stream,
    pub width: u32,
    pub height: u32,
    pub allow_hdr: bool,
    pub vsync: bool,
    pub back_buffer_size: u32,
    pub ret: api::CreatedSwapchainInfoRemote,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SwapchainDestroy {
    pub device: api::Device,
    pub swapchain: api::Swapchain,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SwapchainPresent {
    pub device: api::Device,
    pub stream: api::Stream,
    pub swapchain: api::Swapchain,
    pub image: api::Texture,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PixelFormatToStorage {
    pub format: api::PixelFormat,
    pub ret: api::PixelStorage,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SetLogLevelVerbose {
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SetLogLevelInfo {
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SetLogLevelWarning {
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SetLogLevelError {
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogVerbose {
    pub msg: String,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogInfo {
    pub msg: String,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogWarning {
    pub msg: String,
    pub message_id: u64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogError {
    pub msg: String,
    pub message_id: u64,
}
