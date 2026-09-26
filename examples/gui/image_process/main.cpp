// Image Process -- interactive image processing window (Dear ImGui + LuisaCompute).
//
// The window is split into two resolution adaptive panes: the left pane shows the
// processed RGB image, the right pane shows the processed alpha channel in
// grayscale. A floating "Settings" window holds the load/save buttons and the
// operator list (append / insert / remove, each operator has a color argument).
//
// The Luisa shader is only dispatched when the settings change (new image, new
// operator, edited argument); the GUI itself runs at the vsync frame rate.
//
// A headless self test is available through `--headless` (see headless_test.cpp).

#include "image_process.h"

#ifdef LUISA_ENABLE_GUI
#define IMAGE_PROCESS_HAS_GUI 1
#else
#define IMAGE_PROCESS_HAS_GUI 0
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
#endif

#if IMAGE_PROCESS_HAS_GUI
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <luisa/gui/imgui_window.h>
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <optional>
#include <string>

#include <luisa/core/clock.h>
#include <luisa/runtime/context.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>

namespace image_process {

namespace {

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct Options {
    luisa::string backend;
    bool headless{false};
    luisa::string output_dir{"image_process_output"};
    luisa::string image_path;
    luisa::string save_path;
    luisa::string operator_spec;
    luisa::vector<OperatorEntry> initial_operators;
    uint32_t frames{0u};
};

void print_usage(const char *program) noexcept {
    LUISA_INFO("Usage: {} <backend> [options]", program);
    LUISA_INFO("  <backend>            cuda, dx, vk, metal, hip, fallback");
    LUISA_INFO("  --headless           run the headless self test (no window)");
    LUISA_INFO("  --output-dir <dir>   headless test output directory (default: {})", "image_process_output");
    LUISA_INFO("  --image <file>       load this image on startup (skips the file dialog)");
    LUISA_INFO("  --save-to <file>     save the processed image to this path before exiting");
    LUISA_INFO("  --operators <spec>   initial operator list, e.g. \"mul 0.9 0.9 0.9 1; add 0.1 0 0 0\"");
    LUISA_INFO("  --frames <n>         close the window after n frames (default: 0 = until closed)");
}

/// Parse an operator list specification: "<name> [r g b a]; <name> [r g b a]; ...".
[[nodiscard]] bool parse_operator_spec(luisa::string_view text,
                                       luisa::vector<OperatorEntry> &operators) noexcept {
    auto trim = [](luisa::string_view s) noexcept {
        auto begin = s.find_first_not_of(" \t\r\n");
        if (begin == luisa::string_view::npos) { return luisa::string_view{}; }
        auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(begin, end - begin + 1u);
    };
    auto start = size_t{0u};
    while (start < text.size()) {
        auto separator = text.find(';', start);
        auto segment = trim(text.substr(start, separator == luisa::string_view::npos ? luisa::string_view::npos : separator - start));
        start = separator == luisa::string_view::npos ? text.size() : separator + 1u;
        if (segment.empty()) { continue; }
        auto name_end = segment.find_first_of(" \t");
        auto name = segment.substr(0u, name_end);
        OperatorEntry entry;
        auto found = false;
        for (auto i = 0u; i < op_code_count; i++) {
            auto candidate = luisa::string_view{op_name(static_cast<OpCode>(i))};
            auto equal = name.size() == candidate.size();
            if (equal) {
                for (auto k = 0u; k < name.size(); k++) {
                    if (std::tolower(static_cast<unsigned char>(name[k])) !=
                        std::tolower(static_cast<unsigned char>(candidate[k]))) {
                        equal = false;
                        break;
                    }
                }
            }
            if (equal) {
                entry.code = static_cast<OpCode>(i);
                found = true;
                break;
            }
        }
        if (!found) {
            LUISA_WARNING("Unknown operator '{}' in --operators.", name);
            return false;
        }
        if (name_end != luisa::string_view::npos) {
            auto arguments = trim(segment.substr(name_end));
            for (auto i = 0u; i < 4u && !arguments.empty(); i++) {
                auto space = arguments.find_first_of(" \t");
                auto token = arguments.substr(0u, space);
                entry.argument[i] = std::strtof(std::string{token}.c_str(), nullptr);
                if (space == luisa::string_view::npos) { break; }
                arguments = trim(arguments.substr(space));
            }
        }
        operators.emplace_back(entry);
    }
    return true;
}

[[nodiscard]] bool parse_options(int argc, char *argv[], Options &options) noexcept {
    if (argc <= 1) { return false; }
    options.backend = argv[1];
    for (auto i = 2; i < argc; i++) {
        luisa::string_view arg{argv[i]};
        auto next = [&]() -> luisa::string_view {
            if (i + 1 < argc) { return argv[++i]; }
            LUISA_WARNING("Missing value for option '{}'.", arg);
            return {};
        };
        if (arg == "--headless") {
            options.headless = true;
        } else if (arg == "--output-dir") {
            options.output_dir = next();
        } else if (arg == "--image") {
            options.image_path = next();
        } else if (arg == "--save-to") {
            options.save_path = next();
        } else if (arg == "--operators") {
            options.operator_spec = next();
            if (!parse_operator_spec(options.operator_spec, options.initial_operators)) {
                return false;
            }
        } else if (arg == "--frames") {
            options.frames = static_cast<uint32_t>(std::strtoul(std::string{next()}.c_str(), nullptr, 10));
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            LUISA_WARNING("Unknown option '{}'.", arg);
            return false;
        }
    }
    return !options.backend.empty();
}

// ---------------------------------------------------------------------------
// Small path helpers (UTF-8 strings)
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::string file_name_of(luisa::string_view path) noexcept {
    auto pos = path.find_last_of("/\\");
    return luisa::string{pos == luisa::string_view::npos ? path : path.substr(pos + 1u)};
}

[[nodiscard]] luisa::string stem_of(luisa::string_view path) noexcept {
    auto name = file_name_of(path);
    auto dot = name.find_last_of('.');
    return dot == luisa::string::npos ? name : luisa::string{name.substr(0u, dot)};
}

[[nodiscard]] luisa::string directory_of(luisa::string_view path) noexcept {
    auto pos = path.find_last_of("/\\");
    return pos == luisa::string_view::npos ? luisa::string{} : luisa::string{path.substr(0u, pos + 1u)};
}

[[nodiscard]] const char *const *operator_names() noexcept {
    static const auto names = [] {
        std::array<const char *, op_code_count> result{};
        for (auto i = 0u; i < op_code_count; i++) {
            result[i] = op_name(static_cast<OpCode>(i));
        }
        return result;
    }();
    return names.data();
}

}// namespace

#if IMAGE_PROCESS_HAS_GUI

namespace {

// ---------------------------------------------------------------------------
// Windows file dialogs
// ---------------------------------------------------------------------------

#if defined(_WIN32)

constexpr wchar_t kOpenFilter[] =
    L"All supported images (*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.hdr;*.psd;*.gif;*.pic;*.ppm;*.pgm;*.pnm)\0"
    L"*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.hdr;*.psd;*.gif;*.pic;*.ppm;*.pgm;*.pnm\0"
    L"PNG (*.png)\0*.png\0"
    L"JPEG (*.jpg;*.jpeg)\0*.jpg;*.jpeg\0"
    L"BMP (*.bmp)\0*.bmp\0"
    L"TGA (*.tga)\0*.tga\0"
    L"Radiance HDR (*.hdr)\0*.hdr\0"
    L"All files (*.*)\0*.*\0\0";

constexpr wchar_t kSaveFilter[] =
    L"PNG image (*.png)\0*.png\0"
    L"JPEG image (*.jpg;*.jpeg)\0*.jpg;*.jpeg\0"
    L"BMP image (*.bmp)\0*.bmp\0"
    L"TGA image (*.tga)\0*.tga\0"
    L"Radiance HDR image (*.hdr)\0*.hdr\0"
    L"All files (*.*)\0*.*\0\0";

[[nodiscard]] luisa::string wide_to_utf8(const wchar_t *wide) noexcept {
    if (wide == nullptr || *wide == L'\0') { return {}; }
    auto length = static_cast<int>(std::wcslen(wide));
    auto size = ::WideCharToMultiByte(CP_UTF8, 0, wide, length, nullptr, 0, nullptr, nullptr);
    if (size <= 0) { return {}; }
    luisa::string result(static_cast<size_t>(size), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, wide, length, result.data(), size, nullptr, nullptr);
    return result;
}

[[nodiscard]] bool utf8_to_wide(luisa::string_view utf8, wchar_t *out, size_t capacity) noexcept {
    if (capacity == 0u) { return false; }
    out[0] = L'\0';
    if (utf8.empty()) { return true; }
    auto size = ::MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()),
                                      nullptr, 0);
    if (size <= 0 || static_cast<size_t>(size) >= capacity) { return false; }
    ::MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), out, size);
    out[size] = L'\0';
    return true;
}

[[nodiscard]] int save_filter_index(luisa::string_view extension) noexcept {
    if (extension == ".jpg" || extension == ".jpeg") { return 2; }
    if (extension == ".bmp") { return 3; }
    if (extension == ".tga") { return 4; }
    if (extension == ".hdr") { return 5; }
    return 1;// png
}

[[nodiscard]] luisa::optional<luisa::string> open_image_dialog(GLFWwindow *window) noexcept {
    wchar_t file_buffer[4096]{};
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = window == nullptr ? nullptr : glfwGetWin32Window(window);
    ofn.lpstrFilter = kOpenFilter;
    ofn.nFilterIndex = 1;
    ofn.lpstrFile = file_buffer;
    ofn.nMaxFile = static_cast<DWORD>(sizeof(file_buffer) / sizeof(file_buffer[0]));
    ofn.lpstrTitle = L"Load image";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (::GetOpenFileNameW(&ofn) == FALSE) { return luisa::nullopt; }
    return wide_to_utf8(file_buffer);
}

[[nodiscard]] luisa::optional<luisa::string> save_image_dialog(GLFWwindow *window,
                                                             luisa::string_view directory,
                                                             luisa::string_view file_name,
                                                             luisa::string_view extension) noexcept {
    wchar_t file_buffer[4096]{};
    auto default_path = std::string{directory} + std::string{file_name};
    if (!utf8_to_wide(default_path, file_buffer, sizeof(file_buffer) / sizeof(file_buffer[0]))) {
        LUISA_WARNING("The default save path '{}' is not a valid path.", default_path);
        return luisa::nullopt;
    }
    wchar_t default_extension[16]{};
    auto ext = extension;
    if (!ext.empty() && ext.front() == '.') { ext = ext.substr(1u); }
    if (ext.empty()) { ext = "png"; }
    if (!utf8_to_wide(ext, default_extension, sizeof(default_extension) / sizeof(default_extension[0]))) {
        std::wcsncpy(default_extension, L"png", 15);
    }
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = window == nullptr ? nullptr : glfwGetWin32Window(window);
    ofn.lpstrFilter = kSaveFilter;
    ofn.nFilterIndex = static_cast<DWORD>(save_filter_index(extension));
    ofn.lpstrFile = file_buffer;
    ofn.nMaxFile = static_cast<DWORD>(sizeof(file_buffer) / sizeof(file_buffer[0]));
    ofn.lpstrTitle = L"Save processed image";
    ofn.lpstrDefExt = default_extension;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (::GetSaveFileNameW(&ofn) == FALSE) { return luisa::nullopt; }
    return wide_to_utf8(file_buffer);
}

#else// not Windows

[[nodiscard]] luisa::optional<luisa::string> open_image_dialog(GLFWwindow *) noexcept {
    LUISA_WARNING("Native file dialogs are only implemented on Windows; use --image <path>.");
    return luisa::nullopt;
}

[[nodiscard]] luisa::optional<luisa::string> save_image_dialog(GLFWwindow *, luisa::string_view,
                                                             luisa::string_view, luisa::string_view) noexcept {
    LUISA_WARNING("Native file dialogs are only implemented on Windows; use --save-to <path>.");
    return luisa::nullopt;
}

#endif

// ---------------------------------------------------------------------------
// The application
// ---------------------------------------------------------------------------

class ImageProcessApp {

private:
    Options _options;
    Stream _stream;
    ImageProcessPipeline _pipeline;
    luisa::unique_ptr<ImGuiWindow> _window;
    luisa::vector<OperatorEntry> _operators;
    luisa::string _input_path;
    luisa::string _status{"Load an image to start."};
    bool _status_is_error{false};
    bool _dirty{false};
    uint64_t _result_tex{0u};
    uint64_t _alpha_tex{0u};
    uint64_t _frame_index{0u};

private:
    /// Re-encode the operator list, upload it and dispatch the processing shader.
    void process_if_dirty() noexcept {
        if (!_dirty) { return; }
        _dirty = false;
        if (!_pipeline.has_image()) { return; }
        _pipeline.set_operators(_operators);
    }

    void release_textures() noexcept {
        if (_result_tex != 0u) {
            _window->unregister_texture(_result_tex);
            _result_tex = 0u;
        }
        if (_alpha_tex != 0u) {
            _window->unregister_texture(_alpha_tex);
            _alpha_tex = 0u;
        }
    }

    void load_image(const luisa::string &path) noexcept {
        ImageData data;
        luisa::string error;
        if (!load_image_file(path, data, error)) {
            _status = luisa::format("Load failed: {}", error);
            _status_is_error = true;
            LUISA_WARNING("{}", _status);
            return;
        }
        // Drop the shared texture handles before the device images are recreated.
        release_textures();
        _pipeline.load(luisa::span<const float>{data.pixels.data(), data.pixels.size()},
                       make_uint2(data.width, data.height));
        _result_tex = _window->register_texture(_pipeline.result(), Sampler::linear_linear_edge());
        _alpha_tex = _window->register_texture(_pipeline.alpha(), Sampler::linear_linear_edge());
        _input_path = path;
        _dirty = true;
        _status = luisa::format("Loaded '{}' ({} x {})", file_name_of(path), data.width, data.height);
        _status_is_error = false;
        LUISA_INFO("{}", _status);
    }

    void save_image(const luisa::string &path) noexcept {
        if (!_pipeline.has_image()) {
            _status = "Nothing to save: load an image first.";
            _status_is_error = true;
            return;
        }
        auto extension = lower_extension(path);
        if (!is_supported_save_extension(extension)) {
            _status = luisa::format("Unsupported extension '{}' (use .png/.jpg/.bmp/.tga/.hdr).",
                                    extension.empty() ? "<none>" : extension);
            _status_is_error = true;
            return;
        }
        auto pixels = _pipeline.readback_result();
        luisa::string error;
        if (!save_image_file(path, luisa::span<const float>{pixels.data(), pixels.size()},
                             _pipeline.size().x, _pipeline.size().y, error)) {
            _status = luisa::format("Save failed: {}", error);
            _status_is_error = true;
            LUISA_WARNING("{}", _status);
            return;
        }
        _status = luisa::format("Saved '{}'", file_name_of(path));
        _status_is_error = false;
        LUISA_INFO("{}", _status);
    }

    void draw_background() noexcept {
        auto *viewport = ImGui::GetMainViewport();
        auto *draw_list = ImGui::GetBackgroundDrawList();
        draw_list->AddRectFilled(viewport->Pos,
                                 ImVec2{viewport->Pos.x + viewport->Size.x,
                                        viewport->Pos.y + viewport->Size.y},
                                 IM_COL32(18, 18, 22, 255));
    }

    void draw_image_pane(const char *title, ImVec2 position, ImVec2 size,
                         uint64_t texture, uint2 image_size) noexcept {
        auto ui = _window->dpi_scale();// scale hardcoded layout constants with the display DPI
        ImGui::SetNextWindowPos(position);
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{6.0f * ui, 6.0f * ui});
        ImGui::Begin(title, nullptr,
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                         ImGuiWindowFlags_NoSavedSettings);
        auto content = ImGui::GetContentRegionAvail();
        if (texture != 0u && image_size.x != 0u && image_size.y != 0u &&
            content.x > 8.0f * ui && content.y > 8.0f * ui) {
            auto image_width = static_cast<float>(image_size.x);
            auto image_height = static_cast<float>(image_size.y);
            auto scale = std::min(content.x / image_width, content.y / image_height);
            auto draw_size = ImVec2{image_width * scale, image_height * scale};
            auto cursor = ImGui::GetCursorPos();
            ImGui::SetCursorPos(ImVec2{cursor.x + (content.x - draw_size.x) * 0.5f,
                                       cursor.y + (content.y - draw_size.y) * 0.5f});
            ImGui::Image(texture, draw_size);
            ImGui::SetCursorPos(cursor);
            ImGui::TextDisabled("%u x %u  (%.0f%%)", image_size.x, image_size.y, scale * 100.0f);
        } else {
            auto text_pos = [&] {
                auto text_size = ImGui::CalcTextSize("No image loaded");
                return ImVec2{std::max((content.x - text_size.x) * 0.5f, 0.0f),
                              std::max((content.y - text_size.y) * 0.5f, 0.0f)};
            }();
            ImGui::SetCursorPos(text_pos);
            ImGui::TextDisabled("No image loaded");
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void draw_image_panes() noexcept {
        auto *viewport = ImGui::GetMainViewport();
        auto position = viewport->WorkPos;
        auto size = viewport->WorkSize;
        auto gap = 6.0f * _window->dpi_scale();
        auto left_width = std::floor((size.x - gap) * 0.5f);
        draw_image_pane("RGB", ImVec2{position.x, position.y},
                        ImVec2{left_width, size.y}, _result_tex, _pipeline.size());
        draw_image_pane("Alpha", ImVec2{position.x + left_width + gap, position.y},
                        ImVec2{size.x - left_width - gap, size.y}, _alpha_tex, _pipeline.size());
    }

    void draw_settings_window() noexcept {
        auto ui = _window->dpi_scale();// scale hardcoded layout constants with the display DPI
        auto *viewport = ImGui::GetMainViewport();
        // Note: imgui.ini stores window sizes in ImGui units, so a Settings
        // window saved at 100% is restored at its old size (ImGuiCond_FirstUseEver
        // means the new default applies on a fresh profile).
        ImGui::SetNextWindowPos(ImVec2{viewport->WorkPos.x + 32.0f * ui, viewport->WorkPos.y + 32.0f * ui},
                                ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2{640.0f * ui, 560.0f * ui}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Settings");

        auto &io = ImGui::GetIO();
        auto image_loaded = _pipeline.has_image();
        if (ImGui::Button("Load Image")) {
            if (auto path = open_image_dialog(_window->handle())) { load_image(*path); }
        }
        ImGui::SameLine();
        if (ImGui::Button("Save Image")) {
            process_if_dirty();// make sure the current settings are applied
            auto default_name = image_loaded ? stem_of(_input_path) + "_processed" + lower_extension(_input_path) : luisa::string{"processed.png"};
            if (auto path = save_image_dialog(_window->handle(), directory_of(_input_path),
                                              default_name, lower_extension(_input_path))) {
                save_image(*path);
            }
        }
        ImGui::SameLine();
        ImGui::Text("|  %s  |  %.1f FPS (%.2f ms)", _options.backend.c_str(), io.Framerate,
                    io.Framerate > 0.0f ? 1000.0f / io.Framerate : 0.0f);

        if (_status_is_error) {
            ImGui::TextColored(ImVec4{1.0f, 0.55f, 0.35f, 1.0f}, "%s", _status.c_str());
        } else {
            ImGui::TextColored(ImVec4{0.65f, 0.85f, 1.0f, 1.0f}, "%s", _status.c_str());
        }
        if (image_loaded) {
            ImGui::Text("Source: %s", _input_path.c_str());
            ImGui::Text("Resolution: %u x %u", _pipeline.size().x, _pipeline.size().y);
        }

        ImGui::SeparatorText("Operators");
        ImGui::TextDisabled("Applied in order, top to bottom (drag a color to edit its argument).");

        auto insert_at = -1;
        auto remove_at = -1;
        auto list_height = std::max(120.0f * ui, ImGui::GetContentRegionAvail().y - 72.0f * ui);
        if (ImGui::BeginChild("##operator_list", ImVec2{0.0f, list_height}, ImGuiChildFlags_Borders)) {
            for (size_t i = 0u; i < _operators.size(); i++) {
                auto &entry = _operators[i];
                ImGui::PushID(static_cast<int>(i));
                ImGui::Text("%zu.", i + 1u);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f * ui);
                auto code = static_cast<int>(entry.code);
                if (ImGui::Combo("##op", &code, operator_names(), static_cast<int>(op_code_count))) {
                    entry.code = static_cast<OpCode>(code);
                    _dirty = true;
                }
                ImGui::SameLine();
                if (op_uses_argument(entry.code)) {
                    if (ImGui::ColorEdit4("##arg", entry.argument,
                                          ImGuiColorEditFlags_NoInputs |
                                              ImGuiColorEditFlags_AlphaBar |
                                              ImGuiColorEditFlags_AlphaPreviewHalf)) {
                        _dirty = true;
                    }
                    ImGui::SetItemTooltip("RGBA argument of the operator");
                } else {
                    ImGui::TextDisabled("abs has no argument");
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Insert")) { insert_at = static_cast<int>(i); }
                ImGui::SetItemTooltip("Insert a new operator before this one");
                ImGui::SameLine();
                if (ImGui::SmallButton("Remove")) { remove_at = static_cast<int>(i); }
                ImGui::PopID();
            }
            if (_operators.empty()) {
                ImGui::TextDisabled("(empty list: the output is the input image)");
            }
        }
        ImGui::EndChild();
        if (insert_at >= 0) {
            if (_operators.size() < max_operator_count) {
                _operators.insert(_operators.begin() + insert_at, OperatorEntry{});
                _dirty = true;
            } else {
                _status = luisa::format("Operator limit reached ({}).", max_operator_count);
                _status_is_error = true;
            }
        }
        if (remove_at >= 0) {
            _operators.erase(_operators.begin() + remove_at);
            _dirty = true;
        }
        if (ImGui::Button("Append Operator")) {
            if (_operators.size() < max_operator_count) {
                _operators.emplace_back();
                _dirty = true;
            } else {
                _status = luisa::format("Operator limit reached ({}).", max_operator_count);
                _status_is_error = true;
            }
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%zu / %zu operators", _operators.size(), max_operator_count);

        ImGui::End();
    }

public:
    ImageProcessApp(Device &device, const Options &options) noexcept
        : _options{options},
          _stream{device.create_stream(StreamTag::GRAPHICS)},
          _pipeline{device, _stream},
          _window{luisa::make_unique<ImGuiWindow>(
              device, _stream, "Image Process",
              ImGuiWindow::Config{.size = make_uint2(1600u, 900u), .vsync = true})} {
        _window->with_context([] {
            ImGui::StyleColorsDark();
            auto &style = ImGui::GetStyle();
            style.WindowRounding = 4.0f;
            style.FrameRounding = 3.0f;
            style.GrabRounding = 3.0f;
        });
    }

    [[nodiscard]] int run() noexcept {
        if (!_options.image_path.empty()) { load_image(_options.image_path); }
        if (!_options.initial_operators.empty()) {
            _operators = _options.initial_operators;
            _dirty = true;
        }
        Clock clock;
        while (!_window->should_close() &&
               (_options.frames == 0u || _frame_index < _options.frames)) {
            _window->prepare_frame();
            draw_background();
            draw_image_panes();
            draw_settings_window();
            // Re-dispatch only when the settings changed, never every frame.
            process_if_dirty();
            _window->render_frame();
            _frame_index++;
        }
        auto elapsed_ms = clock.toc();
        LUISA_INFO("Rendered {} frame(s) in {:.1f} ms ({:.1f} FPS, vsync).", _frame_index,
                   elapsed_ms, _frame_index == 0u ? 0.0 : 1000.0 * static_cast<double>(_frame_index) / elapsed_ms);
        // Apply pending settings and honor --save-to before shutting down.
        process_if_dirty();
        auto failed = false;
        if (!_options.save_path.empty()) {
            save_image(_options.save_path);
            failed = _status_is_error;
        }
        _stream << synchronize();
        return failed ? 1 : 0;
    }
};

}// namespace

#endif// IMAGE_PROCESS_HAS_GUI

}// namespace image_process

int main(int argc, char *argv[]) {
    luisa::compute::Context context{argv[0]};
    image_process::Options options;
    if (!image_process::parse_options(argc, argv, options)) {
        image_process::print_usage(argv[0]);
        return 1;
    }
    auto device = context.create_device(options.backend);
    if (options.headless) {
        return image_process::run_headless_tests(device, options.output_dir);
    }
#if IMAGE_PROCESS_HAS_GUI
    image_process::ImageProcessApp app{device, options};
    return app.run();
#else
    LUISA_ERROR("This build has no GUI support; use --headless.");
    return 1;
#endif
}
