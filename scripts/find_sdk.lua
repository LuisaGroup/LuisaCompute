import("net.http")
import("utils.archive")
import("lib.detect.find_file")
import("core.project.config")
local packages = import("packages")

find_sdk = find_sdk or {}

-- use_lib_cache = true

function file_from_github(zip, dir, address, valid_sha256)
    local zip_dir = find_file(zip, {dir})
    local dst_dir = path.join(dir, zip)
    if (zip_dir == nil) then
        local url = vformat(address)
        print("download: " .. url .. zip .. " to: " .. dir)
        http.download(url .. zip, dst_dir, {
            continue = false
        })
    else
        local sha256 = hash.sha256(zip_dir)
        local is_valid = valid_sha256 == sha256
        if not is_valid then
            local url = vformat(address)
            print(zip .. " is invalid, download: " .. url .. zip)
            os.rm(zip_dir)
            http.download(url .. zip, dst_dir, {
                continue = false
            })
        end
    end
end

-- tool

function find_tool_zip(tool_name, dir)
    local zip_dir = find_file(tool_name, {dir})
    return {
        name = tool_name,
        dir = zip_dir
    }
end

function unzip_sdk(tool_name, in_dir, out_dir)
    local zip_file = find_tool_zip(tool_name, in_dir)
    if (zip_file.dir ~= nil) then
        print("install: " .. zip_file.name)
        archive.extract(zip_file.dir, out_dir)
    else
        utils.error("failed to install " .. tool_name .. ", file " .. zip_file.name .. " not found!")
    end
end

function install_sdk(sdk_name)
    local dir = packages.get_or_create_sdk_dir(os.arch())
    local _sdks = packages.sdks()
    local sdk_map = _sdks[sdk_name]
    if not sdk_map then
        utils.error("Invalid sdk: " .. sdk_name)
        return
    end
    file_from_github(sdk_map['name'], dir, packages.sdk_address(), sdk_map['sha256'])
end

function check_file(sdk_name)
    local dir = packages.sdk_dir(os.arch())
    local _sdks = packages.sdks()
    local sdk_map = _sdks[sdk_name]
    local zip = sdk_map['name']
    local zip_dir = find_file(zip, {dir})
    if (zip_dir == nil) then
        return false
    end
    local sha256 = hash.sha256(zip_dir)
    local valid_sha256 = sdk_map['sha256']
    local is_valid = sha256 == valid_sha256
    return is_valid
end
