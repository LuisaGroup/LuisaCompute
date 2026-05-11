import("net.http")
import("utils.archive")
import("lib.detect.find_file")
import("core.project.config")
local _sdks = {
    dx_sdk = {
        name = 'dx_sdk_20260511.zip'
    },
    vk_sdk = {
        name = 'linux_dxc_2025_07_14.x86_64.zip'
    }
}

local function sdk_address(sdk)
    local address = sdk['address']
    if address == nil then
        return 'https://github.com/LuisaGroup/SDKs/releases/download/sdk/'
    end
    if #address == 0 then -- no download
        return nil
    end
    return address
end
local function sdk_mirror_addresses(sdk)
    return sdk['mirror_addresses'] or {}
end
local function sdks()
    return _sdks
end
local lc_project_dir = path.directory(os.scriptdir())
local function sdk_dir(custom_dir)
    if custom_dir then
        if not path.is_absolute(custom_dir) then
            custom_dir = path.absolute(custom_dir, os.projectdir())
        end
        return custom_dir
    else
        return path.join(lc_project_dir, 'SDKs/')
    end
end

local function get_or_create_sdk_dir(custom_dir)
    local dir = sdk_dir(custom_dir)
    local lib = import('lib')
    lib.mkdirs(dir)
    return dir
end

-- use_lib_cache = true

local function try_download(zip, url, mirror_urls, dst_dir, settings)
    local function check_file_valid()
        local zip_dir = find_file(zip, {dst_dir})
        return zip_dir ~= nil
    end
    http.download(url .. zip, dst_dir)
    if check_file_valid() then
        return
    end
    for i, mirror_url in ipairs(mirror_urls) do
        http.download(mirror_url .. zip, dst_dir)
        if check_file_valid() then
            return
        end
    end
    utils.error("Download " .. zip .. " failed, please check your internet.")
end

local function file_from_github(sdk_map, dir)
    local zip, address, mirror_address
    zip = sdk_map['name']
    address = sdk_address(sdk_map)
    if not address then
        return
    end
    mirror_address = sdk_mirror_addresses(sdk_map)

    local zip_dir = find_file(zip, {dir})
    local dst_dir = path.join(dir, zip)
    if (zip_dir == nil) then
        local url = vformat(address)
        print("downloading: " .. url .. zip .. " to: " .. dir)
        try_download(zip, url, mirror_address, dst_dir, {
            continue = false
        })
    end
end

-- tool

local function find_tool_zip(tool_name, dir)
    local zip_dir = find_file(tool_name, {dir})
    return {
        name = tool_name,
        dir = zip_dir
    }
end

local function unzip_sdk(tool_name, in_dir, out_dir)
    local zip_file = find_tool_zip(tool_name, in_dir)
    if (zip_file.dir ~= nil) then
        print("installing: " .. zip_file.name)
        os.mkdir(out_dir)
        archive.extract(zip_file.dir, out_dir)
    else
        utils.error("failed to install " .. tool_name .. ", file " .. zip_file.name .. " not found!")
    end
end

local function install_sdk(sdk_map, custom_dir)
    local dir = get_or_create_sdk_dir(custom_dir)
    local _sdks = sdks()
    if not sdk_map then
        utils.error("Invalid sdk: " .. sdk_map["name"])
        return
    end
    file_from_github(sdk_map, dir)
end

local function check_file(sdk_name, custom_dir)
    local dir = sdk_dir(custom_dir)
    local _sdks = sdks()
    local sdk_map = _sdks[sdk_name]
    local zip = sdk_map['name']
    local zip_dir = find_file(zip, {dir})
    return zip_dir ~= nil
end

function on_install_sdk(target, rule_name)
    local custom_sdk_dir
    custom_sdk_dir = target:extraconf("rules", rule_name, "sdk_dir")
    if not custom_sdk_dir then
        custom_sdk_dir = get_config("lc_sdk_dir")
    end
    if type(custom_sdk_dir) == 'string' and #custom_sdk_dir ==0 then
        return nil
    end
    local libnames = target:extraconf("rules", rule_name, "libnames")
    if type(libnames) == "string" or (type(libnames) == "table" and type(libnames["name"]) == "string") then
        libnames = {libnames}
    end
    local sdks = sdks()
    local _sdk_dir = sdk_dir(custom_sdk_dir)
    os.mkdir(_sdk_dir)
    import("async.jobgraph")
    import("async.runjobs")
    if #libnames == 0 then
        return
    end
    local jobs = jobgraph.new()
    local job_count = 0
    for _, lib in ipairs(libnames) do
        local job_name = target:name() .. tostring(job_count)
        job_count = job_count + 1
        jobs:add(job_name, function()
            local copy_dir = lib["copy_dir"]
            if not copy_dir then
                copy_dir = target:targetdir()
            elseif #copy_dir > 0 then
                copy_dir = path.join(copy_dir, os.host(), os.arch())
            end
            if #copy_dir > 0 then
                os.mkdir(copy_dir)
            end
            local sdk_map

            local function process_sdk_map(sdk_map)
                if sdk_map["plat_spec"] then
                    local t = sdk_map['name']
                    sdk_map['name'] = path.basename(t) .. '-' .. os.host() .. '-' .. os.arch() .. path.extension(t)
                end
                install_sdk(sdk_map, custom_sdk_dir)
            end
            if type(lib) == "string" then
                sdk_map = sdks[lib]
                process_sdk_map(sdk_map)
                local valid = check_file(lib, custom_sdk_dir)
                if not valid then
                    utils.error("Library: " .. sdks()[lib]['name'] .. " not installed, should download from " ..
                                    sdk_address(sdks()[lib]) .. ' to ' .. sdk_dir(custom_sdk_dir) .. '.')
                    return
                end
            else
                sdk_map = lib
                if not sdk_map['address'] then
                    sdk_map['address'] = ''
                end
                process_sdk_map(sdk_map)
            end
            local sdk_name = sdk_map["name"]

            if not sdk_name then
                utils.error("Package invalid without name.")
                return
            end
            local extract_dir = lib["extract_dir"]
            if not extract_dir or #extract_dir == 0 then
                extract_dir = path.join(_sdk_dir, path.basename(sdk_name))
            end
            -- Check cache
            local target_cache_dir = path.join(os.projectdir(), "build/.lcsdk", os.host(), os.arch())
            local target_cache_file = path.join(target_cache_dir, sdk_name .. ".ini")

            local require_extract
            local function is_empty_folder()
                if os.exists(extract_dir) and not os.isfile(extract_dir) then
                    for _, v in ipairs(os.filedirs(path.join(extract_dir, '*'))) do
                        return false
                    end
                    return true
                else
                    return true
                end
            end
            local downloaded_pack = path.join(_sdk_dir, sdk_name)
            if not os.exists(downloaded_pack) then
                utils.error(downloaded_pack .. ' not found.')
                return
            end
            local file_sha256 = hash.sha256(downloaded_pack)
            local function is_cache_mismatch()
                if not os.exists(target_cache_file) then
                    return true
                end
                return io.readfile(target_cache_file) ~= file_sha256
            end
            local function unzip()
                unzip_sdk(sdk_map['name'], _sdk_dir, extract_dir)
                os.mkdir(target_cache_dir)
                io.writefile(target_cache_file, file_sha256)
            end
            if is_empty_folder() then
                print("Package " .. sdk_name .. " extract_dir empty, extracting.")
                unzip()
            elseif is_cache_mismatch() then
                print("Package " .. sdk_name .. " hash mismatch, extracting.")
                unzip()
            end
            if #copy_dir > 0 then
                for _, filepath in ipairs(os.filedirs(path.join(extract_dir, "*"))) do
                    os.cp(filepath, path.join(copy_dir, path.filename(filepath)), {
                        copy_if_different = true,
                        async = true
                    })
                end
            end
        end)
    end

    runjobs(target:name() .. "_install", jobs, {
        comax = 1000,
        timeout = -1,
        timer = function(running_jobs_indices)
            utils.error("timeout.")
        end
    })
end
