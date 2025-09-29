local _sdks = {
    dx_sdk = {
        -- from:
        --> xmake l hash.sha256 SDKs/x64/dx_sdk_20240920.zip
        sha256 = "5b2b59a687ceeab14b9a49bbd1635cb43ed8c4101df99abcca102376076f7787",
        name = 'dx_sdk_20250816.zip',
    }
}

function sdk_address(sdk)
    return sdk['address'] or 'https://github.com/LuisaGroup/SDKs/releases/download/sdk/'
end
function sdk_mirror_addresses(sdk)
    return sdk['mirror_addresses'] or {}
end
function sdks()
    return _sdks
end
function sdk_dir(arch, custom_dir)
    if custom_dir then
        if not path.is_absolute(custom_dir) then
            custom_dir = path.absolute(custom_dir, os.projectdir())
        end
    else
        custom_dir = path.join(os.scriptdir(), 'SDKs/')
    end
    return path.join(custom_dir, arch)
end

function get_or_create_sdk_dir(arch, custom_dir)
    local dir = sdk_dir(arch, custom_dir)
    local lib = import('lib')
    lib.mkdirs(dir)
    return dir
end
