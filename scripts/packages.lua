local _sdks = {
    dx_sdk = {
        sha256 = "f6550326041f947b8f5b9a9fdcfa1dacd8345335a71f589c85b6e20fc9ed1b17",
        name = 'dx_sdk.zip',
        -- address = 
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
        custom_dir = path.join(os.projectdir(), 'SDKs/')
    end
    return path.join(custom_dir, arch)
end

function get_or_create_sdk_dir(arch, custom_dir)
    local dir = sdk_dir(arch, custom_dir)
    local lib = import('lib')
    lib.mkdirs(dir)
    return dir
end
