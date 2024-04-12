local _sdks = {
    dx_sdk = {
        sha256 = "f6550326041f947b8f5b9a9fdcfa1dacd8345335a71f589c85b6e20fc9ed1b17",
        name = 'dx_sdk.zip',
    }
}

function sdk_address()
    return 'https://github.com/LuisaGroup/SDKs/releases/download/sdk/'
end

function sdks()
    return _sdks
end
function sdk_dir(arch)
    return path.join(os.projectdir(), 'SDKs/', arch)
end

function get_or_create_sdk_dir(arch)
    local dir = sdk_dir(arch)
    local lib = import('lib')
    lib.mkdirs(dir)
    return dir
end
