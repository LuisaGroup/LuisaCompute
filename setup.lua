local rootdir = {rootdir = path.join(os.scriptdir(), 'scripts')}
local packages = import("packages", rootdir)
local find_sdk = import("find_sdk", rootdir)
local lib = import("lib", rootdir)
function unzip(name, custom_dir, out_dir)
    local sdk = packages.sdks()[name]
    local dir = packages.sdk_dir(os.arch(), custom_dir)
    find_sdk.unzip_sdk(sdk['name'], dir, out_dir)
end
function main(custom_dir, decompress_dir)
    if os.is_host("windows") then
        find_sdk.install_sdk('dx_sdk', custom_dir)
    end
    if decompress_dir then
        lib.mkdirs(decompress_dir)
        unzip('dx_sdk', custom_dir, decompress_dir)
    end
end