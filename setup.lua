local rootdir = {rootdir = path.join(os.scriptdir(), 'scripts')}
local packages = import("packages", rootdir)
local find_sdk = import("find_sdk", rootdir)
function unzip(name, custom_dir)
    local sdk = packages.sdks()[name]
    local dir = packages.sdk_dir(os.arch(), custom_dir)
    find_sdk.unzip_sdk(sdk['name'], dir, dir)
end
function main(custom_dir, decompress)
    if os.is_host("windows") then
        find_sdk.install_sdk('dx_sdk', custom_dir)
    end
    if decompress then
        unzip('dx_sdk', custom_dir)  
    end
end