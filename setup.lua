local rootdir = {rootdir = path.join(os.scriptdir(), 'scripts')}
local packages = import("packages", rootdir)
local find_sdk = import("find_sdk", rootdir)
function main(...)
    local custom_dir
    for _,v in ipairs(...) do
        custom_dir = v
    end
    if os.is_host("windows") then
        find_sdk.install_sdk('dx_sdk', custom_dir)
    end
end