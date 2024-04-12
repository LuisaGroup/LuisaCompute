local packages = import("packages", {rootdir = 'scripts'})
local find_sdk = import("find_sdk", {rootdir = 'scripts'})
function main()
    if os.is_host("windows") then
        find_sdk.install_sdk('dx_sdk')
    end
end