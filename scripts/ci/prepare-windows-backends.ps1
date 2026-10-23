param(
    [string]$DependencyRoot = "$PSScriptRoot/../../.ci/deps",
    [string]$RocmRoot = '',
    [string]$EmbreeRoot = '',
    [string]$TbbRoot = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$DependencyRoot = [IO.Path]::GetFullPath($DependencyRoot)
New-Item -ItemType Directory -Force $DependencyRoot | Out-Null

function Expand-VerifiedPackage($Url, $Hash, $Destination, $Marker) {
    if (Test-Path -LiteralPath (Join-Path $Destination $Marker)) { return }
    $archive = Join-Path $DependencyRoot ([IO.Path]::GetFileName($Url))
    if (!(Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest -Uri $Url -OutFile $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Hash) {
        throw "SHA256 mismatch: $archive"
    }
    New-Item -ItemType Directory -Force $Destination | Out-Null
    if ($archive.EndsWith('.zip')) {
        Expand-Archive -LiteralPath $archive -DestinationPath $Destination -Force
    } else {
        & tar -xf $archive -C $Destination
        if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $archive" }
    }
    if (!(Test-Path -LiteralPath (Join-Path $Destination $Marker))) {
        throw "Package layout changed: $Destination/$Marker"
    }
}

# Official archives are pinned to the bytes used by the local Windows build.
# ROCm supplies matching AMDGPU/SPIRV/X86 LLVM headers and libraries as well as HIP.
if (!$RocmRoot) {
    $RocmRoot = Join-Path $DependencyRoot 'rocm-10.1.0'
    Expand-VerifiedPackage `
        'https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx120X-all-10.1.0.tar.gz' `
        'E8D5ACD522AA106D685485707E5085D491EAD7D0D84A79EF74DC5995941403B2' `
        $RocmRoot 'lib/llvm/lib/cmake/llvm/LLVMConfig.cmake'
}
if (!$EmbreeRoot) {
    $destination = Join-Path $DependencyRoot 'embree-4.4.1'
    Expand-VerifiedPackage `
        'https://github.com/RenderKit/embree/releases/download/v4.4.1/embree-4.4.1.x64.windows.zip' `
        '5AA1FA3161A720F9C610B06C652B80F421CB2B1FE8221FDF92B74157336FDF8F' `
        $destination 'embree-4.4.1.x64.windows/lib/cmake/embree-4.4.1/embree-config.cmake'
    $EmbreeRoot = Join-Path $destination 'embree-4.4.1.x64.windows'
}
if (!$TbbRoot) {
    $destination = Join-Path $DependencyRoot 'tbb-2023.1.0'
    Expand-VerifiedPackage `
        'https://github.com/uxlfoundation/oneTBB/releases/download/v2023.1.0/oneapi-tbb-2023.1.0-win.zip' `
        'CF6EE0C600FCB5C3A9B65E3E6E4781669D06F1BB1E37970D145FCDE08EED8DA9' `
        $destination 'oneapi-tbb-2023.1.0/lib/cmake/tbb/TBBConfig.cmake'
    $TbbRoot = Join-Path $destination 'oneapi-tbb-2023.1.0'
}
$RocmRoot = (Resolve-Path -LiteralPath $RocmRoot).Path
# hipcc still needs a space-free compiler root on some Windows SDK releases.
if ($RocmRoot.Contains(' ')) {
    $RocmRoot = (New-Object -ComObject Scripting.FileSystemObject).GetFolder($RocmRoot).ShortPath
    if ($RocmRoot.Contains(' ')) { throw 'Use a space-free ROCm installation path.' }
}
$llvm = "$RocmRoot/lib/llvm/lib/cmake/llvm".Replace('\', '/')
$wrapper = Join-Path $DependencyRoot 'llvm-rocm-config'
New-Item -ItemType Directory -Force $wrapper | Out-Null
# The pinned SDK omits CRT metadata. Verify the actual library before supplying it.
$directives = & "$RocmRoot/lib/llvm/bin/llvm-readobj.exe" --coff-directives "$RocmRoot/lib/llvm/lib/LLVMCore.lib"
if ($LASTEXITCODE -ne 0 -or !($directives -match 'RuntimeLibrary=MD_DynamicRelease')) {
    throw 'Cannot verify the ROCm LLVM Release DLL CRT ABI.'
}
@"
include("$llvm/LLVMConfig.cmake")
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
"@ | Set-Content -LiteralPath "$wrapper/LLVMConfig.cmake"
Copy-Item -LiteralPath "$llvm/LLVMConfigVersion.cmake" -Destination $wrapper -Force
$result = [ordered]@{
    rocm = $RocmRoot
    llvm = $wrapper
    embree = (Resolve-Path -LiteralPath $EmbreeRoot).Path
    tbb = (Resolve-Path -LiteralPath $TbbRoot).Path
}
$result | ConvertTo-Json | Set-Content -LiteralPath "$DependencyRoot/environment.json"
$result
