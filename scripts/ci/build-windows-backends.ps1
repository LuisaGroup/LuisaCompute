param(
    [string]$EnvironmentFile = "$PSScriptRoot/../../.ci/deps/environment.json",
    [string]$BuildDirectory = 'build-backends',
    [bool]$Fallback = $true,
    [bool]$Simd = $true,
    [bool]$Hip = $false,
    [bool]$Vulkan = $false,
    [bool]$Dx = $false,
    [bool]$LlvmSpirv = $false,
    [bool]$RunGpuTests = $false,
    [int]$Parallel = 4
)
$ErrorActionPreference = 'Stop'
$envs = Get-Content -LiteralPath $EnvironmentFile -Raw | ConvertFrom-Json
$env:HIP_PATH = $envs.rocm
$env:ROCM_PATH = $envs.rocm
$env:HIP_CLANG_PATH = "$($envs.rocm)/lib/llvm/bin"
$env:HIP_DEVICE_LIB_PATH = "$($envs.rocm)/lib/llvm/amdgcn/bitcode"
$env:HIP_PLATFORM = 'amd'
$env:PATH = "$($envs.rocm)/bin;$env:HIP_CLANG_PATH;$env:PATH"
if ($LlvmSpirv -and $RunGpuTests) { throw 'The experimental LLVM SPIR-V route lacks runtime buffer descriptors; use native XIR for GPU tests.' }
if ($LlvmSpirv -and !$Vulkan) { throw 'LLVM SPIR-V requires the Vulkan backend.' }
function OnOff([bool]$Value) { if ($Value) { 'ON' } else { 'OFF' } }
$configure = @(
    '-S', '.', '-B', $BuildDirectory, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=clang-cl',
    '-DCMAKE_CXX_COMPILER=clang-cl', '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL',
    "-DLLVM_DIR=$($envs.llvm)",
    "-DCMAKE_PREFIX_PATH=$($envs.rocm);$($envs.rocm)/lib/rocm_sysdeps;$($envs.embree)",
    "-DTBB_DIR=$($envs.tbb)/lib/cmake/tbb",
    "-DZLIB_LIBRARY=$($envs.rocm)/lib/rocm_sysdeps/lib/zs.lib",
    "-DLUISA_COMPUTE_ENABLE_FALLBACK=$(OnOff $Fallback)",
    "-DLUISA_COMPUTE_ENABLE_SIMD=$(OnOff $Simd)",
    "-DLUISA_COMPUTE_ENABLE_HIP=$(OnOff $Hip)",
    "-DLUISA_COMPUTE_ENABLE_VULKAN=$(OnOff $Vulkan)",
    "-DLUISA_COMPUTE_ENABLE_DX=$(OnOff $Dx)",
    "-DLUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV=$(OnOff $LlvmSpirv)",
    "-DLUISA_COMPUTE_ENABLE_VK_XIR_SPIRV=$(OnOff ($Vulkan -and !$LlvmSpirv))",
    '-DLUISA_COMPUTE_ENABLE_CUDA=OFF', '-DLUISA_COMPUTE_ENABLE_METAL=OFF',
    '-DLUISA_COMPUTE_ENABLE_DX_CUDA_INTEROP=OFF',
    '-DLUISA_COMPUTE_ENABLE_VK_CUDA_INTEROP=OFF',
    '-DLUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES=OFF',
    '-DLUISA_COMPUTE_BUILD_TESTS=ON', '-DHIPRT_GPU_ARCHS=gfx1201',
    "-DHIP_OCML_FILE=$env:HIP_DEVICE_LIB_PATH/ocml.bc",
    "-DHIP_OCKL_FILE=$env:HIP_DEVICE_LIB_PATH/ockl.bc"
)
New-Item -ItemType Directory -Force $BuildDirectory | Out-Null
& cmake @configure 2>&1 | Tee-Object "$BuildDirectory/configure.log"
if ($LASTEXITCODE -ne 0) { throw 'Configure failed.' }
& cmake --build $BuildDirectory --parallel $Parallel 2>&1 | Tee-Object "$BuildDirectory/build.log"
if ($LASTEXITCODE -ne 0) { throw 'Full build failed; no tests were run.' }
$bin = Join-Path $BuildDirectory 'bin'
if ($Fallback -or $Simd) {
    Copy-Item "$($envs.embree)/bin/embree4.dll" $bin -Force
    Copy-Item "$($envs.tbb)/redist/intel64/vc14/tbb12.dll" $bin -Force
}
if ($Hip) {
    Copy-Item "$($envs.rocm)/bin/amdhip64_7.dll" $bin -Force
    Copy-Item "$($envs.rocm)/bin/rocm_kpack.dll" $bin -Force
    Copy-Item "$($envs.rocm)/bin/hiprtc0716.dll" $bin -Force
    Copy-Item "$($envs.rocm)/bin/hiprtc-builtins0716.dll" $bin -Force
}
& ctest --test-dir $BuildDirectory --output-on-failure --no-tests=error --timeout 120 `
    -L '^(unit_core|unit_xir|unit_fallback|unit_simd|unit_llvm|hlsl)$' `
    --output-junit cpu-tests.xml 2>&1 | Tee-Object "$BuildDirectory/cpu-tests.log"
if ($LASTEXITCODE -ne 0) { throw 'CPU tests failed.' }
if ($Simd) {
    & ctest --test-dir $BuildDirectory --output-on-failure --no-tests=error --timeout 120 `
        -R '^test_simd_runtime_widths$' --output-junit simd-width-tests.xml `
        2>&1 | Tee-Object "$BuildDirectory/simd-width-tests.log"
    if ($LASTEXITCODE -ne 0) { throw 'SIMD width tests failed.' }
}
if ($RunGpuTests) {
    $backends = @()
    if ($Hip) { $backends += 'hip' }
    if ($Vulkan) { $backends += 'vk' }
    if ($Dx) { $backends += 'dx' }
    if (!$backends) { throw 'GPU tests requested without a GPU backend.' }
    foreach ($backend in $backends) {
        foreach ($test in @('test_buffer', 'test_hip_codegen_arithmetic', 'test_atomic')) {
            & "$bin/$test.exe" $backend 2>&1 | Tee-Object "$BuildDirectory/$test-$backend.log"
            if ($LASTEXITCODE -ne 0) { throw "$test failed on $backend" }
        }
    }
}
