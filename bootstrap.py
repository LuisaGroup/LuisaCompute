import multiprocessing
import os
import sys
import json
import shutil
from subprocess import Popen, call, DEVNULL, check_output
from typing import List

ALL_FEATURES = ['dsl', 'python', 'gui', 'cuda', 'cpu', 'remote', 'dx', 'metal', 'vulkan', 'tests']
ALL_DEPENDENCIES = ['rust', 'ninja', 'xmake', 'cmake']
ALL_CMAKE_DEPENDENCIES = ['ninja', 'cmake', 'rust']
ALL_XMAKE_DEPENDENCIES = ['xmake', 'rust']

DEPS_DIR = '.deps'
DOWNLOAD_DIR = f'{DEPS_DIR}/downloads'


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'


def check_rust():
    try:
        ret = call(['rustc', '--version'], stdout=DEVNULL, stderr=DEVNULL)
        return ret == 0
    except FileNotFoundError:
        # try the default install location
        # %USERPROFILE%\.cargo\bin
        try:
            ret = call([os.path.expanduser('~/.cargo/bin/rustc'), '--version'], stdout=DEVNULL, stderr=DEVNULL)
            return ret == 0
        except FileNotFoundError:
            pass
    return False


def check_cmake(cmake_exe):
    try:
        ret = call([cmake_exe, '--version'], stdout=DEVNULL, stderr=DEVNULL)
        return ret == 0
    except FileNotFoundError:
        return False


def check_xmake(xmake_exe):
    try:
        ret = call([xmake_exe, '--version'], stdout=DEVNULL, stderr=DEVNULL)
        return ret == 0
    except FileNotFoundError:
        return False


def check_nvcc():
    try:
        ret = call(['nvcc', '--version'], stdout=DEVNULL, stderr=DEVNULL)
        return ret == 0
    except FileNotFoundError:
        return False


def check_ninja(ninja_exe):
    try:
        ret = call([ninja_exe, '--version'], stdout=DEVNULL, stderr=DEVNULL)
        return ret == 0
    except FileNotFoundError:
        return False


def check_vk():
    try:
        ret = call(['vulkaninfo'], stdout=DEVNULL, stderr=DEVNULL)
        return ret == 0
    except FileNotFoundError:
        return False


def print_red(msg, *args, **kwargs):
    print(Colors.RED + msg.format(*args, **kwargs) + Colors.END, *args, **kwargs)


print_missing_rust_warning = False


def missing_rust_warning():
    print_red("Warning: Rust is required for future releases.", file=sys.stderr)
    print_red('We strongly recommend you to install Rust **now** to prevent future breakage.', file=sys.stderr)
    print_red("Please install Rust manually or by running `python bootstrap.py -i rust`.", file=sys.stderr)
    print_red('Features require Rust:', file=sys.stderr)
    print_red('  - CPU backend', file=sys.stderr)
    print_red('  - Remote backend', file=sys.stderr)
    print_red('  - IR module', file=sys.stderr)
    print_red('  - Automatic differentiation', file=sys.stderr)


def get_available_features():
    global print_missing_rust_warning
    # CPU and Remote are always enabled
    features = ['dsl', 'python', 'gui', 'tests']
    if check_rust():
        features.append('cpu')
        features.append('remote')
    else:
        print_missing_rust_warning = True

    # enable DirectX on Windows by default
    if sys.platform == 'win32':
        features.append('dx')
    # enable Metal on macOS by default
    if sys.platform == 'darwin':
        features.append('metal')
    # enable CUDA if available
    try:
        if 'CUDA_PATH' in os.environ or check_nvcc():
            features.append('cuda')
    except FileNotFoundError:
        pass
    try:
        if 'VULKAN_SDK' in os.environ or 'VK_SDK_PATH' in os.environ or check_vk():
            features.append('vulkan')
    except FileNotFoundError:
        pass
    return features


def get_default_toolchain():
    if sys.platform == 'win32':
        return 'msvc'
    elif sys.platform == 'darwin':
        return 'llvm'
    elif sys.platform == 'linux':
        return 'gcc'
    else:
        raise ValueError(f'Unknown platform: {sys.platform}')


def download_file(url: str, name: str):
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    output_path = f'{DOWNLOAD_DIR}/{name}'
    import urllib.request
    print(f'Downloading "{url}" to "{output_path}"...')
    max_tries = 3
    for i in range(max_tries):
        try:
            urllib.request.urlretrieve(url, output_path)
            return output_path
        except Exception as e:
            print(f'Failed to download "{url}": {e}. Retrying ({i + 2}/{max_tries})...')


def unzip_file(in_path: str, out_path: str):
    if in_path.lower().endswith('.zip'):
        import zipfile
        print(f'Unzipping "{in_path}" to "{out_path}"...')
        with zipfile.ZipFile(in_path, 'r') as zip_ref:
            zip_ref.extractall(out_path)
    elif in_path.lower().endswith('.tar.gz'):
        import tarfile
        print(f'Unzipping "{in_path}" to "{out_path}"...')
        with tarfile.open(in_path, 'r:gz') as tar_ref:
            tar_ref.extractall(out_path)
    else:
        raise ValueError(f'Unknown file type: {in_path}')


def install_ninja(skip_installed):
    if skip_installed and os.path.exists(f'{DEPS_DIR}/ninja_bin'):
        pass
    if sys.platform == 'win32':
        ninja_platform = 'win'
    elif sys.platform == 'linux':
        ninja_platform = 'linux'
    elif sys.platform == 'darwin':
        ninja_platform = 'mac'
    else:
        raise ValueError(f'Unknown platform: {sys.platform}')
    ninja_version = "1.11.1"
    ninja_file = f"ninja-{ninja_platform}.zip"
    ninja_url = f"https://github.com/ninja-build/ninja/releases/download/v{ninja_version}/{ninja_file}"
    zip_path = download_file(ninja_url, ninja_file)
    unzip_file(zip_path, DEPS_DIR)
    if sys.platform == 'win32':
        ninja_bin = f"{DEPS_DIR}/ninja.exe"
    else:
        ninja_bin = f"{DEPS_DIR}/ninja"
        call(['chmod', '+x', ninja_bin])
    with open(f"{DEPS_DIR}/ninja_bin", 'w') as f:
        f.write(ninja_bin)


def install_xmake(skip_installed):
    if skip_installed and os.path.exists(f"{DEPS_DIR}/xmake_bin"):
        return
    ps_file = download_file("https://fastly.jsdelivr.net/gh/xmake-io/xmake@master/scripts/get.ps1", "get_xmake.ps1")
    with open(ps_file, 'r') as f:
        ps_script = f.read()
    # find the latest version $LastRelease = "value"
    last_release = ps_script.split('$LastRelease = "')[1].split('"')[0]
    if sys.platform == 'win32':
        file_name = "xmake-master.win64.zip"
        xmake_url = f"https://github.com/xmake-io/xmake/releases/download/{last_release}/{file_name}"
        xmake_file = download_file(xmake_url, file_name)
        unzip_file(xmake_file, DEPS_DIR)
        with open(f"{DEPS_DIR}/xmake_bin", "w") as f:
            f.write(f"{DEPS_DIR}/xmake/xmake.exe")
    else:
        file_name = f"xmake-master.tar.gz"
        xmake_url = f"https://github.com/xmake-io/xmake/releases/download/{last_release}/{file_name}"
        xmake_file = download_file(xmake_url, file_name)
        unzip_file(xmake_file, DEPS_DIR)
        output_dir = f'{DEPS_DIR}/xmake-{last_release.replace("v", "")}'
        call(['sh', './configure'], cwd=output_dir)
        call(['make', f'-j{multiprocessing.cpu_count()}'], cwd=output_dir)
        call(['make', 'install', 'PREFIX=.'], cwd=output_dir)
        with open(f"{DEPS_DIR}/xmake_bin", "w") as f:
            f.write(f"{output_dir}/bin/xmake")


def install_cmake(skip_installed):
    if skip_installed and os.path.exists(f'{DEPS_DIR}/cmake_bin'):
        return
    if sys.platform == 'win32':
        cmake_platform = 'windows'
        cmake_suffix = 'zip'
        cmake_arch = 'x86_64'
    elif sys.platform == 'linux':
        cmake_platform = 'linux'
        cmake_suffix = 'tar.gz'
        cmake_arch = 'x86_64'
    elif sys.platform == 'darwin':
        cmake_platform = 'macos'
        cmake_suffix = 'tar.gz'
        cmake_arch = 'universal'
    else:
        raise ValueError(f'Unknown platform: {sys.platform}')
    cmake_version = "3.26.3"
    cmake_file = f"cmake-{cmake_version}-{cmake_platform}-{cmake_arch}"
    cmake_url = f"https://github.com/Kitware/CMake/releases/download/v{cmake_version}/{cmake_file}.{cmake_suffix}"
    zip_path = download_file(cmake_url, f"{cmake_file}.{cmake_suffix}")
    unzip_file(zip_path, DEPS_DIR)
    # symlink the executable
    with open(f'{DEPS_DIR}/cmake_bin', 'w') as f:
        if sys.platform == 'win32':
            f.write(f"{DEPS_DIR}/{cmake_file}/bin/cmake.exe")
        elif sys.platform == 'darwin':
            f.write(f"{DEPS_DIR}/{cmake_file}/CMake.app/Contents/bin/cmake")
        else:  # linux
            call(['chmod', '+x', f"{DEPS_DIR}/{cmake_file}/bin/cmake"])
            f.write(f"{DEPS_DIR}/{cmake_file}/bin/cmake")


def install_rust(skip_installed):
    if skip_installed and check_rust():
        return
    if sys.platform == 'win32':
        # download https://static.rust-lang.org/rustup/dist/i686-pc-windows-gnu/rustup-init.exe
        rustup_init_exe = download_file('https://static.rust-lang.org/rustup/dist/i686-pc-windows-gnu/rustup-init.exe',
                                        'rustup-init.exe')
        call([rustup_init_exe, '-y'])
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        os.system('curl https://sh.rustup.rs -sSf | sh -s -- -y')
    else:
        raise ValueError(f'Unknown platform: {sys.platform}')


def install_deps(deps, skip_installed):
    if deps:
        print(f'Installing dependencies: {", ".join(deps)}')
        for dep in deps:
            if dep == 'rust':
                install_rust(skip_installed)
            elif dep == 'ninja':
                install_ninja(skip_installed)
            elif dep == 'cmake':
                install_cmake(skip_installed)
            elif dep == 'xmake':
                install_xmake(skip_installed)


def get_config(parsed_args):
    config = {
        'cmake_args': [],
        'xmake_args': [],
        'output': 'build',
    }
    # check if config.json exists
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config.update(json.load(f))
    config['build_system'] = parsed_args['build_system']
    if "toolchain" in parsed_args:
        config['toolchain'] = parsed_args['toolchain']
    if "output" in parsed_args:
        config['output'] = parsed_args['output']
    if "features" in parsed_args:
        config['features'] = parsed_args['features']

    def find_program(name):
        if os.path.exists(f'{DEPS_DIR}/{name}_bin'):
            with open(f'{DEPS_DIR}/{name}_bin', 'r') as f:
                return os.path.abspath(f.read().strip())
        else:
            return name

    if config["build_system"] == "cmake":
        if "additional_args" in parsed_args:
            config['cmake_args'] += parsed_args['additional_args']
        if "cmake_exe" not in config or config["cmake_exe"] == "cmake":
            config['cmake_exe'] = find_program("cmake")
        if "ninja_exe" not in config or config["ninja_exe"] == "ninja":
            config['ninja_exe'] = find_program("ninja")
    elif config["build_system"] == "xmake":
        if "additional_args" in parsed_args:
            config['xmake_args'] += parsed_args['additional_args']
        if "xmake_exe" not in config or config["xmake_exe"] == "xmake":
            config['xmake_exe'] = find_program("xmake")

    available_features = get_available_features()
    if "features" not in config:
        config['features'] = available_features
    else:
        config['features'] = [f for f in config['features'] if f in available_features]

    return config


def print_help():
    print('Usage: python bootstrap.py [build system] [mode] [options]')
    print('Build system:')
    print('  cmake                  Use CMake')
    print('  xmake                  Use xmake')
    print('Mode (required only when "--config | -c" or "--build | -b" is specified):')
    print('  release                Release mode (default)')
    print('  debug                  Debug mode')
    print('  reldbg                 Release with debug infomation mode')
    print('Options:')
    print('  --ignore-submodules    Skip the submodule integrity check')
    print('  --config    | -c       Configure build system')
    print(
        '  --build     | -b [N]   Build with N threads (default: number of CPU cores; this options implies "--config | -c")')
    print('  --toolchain | -t [toolchain]      Configure toolchain (effective only',
          'when "--config | -c" or "--build | -b" is specified)')
    print('      Toolchains:')
    print('          msvc[-version]     Use MSVC toolchain (default on Windows; available on Windows only)')
    print('          llvm[-version]     Use LLVM toolchain (default on macOS; available on Windows, macOS, and Linux)')
    print('          gcc[-version]      Use GCC toolchain (default on Linux; available on Linux only)')
    print('  --features  | -f [[no-]features]  Add/remove features')
    print('      Features:')
    print('          all                Enable all features listed below that are detected available')
    print('          [no-]dsl           Enable (disable) C++ DSL support')
    print('          [no-]python        Enable (disable) Python support')
    print('          [no-]gui           Enable (disable) GUI support')
    print('          [no-]cuda          Enable (disable) CUDA backend')
    print('          [no-]cpu           Enable (disable) CPU backend')
    print('          [no-]remote        Enable (disable) remote backend')
    print('          [no-]dx            Enable (disable) DirectX backend')
    print('          [no-]metal         Enable (disable) Metal backend')
    print('          [no-]vulkan        Enable (disable) Vulkan backend')
    print('          [no-]tests         Enable (disable) tests')
    print('  --clean   | -C         Clean build directory')
    print('  --install | -i [deps]  Install dependencies')
    print('      Dependencies:')
    print('          all                Install all dependencies required by CMake and XMake builds',
          '(' + ', '.join(ALL_DEPENDENCIES) + ')')
    print('          all-cmake          Install all dependencies required by CMake builds',
          '(' + ', '.join(ALL_CMAKE_DEPENDENCIES) + ')')
    print('          all-xmake          Install all dependencies required by XMake builds',
          '(' + ', '.join(ALL_XMAKE_DEPENDENCIES) + ')')
    print('          rust               Install Rust toolchain')
    print('          cmake              Install CMake')
    print('          xmake              Install xmake')
    print('          ninja              Install Ninja')
    print('  --skip-installed | -s      Skip installing dependencies that are already installed')
    print('  --output  | -o [folder]    Path to output directory (default: build)')
    print('  -- [args]              Pass arguments to build system')


def dump_cmake_options(config: dict):
    with open("scripts/options.cmake.template") as f:
        options = f.read()
    for feature in config['features']:
        options = options.replace(f'[[feature_{feature}]]', 'ON')
    for feature in ALL_FEATURES:
        if feature not in config['features']:
            options = options.replace(f'[[feature_{feature}]]', 'OFF')
    with open("scripts/options.cmake", 'w') as f:
        f.write(options)


def dump_xmake_options(config: dict):
    xmake_var = config.get("xmake_exe")
    if xmake_var is None:
        xmake_var = "xmake"
    cmd = f"{xmake_var} lua scripts/write_options.lua"
    if "toolchain" in config:
        cmd += " toolchain=" + config["toolchain"]
    features = config['features']
    def add_feature(option: str, feature: str):
        if feature in features:
            return f" {option}=true"
        else:
            return f" {option}=false"
    if features:
        cmd += add_feature("enable_dsl", "dsl")
        cmd += add_feature("python", "python")
        cmd += add_feature("enable_gui", "gui")
        cmd += add_feature("dx_backend", "dx")
        cmd += add_feature("cuda_backend", "cuda")
        cmd += add_feature("metal_backend", "metal")
        # cmd += add_feature("remote_backend", "remote")
        cmd += add_feature("cpu_backend", "cpu")
        cmd += add_feature("enable_ir", "cpu")
    os.system(cmd)


def dump_build_system_options(config: dict):
    build_sys = config['build_system']
    if build_sys == 'cmake':
        dump_cmake_options(config)
    elif build_sys == 'xmake':
        dump_xmake_options(config)
    else:
        raise ValueError(f'Unknown build system: {build_sys}')


def find_msvc(version, pattern):
    if os.path.exists(f'{DEPS_DIR}/vswhere_bin'):
        with open(f'{DEPS_DIR}/vswhere_bin', 'r') as f:
            vswhere_exe = f.read().strip()
    else:
        vswhere_version = '3.1.1'
        vswhere_url = f'https://github.com/microsoft/vswhere/releases/download/{vswhere_version}/vswhere.exe'
        vswhere_exe = download_file(vswhere_url, 'vswhere.exe')
        with open(f'{DEPS_DIR}/vswhere_bin', 'w') as f:
            f.write(vswhere_exe)

    if version is None:
        version_args = []
    elif version == 2019 or version == 16:
        version_args = ['-version', '[16.0,17.0)']
    elif version == 2022 or version == 17:
        version_args = ['-version', '[17.0,18.0)']
    else:
        if version != 0:
            print_red(f'Unsupported MSVC version: {version}. Using latest version.')
        version_args = ['-latest']

    vswhere_args = [vswhere_exe, '-format', 'json', '-utf8',
                    '-nologo', '-sort', '-products', '*', '-find', pattern] + version_args

    try:
        output = check_output(vswhere_args)
    except Exception as e:
        print_red(f'Failed to find MSVC: {e}')
        return None

    def parse_msvc_version(path):
        try:
            path = path.replace('\\', '/').lower().replace(pattern, '').split('/')[-1]
            return [int(x) for x in path.split('.')]
        except:
            return [0, 0, 0]

    output = json.loads(output.decode('utf-8'))
    if not output:
        print_red('Failed to find MSVC')
        return None

    sorted_output = [x.replace('\\', '/') for x in sorted(output, key=lambda x: parse_msvc_version(x))]
    if version is None:
        return sorted_output
    else:
        return sorted_output[-1]


def find_llvm(version):
    found_clang = {}
    candidate_paths = os.environ.get('PATH', '')
    if sys.platform == 'win32':
        candidate_paths = [p.replace('\\', '/').strip() for p in candidate_paths.split(';') if p.strip()]
        clang_from_prog = 'C:/Program Files/LLVM/bin'
        if os.path.exists(clang_from_prog) and clang_from_prog not in candidate_paths:
            candidate_paths.append(clang_from_prog)
        clang_from_vs = find_msvc(None, '**/VC/Tools/Llvm/x64/bin/clang-cl.exe')
        if clang_from_vs:
            candidate_paths += [os.path.dirname(x) for x in clang_from_vs]
    else:
        candidate_paths = [p.strip() for p in candidate_paths.split(':') if p.strip()]

    def get_llvm_version(clang_exe):
        try:
            output = check_output([clang_exe, '--version']).decode('utf-8')
            index = output.index('clang version')
            output = output[index:].split('\n')[0].strip()
            v = [int(x) for x in output.split(' ')[2].split('.')]
            return v if v[0] == version or version == 0 else None
        except:
            return None

    for path in candidate_paths:
        clang_exe = f'{path}/clang'
        clang_version = get_llvm_version(clang_exe)
        if clang_version:
            found_clang[clang_exe] = clang_version

    if not found_clang:
        print_red('Failed to find LLVM')
        return None

    clang_exe = sorted(found_clang, key=lambda x: found_clang[x])
    for c in clang_exe:
        print(f'Found LLVM: {c} (version = {".".join([str(x) for x in found_clang[c]])})')
    return os.path.dirname(clang_exe[-1])


def prepare_msvc_environment(toolchain_version: int):
    vcvars_bat = find_msvc(toolchain_version, '**/Auxiliary/Build/vcvars64.bat')
    if not vcvars_bat:
        return None
    try:
        env_vars = check_output([vcvars_bat, '&&', sys.executable, '-c',
                                 'import os; import json; print("[[ENVIRON]] =", json.dumps(dict(os.environ)))'])
    except Exception as e:
        print_red(f'Failed to dump environment variables: {e}')
        return None
    env_vars = env_vars.decode('utf-8').split('[[ENVIRON]] = ')[1]
    env_vars = json.loads(env_vars)
    os.environ.update(env_vars)


def prepare_toolchain_environment(build_config, build_system):
    toolchain = build_config['toolchain']
    version = build_config['toolchain_version']
    if toolchain == 'msvc':
        if build_system == "cmake":
            prepare_msvc_environment(version)
    elif toolchain == 'llvm':
        pass
    elif toolchain == 'gcc':
        pass
    else:
        raise ValueError(f'Unknown toolchain: {toolchain}-{version}')


def build_system_config_args_cmake(config: dict, mode: str, toolchain: str, toolchain_version: int):
    args = config['cmake_args']
    cmake_mode = {
        'debug': 'Debug',
        'release': 'Release',
        'reldbg': 'RelWithDebInfo'
    }
    args.append(f'-DCMAKE_BUILD_TYPE={cmake_mode[mode]}')
    if toolchain == 'msvc':
        args.append(f'-DCMAKE_C_COMPILER=cl.exe')
        args.append(f'-DCMAKE_CXX_COMPILER=cl.exe')
    elif toolchain == 'llvm':
        llvm_bin = find_llvm(toolchain_version).replace('\\', '/')
        if sys.platform == 'win32':
            args.append(f"-DCMAKE_C_COMPILER={llvm_bin}/clang-cl.exe")
            args.append(f"-DCMAKE_CXX_COMPILER={llvm_bin}/clang-cl.exe")
        else:
            args.append(f"-DCMAKE_C_COMPILER={llvm_bin}/clang")
            args.append(f"-DCMAKE_CXX_COMPILER={llvm_bin}/clang++")
    elif toolchain == 'gcc':
        if toolchain_version == 0:
            gcc_exe = "gcc"
            gxx_exe = "g++"
        else:
            gcc_exe = f"gcc-{toolchain_version}"
            gxx_exe = f"g++-{toolchain_version}"
        try:
            check_output([gcc_exe, '--version'])
            check_output([gxx_exe, '--version'])
        except:
            print_red(f'Failed to find GCC-{toolchain_version}')
            return None
        args.append(f"-DCMAKE_C_COMPILER={gcc_exe}")
        args.append(f"-DCMAKE_CXX_COMPILER={gxx_exe}")
    return args


def build_system_config_args_xmake(config: dict, mode: str, toolchain: str, toolchain_version: int):
    args = config['xmake_args']
    xmake_mode = {
        'debug': 'debug',
        'release': 'release',
        'reldbg': 'releasedbg'
    }
    args.append('-m')
    args.append(xmake_mode[mode])
    args.append("-c")
    return args


def build_system_config_args(config, mode, toolchain, toolchain_version):
    if config['build_system'] == 'cmake':
        return build_system_config_args_cmake(config, mode, toolchain, toolchain_version)
    elif config['build_system'] == 'xmake':
        return build_system_config_args_xmake(config, mode, toolchain, toolchain_version)
    else:
        raise ValueError(f'Unknown build system: {config["build_system"]}')


def get_build_config(build_dir, parsed_args):
    build_config = {
        'mode': 'release',
        'toolchain_version': 0,
        'jobs': multiprocessing.cpu_count(),
    }

    if os.path.exists(os.path.join(build_dir, 'build_config.json')):
        with open(os.path.join(build_dir, 'build_config.json')) as f:
            build_config.update(json.load(f))

    if 'mode' in parsed_args:
        build_config['mode'] = parsed_args['mode']
    if 'build' in parsed_args:
        build_config['jobs'] = parsed_args['build']

    toolchain = get_default_toolchain()
    if 'toolchain' in parsed_args:
        toolchain = parsed_args['toolchain']
        build_config.pop('toolchain_version')
    elif 'toolchain' in build_config:
        toolchain = build_config['toolchain']
    toolchain_version = 0
    if 'toolchain_version' in build_config:
        toolchain_version = build_config['toolchain_version']
    if '-' in toolchain:
        toolchain, toolchain_version = toolchain.split('-')
        toolchain_version = int(toolchain_version)
    build_config['toolchain'] = toolchain
    build_config['toolchain_version'] = toolchain_version
    return build_config


def init_submodule():
    import configparser
    git_submodules = configparser.ConfigParser()
    git_submodules.read(".gitmodules")
    git_submodules = {git_submodules[sec]['path']: git_submodules[sec]['url']
                      for sec in git_submodules.sections()}

    if os.path.exists('.git'):
        for i in range(3):
            if os.system('git submodule update --init --recursive') == 0:
                break
    else:
        print_red('Warning: this repository seems not to be a git repository.')
        print_red('It is likely you have download the zip file from GitHub.')
        print_red('We will try to fix the missing submodules but we strongly')
        print_red('recommend that you **clone** the repository with --recursive option.')
        for path, url in git_submodules.items():
            if not os.path.exists(f'{path}/.git'):
                print(f'Cloning submodule {path} from {url}')
                for i in range(3):
                    if os.system(f'git clone --recursive {url} {path}') == 0:
                        break
            else:
                print(f'Updating submodule {path}')
                for i in range(3):
                    if call(['git', 'pull'], cwd=path) == 0:
                        break
                for i in range(3):
                    if call(['git', 'submodule', 'update', '--init', '--recursive'], cwd=path) == 0:
                        break
    for s in git_submodules:
        if not os.path.exists(s):
            print_red(f'Fatal error: submodule in {s} not found.')
            print_red('Please clone the repository with --recursive option.')
            sys.exit(1)


def parse_cli_args(args):
    if len(args) == 1 or '--help' in args or '-h' in args:
        print_help()
        return None

    args = args[1:]

    # find the first '--' and treat everything after it as additional arguments
    additional_args = []
    if '--' in args:
        index = args.index('--')
        additional_args = args[index + 1:]
        args = args[:index]

    # find the first argument that starts with '-'
    index = 0
    while index < len(args):
        if args[index].startswith('-'):
            break
        index += 1
    positional_args = args[:index]
    args = args[index:]

    arg_keys = {
        '-b': 'build',
        '-c': 'config',
        '-C': 'clean',
        '-o': 'output',
        '-i': 'install',
        '-t': 'toolchain',
        '-f': "features",
        '-s': 'skip-installed',
        '--build': 'build',
        '--config': 'config',
        '--clean': 'clean',
        '--output': 'output',
        '--install': 'install',
        '--toolchain': 'toolchain',
        '--features': 'features',
        '--skip-installed': 'skip-installed',
        '--ignore-submodules': 'ignore-submodules',
    }

    # parse keyword arguments
    keyword_args = {}
    i = 0
    while i < len(args):
        arg = args[i]
        values = []
        assert arg.startswith('-')
        i += 1
        while i < len(args) and not args[i].startswith('-'):
            values.append(args[i])
            i += 1
        if arg in arg_keys:
            arg = arg_keys[arg]
            if arg not in keyword_args:
                keyword_args[arg] = values
            else:
                keyword_args[arg].extend(values)
        else:
            print_red(f'Ignoring invalid keyword argument: {arg} {" ".join(values)}')

    parsed_args = {}

    # build system
    if 'xmake' in positional_args:
        parsed_args['build_system'] = 'xmake'
        positional_args.remove('xmake')
    elif 'cmake' in positional_args:
        parsed_args['build_system'] = 'cmake'
        positional_args.remove('cmake')

    # build mode
    if 'debug' in positional_args:
        parsed_args['mode'] = 'debug'
        positional_args.remove('debug')
    elif 'release' in positional_args:
        parsed_args['mode'] = 'release'
        positional_args.remove('release')
    elif 'reldbg' in positional_args:
        parsed_args['mode'] = 'reldbg'
        positional_args.remove('reldbg')

    if 'ignore-submodules' in positional_args:
        parsed_args['ignore-submodules'] = True
        positional_args.remove('ignore-submodules')
    else:
        parsed_args['ignore-submodules'] = False

    if positional_args:
        print_red(f'Invalid positional arguments: {positional_args}')
        print_help()
        return None

    # features
    if 'features' in keyword_args:
        features = keyword_args['features']
        if not keyword_args['features']:
            print_red('"--feature | -f" is specified on the command line but no features specified.')
            print_help()
            return None
        valid_features = set()
        for f in features:
            f = f.lower()
            if f == 'all':
                valid_features.update(ALL_FEATURES)
            elif f in ALL_FEATURES:
                valid_features.add(f)
            elif f.startswith('no-'):
                valid_features.discard(f[3:])
            else:
                print_red(f'Ignoring invalid feature "{f}"')
        parsed_args['features'] = list(valid_features)

    # deps
    if 'install' in keyword_args:
        if not keyword_args['install']:
            print_red('"--install | -i" is specified on the command line but no dependencies specified.')
            print_help()
            return None
        valid_deps = set()
        for dep in keyword_args['install']:
            dep = dep.lower()
            if dep == 'all':
                valid_deps.update(ALL_DEPENDENCIES)
            elif dep == 'all-cmake':
                valid_deps.update(ALL_CMAKE_DEPENDENCIES)
            elif dep == 'all-xmake':
                valid_deps.update(ALL_XMAKE_DEPENDENCIES)
            elif dep in ALL_DEPENDENCIES:
                valid_deps.add(dep)
            else:
                print_red(f'Ignoring invalid dependency "{dep}"')
        parsed_args['install'] = list(valid_deps)

    # output
    if 'output' in keyword_args:
        output = keyword_args['output']
        if not output:
            print_red('"--output | -o" is specified on the command line but no output specified.')
            print_help()
            return None
        if len(output) > 1:
            print_red(f'"--output | -o" is specified on the command line but has more than one arguments: {output}.')
            print_help()
            return None
        parsed_args['output'] = output[0]

    # toolchain
    if 'toolchain' in keyword_args:
        toolchain = keyword_args['toolchain']
        if not toolchain:
            print_red('"--toolchain | -t" is specified on the command line but no toolchain specified.')
            print_help()
            return None
        if len(toolchain) > 1:
            print_red(
                f'"--toolchain | -t" is specified on the command line but has more than one arguments: {toolchain}.')
            print_help()
            return None
        toolchain = toolchain[0].lower()
        if not toolchain.startswith('msvc') and not toolchain.startswith('gcc') and not toolchain.startswith('llvm'):
            print_red(f'Unknown toolchain: {toolchain}')
            print_help()
            return None
        parsed_args['toolchain'] = toolchain

    # build jobs
    if 'build' in keyword_args:
        build_jobs = keyword_args['build']
        if not build_jobs:
            jobs = multiprocessing.cpu_count()
        elif len(build_jobs) == 1 and build_jobs[0].isdigit():
            jobs = int(build_jobs[0]) or multiprocessing.cpu_count()
        else:
            print_red(
                f'"--build | -b" requires none or one integral argument. The specified value(s) {keyword_args["build"]} will be ignored.')
            jobs = multiprocessing.cpu_count()
        parsed_args['build'] = max(jobs, 1)

    # bool options
    if 'clean' in keyword_args:
        if keyword_args['clean']:
            print_red(
                f'"--clean | -C" requires no arguments. The specified value(s) {keyword_args["clean"]} will be ignored.')
        parsed_args['clean'] = True

    if 'config' in keyword_args:
        if keyword_args['config']:
            print_red(
                f'"--config | -C" requires no arguments. The specified value(s) {keyword_args["config"]} will be ignored.')
        parsed_args['config'] = True

    if 'ignore-submodules' in keyword_args:
        if keyword_args['ignore-submodules']:
            print_red(
                f'"--ignore-submodules" requires no arguments. The specified value(s) {keyword_args["ignore-submodules"]} will be ignored.')
        parsed_args['ignore-submodules'] = True

    if 'skip-installed' in keyword_args:
        if keyword_args['skip-installed']:
            print_red(
                f'"--skip-installed" requires no arguments. The specified value(s) {keyword_args["skip-installed"]} will be ignored.')
        parsed_args['skip_installed'] = True

    # additional args
    parsed_args['additional_args'] = additional_args

    return parsed_args


def config_project(config, build_config):
    output = config['output'].replace('\\', '/')
    build_system = config['build_system']
    mode = build_config['mode']
    toolchain = build_config['toolchain']
    toolchain_version = build_config['toolchain_version']

    print(f'Build System: {build_system}')
    print(f'Configuration')
    print(f'  Mode: {mode}')
    print(f'  Toolchain: {toolchain}-{toolchain_version if toolchain_version else "default"}')
    print(f'  Output: {output}')

    args = build_system_config_args(config, mode, toolchain, toolchain_version)
    if args is None:
        print_red('Failed to generate build system arguments.')
        return 1

    if "additional_args" in config:
        args += config["additional_args"]

    if config['build_system'] == 'cmake':
        cmake_exe = config['cmake_exe']
        ninja_exe = config['ninja_exe']
        if not check_cmake(cmake_exe):
            print_red('CMake not found. Please install CMake first.')
            print_red('CMake can be installed by running `python3 bootstrap.py -i cmake`.')
            return 1
        if not check_ninja(ninja_exe):
            print_red('Ninja not found. Please install Ninja first.')
            print_red('Ninja can be installed by running `python3 bootstrap.py -i ninja`.')
            return 1
        ninja_exe = ninja_exe.replace('\\', '/')
        args = [cmake_exe, '-S', '.', '-B', output, '-G', 'Ninja',
                f'-DCMAKE_MAKE_PROGRAM={ninja_exe}'] + args
        print(f'Configuring the project: {" ".join(args)}')
        return call(args)
    elif config['build_system'] == 'xmake':
        xmake_exe = config['xmake_exe']
        if not check_xmake(xmake_exe):
            print_red('xmake not found. Please install xmake first.')
            print_red('xmake can be installed by running `python3 bootstrap.py -i xmake`.')
            return 1
        args = [xmake_exe, 'f'] + args + ['-o', output]
        print(f'Configuring the project: {" ".join(args)}')
        return call(args)
    else:
        print_red(f'Unknown build system: {config["build_system"]}')
        return 1


def build_project(config, build_config):
    output = config['output'].replace('\\', '/')
    build_system = config['build_system']
    build_jobs = build_config['jobs']
    if build_system == 'cmake':
        cmake_exe = config['cmake_exe']
        args = [cmake_exe, '--build', output, '-j', str(build_jobs)]
        print(f'Building the project: {" ".join(args)}')
        return call(args)
    elif build_system == 'xmake':
        xmake_exe = config['xmake_exe']
        print(f'Building the project: xmake')
        args = [xmake_exe, '-w', '-j', str(build_jobs)]
        return call(args)
    else:
        print_red(f'Unknown build system: {config["build_system"]}')
        return 1


def main(args: List[str]):
    parsed_args = parse_cli_args(args)
    if parsed_args is None:
        return

    print(parsed_args)
    if not parsed_args.get("ignore-submodules"):
        init_submodule()

    # install deps: this is done before config because feature detection may require deps
    if 'install' in parsed_args:
        install_deps(parsed_args['install'], parsed_args.get('skip-installed', False))

    run_build = 'build' in parsed_args
    run_config = run_build or ('config' in parsed_args and parsed_args['config'])

    config = get_config(parsed_args)

    # write config.json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    if "build_system" not in config:
        print_red('Build system not specified. No config, build, or clean will be performed.')
        return

    # dump build system options, e.g., options.cmake and options.lua
    dump_build_system_options(config)

    # get build config
    output = config['output'].replace('\\', '/')
    build_config = get_build_config(output, parsed_args)

    if "clean" in parsed_args:
        if os.path.exists(output):
            print(f'Cleaning {output}...')
            shutil.rmtree(output)
        if config['build_system'] == 'xmake':
            if os.path.exists('.xmake'):
                print(f'Cleaning .xmake...')
                shutil.rmtree('.xmake')
            if os.path.exists('bin'):
                print(f'Cleaning bin...')
                shutil.rmtree('bin')

    if run_config or run_build:

        if not os.path.exists(output):
            os.mkdir(output)

        # get toolchain environment
        environ_backup = dict(os.environ)
        prepare_toolchain_environment(build_config, config["build_system"])
        build_config["environment"] = {
            k: v for k, v in os.environ.items()
            if k not in environ_backup or environ_backup[k] != v
        }

        # dump build config
        with open(os.path.join(output, 'build_config.json'), 'w') as f:
            json.dump(build_config, f, indent=4)

        if config_project(config, build_config) != 0:
            print_red('Failed to configure the project.')
            return 1
        if run_build:
            if build_project(config, build_config) != 0:
                print_red('Failed to build the project.')
                return 1


if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_path)
    ec = main(sys.argv)
    if print_missing_rust_warning:
        missing_rust_warning()
    sys.exit(ec)
