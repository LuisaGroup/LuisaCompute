import multiprocessing
import os
import sys
import shutil
from subprocess import Popen, call, DEVNULL
from typing import List

ALL_FEATURES = ['dsl', 'python', 'gui', 'cuda', 'cpu', 'remote', 'dx', 'metal', 'vulkan']
ALL_DEPENDENCIES = ['rust', 'ninja', 'xmake', 'cmake']

DEPS_DIR = '.deps'
DOWNLOAD_DIR = f'{DEPS_DIR}/downloads'


def check_rust():
    try:
        ret = call(['rustc', '--version'], stdout=DEVNULL, stderr=DEVNULL)
        return ret == 0
    except FileNotFoundError:
        return False


def get_default_features():
    # CPU and Remote are always enabled
    features = ['dsl', 'python', 'gui']
    if not check_rust():
        print("Warning: Rust is required for future releases.", file=sys.stderr)
        print('We strongly recommend you to install Rust **now** to prevent future breakage.', file=sys.stderr)
        print("Please install Rust manually or by running `python bootstrap.py -i rust`.", file=sys.stderr)
        print('Features requires Rust:', file=sys.stderr)
        print('  - CPU backend', file=sys.stderr)
        print('  - Remote backend', file=sys.stderr)
        print('  - IR module', file=sys.stderr)
        print('  - Automatic differentiation', file=sys.stderr)
        features.append('cpu')
        features.append('remote')

    # enable DirectX on Windows by default
    if sys.platform == 'win32':
        features.append('dx')
    # enable Metal on macOS by default
    if sys.platform == 'darwin':
        features.append('metal')
    # enable CUDA if available
    try:
        if 'CUDA_PATH' in os.environ or call(['nvcc', '--version'], stdout=DEVNULL, stderr=DEVNULL) == 0:
            features.append('cuda')
    except FileNotFoundError:
        pass
    try:
        if 'VULKAN_SDK' in os.environ or 'VK_SDK_PATH' in os.environ or call(['vulkaninfo'], stdout=DEVNULL,
                                                                             stderr=DEVNULL) == 0:
            features.append('vulkan')
    except FileNotFoundError:
        pass
    return features


def get_default_toolchain():
    if sys.platform == 'win32':
        return 'msvc'
    elif sys.platform == 'darwin':
        return 'clang'
    elif sys.platform == 'linux':
        return 'gcc'
    else:
        raise ValueError(f'Unknown platform: {sys.platform}')


def get_default_mode():
    return 'release'


def get_default_config():
    return {
        'cmake_args': [],
        'xmake_args': [],
        'build_system': 'cmake',
        'features': get_default_features(),
        'output': 'build',
    }


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


def install_ninja():
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


def install_xmake():
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


def install_cmake():
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
            f.write(f"{DEPS_DIR}/{cmake_file}/bin/cmake")


def install_rust():
    if sys.platform == 'win32':
        # download https://static.rust-lang.org/rustup/dist/i686-pc-windows-gnu/rustup-init.exe
        rustup_init_exe = download_file('https://static.rust-lang.org/rustup/dist/i686-pc-windows-gnu/rustup-init.exe',
                                        'rustup-init.exe')
        call([rustup_init_exe, '-y'])
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        os.system('curl https://sh.rustup.rs -sSf | sh -s -- -y')
    else:
        raise ValueError(f'Unknown platform: {sys.platform}')


def install_dep(build_sys: str, dep: str):
    if dep == 'rust':
        install_rust()
    elif build_sys == 'cmake':
        if dep == 'ninja':
            install_ninja()
        elif dep == 'cmake':
            install_cmake()
    elif build_sys == 'xmake' and dep == 'xmake':
        install_xmake()
    else:
        print(f'The specified dependency "{dep}" is ignored.', file=sys.stderr)


def get_config():
    config = get_default_config()
    # check if config.json exists
    if os.path.exists('config.json'):
        import json
        with open('config.json', 'r') as f:
            config.update(json.load(f))
    return config


def print_help():
    print('Usage: python bootstrap.py [build system] [mode] [options]')
    print('Build system:')
    print('  cmake                  Use CMake')
    print('  xmake                  Use xmake')
    print('Mode (effective only when "--config | -c" or "--build | -b" is specified):')
    print('  release                Release mode (default)')
    print('  debug                  Debug mode')
    print('  reldbg                 Release with debug infomation mode')
    print('Options:')
    print('  --config    | -c       Configure build system')
    print(
        '  --toolchain | -t [toolchain]      Configure toolchain (effective only when "--config | -c" or "--build | -b" is specified)')
    print('      Toolchains:')
    print('          msvc[-version]     Use MSVC toolchain (default on Windows; available on Windows only)')
    print('          clang[-version]    Use Clang toolchain (default on macOS; available on Windows, macOS, and Linux)')
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
    print('  --build   | -b [N]     Build (N = number of jobs)')
    print('  --clean   | -C         Clean build directory')
    print('  --install | -i [deps]  Install dependencies')
    print('      Dependencies:')
    print('          all                Install all dependencies as listed below')
    print('          rust               Install Rust toolchain')
    print('          cmake              Install CMake')
    print('          xmake              Install xmake')
    print('          ninja              Install Ninja')
    print('  --output  | -o [folder]    Path to output directory (default: build)')
    print('  -- [args]              Pass arguments to build system')


def dump_cmake_options(config: dict):
    with open("options.cmake.template") as f:
        options = f.read()
    for feature in config['features']:
        options = options.replace(f'[[feature_{feature}]]', 'ON')
    for feature in ALL_FEATURES:
        if feature not in config['features']:
            options = options.replace(f'[[feature_{feature}]]', 'OFF')
    with open("options.cmake", 'w') as f:
        f.write(options)


def dump_xmake_options(config: dict):
    # TODO: @Maxwell help pls
    pass


def dump_build_system_options(config: dict):
    build_sys = config['build_system']
    if build_sys == 'cmake':
        dump_cmake_options(config)
    elif build_sys == 'xmake':
        dump_xmake_options(config)
    else:
        raise ValueError(f'Unknown build system: {build_sys}')


def build_system_args_cmake(config: dict, mode: str, toolchain: str) -> List[str]:
    args = config['cmake_args']
    cmake_mode = {
        'debug': 'Debug',
        'release': 'Release',
        'reldbg': 'RelWithDebInfo'
    }
    args.append(f'-DCMAKE_BUILD_TYPE={cmake_mode[mode]}')
    toolchain_version = 0
    if '-' in toolchain:
        toolchain, toolchain_version = toolchain.split('-')
        toolchain_version = int(toolchain_version)
    if toolchain == 'msvc':
        pass
    elif toolchain == 'clang':
        args.append("-DCMAKE_C_COMPILER=clang")
        args.append("-DCMAKE_CXX_COMPILER=clang++")
        pass
    elif toolchain == 'gcc':
        pass
    return args


def build_system_args_xmake(config: dict, mode: str, toolchain: str) -> List[str]:
    args = config['xmake_args']
    xmake_mode = {
        'debug': 'debug',
        'release': 'release',
        'reldbg': 'releasedbg'
    }
    args.append(f'-m {xmake_mode[mode]}')
    # TODO: @Maxwell help pls
    return args


def build_system_args(config, mode, toolchain) -> List[str]:
    if config['build_system'] == 'cmake':
        return build_system_args_cmake(config, mode, toolchain)
    elif config['build_system'] == 'xmake':
        return build_system_args_xmake(config, mode, toolchain)
    else:
        raise ValueError(f'Unknown build system: {config["build_system"]}')


submods = [
    'corrosion',
    'EASTL',
    ## TODO: add more submodules here
]


def init_submodule():
    if os.path.exists('.git'):
        os.system('git submodule update --init --recursive')
    else:
        for s in submods:
            if not os.path.exists(f'src/ext/{s}'):
                print(f'Fatal error: submodule in src/ext/{s} not found.', file=sys.stderr)
                print('Please clone the repository with --recursive option.', file=sys.stderr)
                sys.exit(1)


def main(args: List[str]):
    init_submodule()
    if len(args) == 1:
        print_help()
        return
    config = get_config()
    for i, arg in enumerate(args):
        if arg.lower() in ('cmake', 'xmake'):
            config['build_system'] = arg.lower()
            args.pop(i)
            break
    mode = get_default_mode()
    toolchain = get_default_toolchain()
    for i, arg in enumerate(args):
        if arg.lower() in ('debug', 'release', 'reldbg'):
            mode = arg.lower()
            args.pop(i)
            break
    i = 1
    run_config = False
    run_build = False
    build_jobs = multiprocessing.cpu_count()
    while i < len(args):
        opt = args[i]
        if opt == '--clean' or opt == '-C':
            if os.path.exists(config['output']):
                import shutil
                shutil.rmtree(config['output'])
            if os.path.exists('options.cmake.cli'):
                os.remove('options.cmake.cli')
            if os.path.exists('options.xmake.cli'):
                os.remove('options.xmake.cli')
            return
        elif opt == '--help' or opt == '-h':
            print_help()
            return
        elif opt == '--config' or opt == '-c':
            run_config = True
            i += 1
        elif opt == '--build' or opt == '-b':
            run_build = run_config = True
            i += 1
            if i < len(args) and not args[i].startswith('-'):
                build_jobs = int(args[i])
                i += 1
        elif opt == '--features' or opt == '-f':
            i += 1
            while i < len(args) and not args[i].startswith('-'):
                f = args[i].lower()
                if f == 'all':
                    config['features'] = get_default_features()
                elif f.startswith('no-'):
                    f = f[3:]
                    if f in config['features']:
                        config['features'].remove(f)
                else:
                    if f not in config['features']:
                        config['features'].append(f)
                i += 1
        elif opt == '--install' or opt == '-i':
            i += 1
            deps = []
            while i < len(args) and not args[i].startswith('-'):
                deps.append(args[i].lower())
                i += 1
            if "all" in deps:
                deps = ALL_DEPENDENCIES
            for d in deps:
                install_dep(config['build_system'], d)
        elif opt == '--output' or opt == '-o':
            config['output'] = args[i + 1]
            i += 2
        elif opt == '--toolchain' or opt == '-t':
            toolchain = args[i + 1].lower()
            i += 2
        elif opt == "--":
            if config['build_system'] == 'cmake':
                config['cmake_args'] = args[i + 1:]
            elif config['build_system'] == 'xmake':
                config['xmake_args'] = args[i + 1:]
            else:
                raise ValueError(f'Unknown build system: {config["build_system"]}')
            break
        else:
            raise ValueError(f'Unknown option: {opt}')

    # print bootstrap information
    print(f'Build System: {config["build_system"]}')
    print(f'Run Configuration: {run_config}')
    if run_config:
        print(f'  Mode: {mode}')
        print(f'  Toolchain: {toolchain}')
        print(f'  Output: {config["output"]}')
    print(f'Run Build: {run_build}')
    if run_build:
        print(f'  Build Jobs: {build_jobs}')

    # write config.json
    import json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # dump build system options, e.g., options.cmake and options.lua
    dump_build_system_options(config)

    # config and build
    output = config['output']
    if run_config or run_build:
        if not os.path.exists(output):
            os.mkdir(output)

    # config build system
    if run_config:
        args = build_system_args(config, mode, toolchain)
        if config['build_system'] == 'cmake':
            args = ['cmake', '..'] + args
            print(f'Configuring the project: {" ".join(args)}')
            p = Popen(args, cwd=output)
            p.wait()
        elif config['build_system'] == 'xmake':
            args = ['xmake', 'f'] + args
            print(f'Configuring the project: {" ".join(args)}')
            p = Popen(args)
            p.wait()
        else:
            raise ValueError(f'Unknown build system: {config["build_system"]}')
    if run_build:
        if config['build_system'] == 'cmake':
            args = ['cmake', '--build', '.', '-j', str(build_jobs)]
            print(f'Building the project: {" ".join(args)}')
            p = Popen(args, cwd=output)
            p.wait()
        elif config['build_system'] == 'xmake':
            print(f'Building the project: xmake')
            os.system('xmake')
        else:
            raise ValueError(f'Unknown build system: {config["build_system"]}')


if __name__ == '__main__':
    main(sys.argv)
