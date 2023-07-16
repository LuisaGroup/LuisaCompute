import subprocess
import os 
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Entry for Test Script")
    parser.add_argument("--builtin", type=bool, default=False, help="use default configuration")
    parser.add_argument("--config", type=str, default="default", help="configuration file")
    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_file_path = ""
    if args.config == "default":
        # if no `config.json`, copy `_config_<platform>.json` to `config.json` as default test configuration
        default_config_file_path = os.path.join(current_dir, "config/builtin/_config_win.json")
        config_file_path = os.path.join(current_dir, "config.json")
        if os.path.isfile(config_file_path) == False:
            print("connot find default configuration from {}".format(config_file_path))
            print("will copy the default configuration from {}".format(default_config_file_path))
            subprocess.run(["powershell", "Copy-Item", default_config_file_path, config_file_path], shell=True)

    else:
        # set config file path
        if args.builtin:
            print("use builtin configuration, config: {}".format(args.config))
            config_file_path = os.path.join(current_dir, "config/builtin/{}_win.json".format(args.config))
        else:
            print("use custom configuration, config: {}".format(args.config))
            config_file_path = os.path.join(current_dir, "config/custom/{}.json".format(args.config))

    # read configuration from `config.json`

    config = {}
    with open(config_file_path, "r") as f:
        config = json.load(f)
        f.close()

    # if get configuration successfully, run test

    if config != set():
        print("get configuration successfully")
        build_system = config["build_system"]
        device_list = config["device_list"]
        feat_list = config["feat_list"]
        if build_system == "cmake":
            raise NotImplementedError
        elif build_system == "xmake":
            print("get build system: xmake")
            print("start to run test")
            for feat in feat_list:
                for device in device_list:
                    print("runnint test_{} on {}".format(feat, device))
                    subprocess.run(["powershell", "xmake", "run", "test_{}".format(feat), device], shell=True)
                    print("test_{} on {} finished".format(feat, device))
            # TODO: switch device for [dx, cuda, vk, metal]
            print("all test finished")
    else:
        print("get configuration failed")
        exit(1)