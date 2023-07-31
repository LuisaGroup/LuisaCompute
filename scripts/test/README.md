# TEST script for LC

The python test script are built for batch-testing jobs. If you want to test a single feature when developing, run the executable natively with your own filters.

e.g. `xmake run test_feat dx -tc=context`

when release stage, the build system will build all test suites into one executable called `test_all`, and the scripts here are used to control whether to test with `test_all` according to different environment.

The scripting directory structure:

- `test_<platform>.py`: the test script entry point for each platform
- `config/`: the folder to place self-defined test cases, if you would like to run 
  - `config/builtin`: the builtin test cases
  - `config/custom`: the self-customized test cases
- `config.json`: the self-generated default config, will be used for no appendix parameters, ignored by git

### Use builtin Testing script

we have made some built-in testing script placed in `config/builtin` directory, you can easily call them by its name: `python scripts/test/test_win.py --config=cuda_common`

### Use Customized Testing Script

A sample custom config

```json
{
    "build_system": "xmake",
    "device_list": [
        "dx",
        "cuda"
    ],
    "feat_list": [
        "feat"
    ]
}
```

place it into `config/custom` directory with some name like `dummy.json`, and run it with 

`python scripts/test/test_win.py --config=zzh --custom=true`

Then the script will run for `feat` test suites on `dx` and `cuda` devices.


