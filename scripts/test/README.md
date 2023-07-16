# TEST script for LC

The python test script are built for batch-testing jobs. If you want to test a single feature when developing, run the executable natively with your own filters.

e.g. `xmake run test_main dx -tc=context`

The scripting directory structure:

- `test_<platform>.py`: the test script entry point for each platform
- `config/`: the folder to place self-defined test cases, if you would like to run 
  - `config/builtin`: the builtin test cases
  - `config/custom`: the self-customized test cases
- `config.json`: the self-generated default config, will be used for no appendix parameters

