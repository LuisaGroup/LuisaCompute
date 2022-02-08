ispc ispc_test.ispc -o ispc_test.o --cpu=apple-a14 -woff && clang++ ispc_test.cpp ispc_test.o -o ispc_test && ./ispc_test
