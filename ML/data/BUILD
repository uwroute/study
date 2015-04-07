cc_library(
    name = 'data',
    srcs = [
        'data.cpp'
    ],
    incs = ['../'],
)
cc_binary(
    name = 'load_data',
    srcs = [
        'data_test.cpp',
    ],
    incs = ['../../ThirdParty/gflags/include/',
           '../'],
    deps = ['//ML/Common:common',
            '//ThirdParty/gflags/lib:gflags',
            ':data'
           ],
)
