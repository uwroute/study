cc_library(
    name = 'data',
    srcs = [
    'data.cpp',
    ],
    incs = ['../'],
    deps = ['//ML/Common:common'],
    optimize = ['O3']
)
cc_binary(
    name = 'load_data',
    srcs = [
        'data_test.cpp',
        'data.cpp'
    ],
    incs = ['../../ThirdParty/gflags/include/',
           '../'],
    deps = ['//ML/Common:common',
            '//ThirdParty/gflags/lib:gflags'
           ],
    optimize = ['O3']
)
