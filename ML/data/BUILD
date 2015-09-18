cc_library(
    name = 'data',
    srcs = [
    'data.cpp',
    ],
    incs = ['../'],
    deps = ['//study/ML/Common:common'],
    optimize = ['O3']
)
cc_binary(
    name = 'load_data',
    srcs = [
        'data_test.cpp',
        'data.cpp'
    ],
    incs = ['../../../ThirdParty/gflags/include/',
           '../'],
    deps = ['//study/ML/Common:common',
            '//ThirdParty/gflags/lib:gflags'
           ],
    optimize = ['O3']
)
cc_binary(
    name = 'load_data_tpl',
    srcs = [
        'data_tpl_test.cpp',
    ],
    incs = ['../../../ThirdParty/gflags/include/',
           '../'],
    deps = ['//study/ML/Common:common',
            '//ThirdParty/gflags/lib:gflags'
           ],
    optimize = ['O3']
)
