cc_binary(
    name = 'matchbox_train',
    srcs = [
        'matchbox_train.cpp',
        'matchbox.cpp'
    ],
    incs = ['../../../ThirdParty/gflags/include/',
           '../'],
    deps = ['//study/ML/Common:common',
            '//ThirdParty/gflags/lib:gflags',
            '//study/ML/data:data'
           ],
    optimize = ['O3']
)

