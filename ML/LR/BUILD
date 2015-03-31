cc_binary(
    name = 'lr',
    srcs = [
        'SparseLR_test.cpp',
        'SparseBatchLR.cpp'
    ],
    incs = ['../../ThirdParty/gflags/include/'],
    deps = ['//ML/Common:common',
            '//ThirdParty/gflags/lib:gflags'
           ],
    #optimize = ['O']
)
