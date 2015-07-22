cc_binary(
    name = 'adpredictor_train',
    srcs = [
        'adpredictor_train.cpp',
        'adpredictor.cpp'
    ],
    incs = ['../../../ThirdParty/gflags/include/',
           '../'],
    deps = ['//study/ML/Common:common',
            '//ThirdParty/gflags/lib:gflags',
            '//study/ML/data:data'
           ],
    optimize = ['O3']
)

cc_binary(
    name = 'adpredictor_predict',
    srcs = [
        'adpredictor_predict.cpp',
        'adpredictor.cpp'
    ],
    incs = ['../../..//ThirdParty/gflags/include/',
           '../'],
    deps = ['//study/ML/Common:common',
            '//ThirdParty/gflags/lib:gflags',
            '//study/ML/data:data'
           ],
    optimize = ['O3']
)
