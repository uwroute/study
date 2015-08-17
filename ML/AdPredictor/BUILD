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

cc_binary(
    name = 'mt_adpredictor_train',
    srcs = [
        'adpredictor.cpp',
        'mt_adpredictor_train.cpp',
        'mt_adpredictor.cpp'
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
    name = 'adpredictor_pure',
    srcs = [
        'adpredictor_pure.cpp',
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

cc_binary(
    name = 'padpredictor_train',
    srcs = [
        'parallel_adpredictor_train.cpp',
        'parallel_adpredictor.cpp'
    ],
    incs = ['../../../ThirdParty/gflags/include/',
           '../'],
    deps = ['//study/ML/Common:common',
            '//ThirdParty/gflags/lib:gflags',
            '//study/ML/data:data'
           ],
    optimize = ['O3']
)