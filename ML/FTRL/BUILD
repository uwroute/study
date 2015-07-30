cc_binary(
    name = 'ftrl_train',
    srcs = [
        'ftrl_train.cpp',
        'ftrl.cpp'
    ],
    incs = ['../../../ThirdParty/gflags/include/',
           '../'],
    deps = ['//study/ML/Common:common',
            '//ThirdParty/gflags/lib:gflags',
            '//study/ML/data:data'
           ],
    optimize = ['O3'],
    #cxxflags='-std=gnu++0x',
)

cc_binary(
    name = 'ftrl_predict',
    srcs = [
        'ftrl_predict.cpp',
        'ftrl.cpp'
    ],
    incs = ['../../../ThirdParty/gflags/include/',
           '../'],
    deps = ['//study/ML/Common:common',
            '//ThirdParty/gflags/lib:gflags',
            '//study/ML/data:data'
           ],
    optimize = ['O3']
)
