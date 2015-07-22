cc_binary(
    name = 'lr_train',
    srcs = [
        'lr_train.cpp',
        'logistic_model.cpp'
    ],
    incs = ['../../../ThirdParty/gflags/include/',
            '../'],
    deps = ["//study/ML/opt:opt",
            '//ThirdParty/gflags/lib:gflags',
            "//study/ML/data:data",
            "//study/ML/Common:common"],
    optimize = ['O3']
)
cc_binary(
    name = 'lr_predict',
    srcs = [
        'lr_predict.cpp',
        'logistic_model.cpp'
    ],
    incs = ['../../../ThirdParty/gflags/include/',
            '../'],
    deps = ["//study/ML/opt:opt",
            '//ThirdParty/gflags/lib:gflags',
            "//study/ML/data:data",
            "//study/ML/Common:common"],
    optimize = ['O3']
)
