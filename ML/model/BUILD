cc_binary(
    name = 'lr_train',
    srcs = [
        'lr_train.cpp',
        'logistic_model.cpp'
    ],
    incs = ['../../ThirdParty/gflags/include/',
            '../'],
    deps = ["//ML/opt:opt",
            '//ThirdParty/gflags/lib:gflags',
            "//ML/data:data",
            "//ML/Common:common"],
    optimize = ['O3']
)
cc_binary(
    name = 'lr_predict',
    srcs = [
        'lr_train.cpp',
        'logistic_model.cpp'
    ],
    incs = ['../../ThirdParty/gflags/include/',
            '../'],
    deps = ["//ML/opt:opt",
            '//ThirdParty/gflags/lib:gflags',
            "//ML/data:data",
            "//ML/Common:common"],
    optimize = ['O3']
)
