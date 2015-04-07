cc_binary(
    name = 'lr_train',
    srcs = [
        'logistic.cpp',
        'logistic_model.cpp'
    ],
    incs = ['../../ThirdParty/gflags/include/',
            '../'],
    deps = ["//ML/opt:opt",
            '//ThirdParty/gflags/lib:gflags',
            "//ML/data:data"]
)
