cc_binary(
    name = 'plr_train',
    srcs = [
        'main.cpp',
        'state.cpp',
	'read.cpp',
	'grad.cpp',
	'data.cpp',
	'owlqn.cpp'
    ],
    incs = ['../../../ThirdParty/gflags/include/',
           '../'],
    deps = ['//study/ML/Common:common',
            '//ThirdParty/gflags/lib:gflags'
           ],
    optimize = ['O3'],
    #cxxflags='-std=gnu++0x',
)
