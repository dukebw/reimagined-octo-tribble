project('rot-ml-library', 'c')

gsl_dep = dependency('gsl')
hip_hcc_dep = dependency('hip_hcc')
openblas_dep = dependency('openblas')
rocblas_dep = dependency('rocblas')
torch_dep = dependency('ATen')

c_extra_args = ['-x', 'c++', '-hc', '-D__HIPCC__']
link_extra_args = ['-lstdc++']
lib = static_library('rot_ml',
                     'math/rot_math.c',
                     'memory/rot_arena.c',
                     'nn/rot_nn.c',
                     c_args : c_extra_args,
                     dependencies : [openblas_dep],
                     link_args : link_extra_args)

test_math = executable('test_math',
                       'tests/test_math.c',
                       'tests/min_unit.c',
                       'error/stopif.c',
                       'error/log_error.c',
                       c_args : c_extra_args,
                       link_args : link_extra_args,
                       link_with : lib,
                       dependencies : [openblas_dep,
                                       gsl_dep,
                                       hip_hcc_dep,
                                       rocblas_dep,
                                       torch_dep])

test('Test rot_math.', test_math)
