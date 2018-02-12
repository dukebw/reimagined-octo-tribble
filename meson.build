project('rot-ml-library', 'c')

cublas_dep = dependency('cublas')
cudart_dep = dependency('cudart')
gsl_dep = dependency('gsl')
openblas_dep = dependency('openblas')
torch_dep = dependency('ATen')

lib = static_library('rot_ml', 'math/rot_math.c', 'memory/rot_arena.c')

test_math = executable('test_math',
                       'tests/test_math.c',
                       'tests/min_unit.c',
                       'error/stopif.c',
                       'error/log_error.c',
                       link_with : lib,
                       dependencies : [cublas_dep,
                                       cudart_dep,
                                       openblas_dep,
                                       gsl_dep,
                                       torch_dep])

test('Test rot_math.', test_math)
