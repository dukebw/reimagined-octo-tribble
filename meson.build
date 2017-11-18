project('rot-ml-library', 'c')

gsl_dep = dependency('gsl')
openblas_dep = dependency('openblas')
torch_dep = dependency('TH')

lib = static_library('rot_ml', 'math/rot_math.c', 'memory/rot_arena.c')

test_math = executable('test_math',
                       'tests/test_math.c',
                       'tests/min_unit.c',
                       'error/stopif.c',
                       link_with : lib,
                       dependencies : [gsl_dep, torch_dep])

test('Test rot_math.', test_math)
