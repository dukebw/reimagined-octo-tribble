project('rot-ml-library', 'c')

cublas_dep = dependency('cublas', required : false)
cudart_dep = dependency('cudart', required : false)
gsl_dep = dependency('gsl')
hip_hcc_dep = dependency('hip_hcc', required : false)
openblas_dep = dependency('openblas')
rocblas_dep = dependency('rocblas', required : false)
torch_dep = dependency('ATen')

c_extra_args = ['-x', 'c++']
link_extra_args = ['-lstdc++']
if hip_hcc_dep.found()
        c_extra_args += ['-hc', '-D__HIPCC__']
endif

lib_src = ['math/rot_math.c',
           'memory/rot_arena.c']

if hip_hcc_dep.found()
        lib_src += ['platform/miopen.c']
        add_global_arguments('-DPLATFORM_MIOPEN=1', language : 'c')
endif

if cublas_dep.found()
        lib_src += ['platform/cudnn.c']
        add_global_arguments('-DPLATFORM_CUDNN=1', language : 'c')
endif

lib = static_library('rot_ml',
                     sources : lib_src,
                     c_args : c_extra_args,
                     dependencies : [openblas_dep],
                     link_args : link_extra_args)

test_math_src = ['tests/test_math.c',
                 'tests/min_unit.c',
                 'error/stopif.c',
                 'error/log_error.c']

if hip_hcc_dep.found()
        test_math_src += ['tests/test_miopen.c']
endif

if cublas_dep.found()
        test_math_src += ['tests/test_cudnn.c']
endif

test_math = executable('test_math',
                       sources : test_math_src,
                       c_args : c_extra_args,
                       link_args : link_extra_args,
                       link_with : lib,
                       dependencies : [cublas_dep,
                                       cudart_dep,
                                       openblas_dep,
                                       gsl_dep,
                                       hip_hcc_dep,
                                       rocblas_dep,
                                       torch_dep])

test('Test rot_math.', test_math)
