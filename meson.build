project('rot-ml-library', 'c')

lib = static_library('rot_ml', 'math/rot_math.c')

test_math = executable('test_math', 'tests/test_math.c', link_with : lib)
test('Test rot_math.', test_math)
