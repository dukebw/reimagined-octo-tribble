# reimagined-octo-tribble machine learning library


## Building

[Meson](https://github.com/mesonbuild/meson) and
[Ninja](https://github.com/ninja-build/ninja) are dependencies for the build.

```
CC=clang CXX=clang meson build --buildtype debugoptimized
cd build
ninja && ninja test
```


## Project goals

1. Run as fast as possible in terms of both optimizing computation speed and
   minimizing library baggage.

2. Provide a granular C API that can allow control over the training/inference
   loop at both high and low abstraction levels.

3. Support both AMD and NVIDIA GPUs, as well as CPU.


## Contact

Brendan Duke: brendanw.duke@gmail.com
