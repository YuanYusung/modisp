# Sources and attribution

The Dunkin compound-matrix evaluator and SH recurrence are transcriptions and
adaptations of the CPS/disba formulations. Preserve the following attribution:

- disba, Copyright (c) 2020 Keurfon Luu, BSD-3-Clause.
- Computer Programs in Seismology, D. R. Russell and R. B. Herrmann, Saint Louis
  University; original surf96 notices date from 1986 and 1991. CPS is currently
  distributed by its authors under the MIT license.
- Dunkin (1965), BSSA 55(2), 335–358.
- Pan et al. (2026), doi:10.1785/0120250207: density-guided sampling and quadratic
  extrema motivate this independently implemented root search.
- Zhu, D., Xu, T., Hao, J. and Yao, Z. (2026), "An Adaptive Strategy for Robust
  and Efficient Computation of Dispersion Curves in a Layered Half-Space,"
  BSSA 116, 1383–1394, doi:10.1785/0120260071: equations 13–14 motivate the
  independently implemented local integral-consistency search. Its signed,
  smoothly rescaled Dunkin/SH feature and acceptance safeguards are specific
  to modisp. The paper validates GRT, not this implementation.

PyGRT (GPL-3.0) is used only as a pinned external validation executable
(`pygrt-kit==0.18.0`). No PyGRT source, binary or runtime linkage is included in
modisp. Its implementations were not copied into the BSD source tree.

QEDispInv source and binaries are not runtime dependencies and are not included
in the published source/wheel packages. Optional external reference tests retain
their upstream licenses separately. pybind11 and scikit-build-core are build
dependencies and retain their own BSD-3-Clause licenses.

## disba license

Copyright (c) 2020, Keurfon Luu. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## CPS MIT license

Source: https://rbherrmann.github.io/ComputerProgramsSeismology/mitlicense.html

Copyright (c) 2024 Robert B. Herrmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## pybind11 license (compiled binding headers)

Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Please also refer to the file .github/CONTRIBUTING.md, which clarifies licensing of
external contributions to this project including patches, pull requests, etc.

## Optional CUDA runtime

The CUDA plugin statically links NVIDIA CUDA Runtime 12.6. Its NVIDIA license
and third-party notices are included in the plugin distribution as
`licenses/NVIDIA_CUDA_12.6_EULA.txt`. These terms apply to that runtime; the CPU
base package does not include or link it. The project source remains under the
BSD-3-Clause license with the CPS/disba/pybind11 notices above.
