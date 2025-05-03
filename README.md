# coconutFluid-pressureGPU2D

the name is too long...

# Description

This project is use SIMPLE-like algorithm to solving the N-S equation in 2D plane. We implement the 2D vortex street problem and user can interact with the fluid, so called "smoke".

Employing the Semi Lagrangian scheme to push the smoke and velocity move under the presure.

Different with another project. In pressureGPU2D, we employ CUDA to accelerate the calculate speed.

# How to compile

1. Only can run on windows. Compiling on windows is strongly recommended. Further more, your computer must have a NVIDIA graphics card.

2. Please download VisualStudio2022 and Easyx Lib as environment configure.

Easyx Lib link: https://easyx.cn/

3. Download the cuda sdk on NVIDIA website, the recommended version is 12.6
   
4. Open sln file in VisualStudio2022, and then compile. May need to change the architecture sm_86 to other
