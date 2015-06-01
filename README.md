# UnifiedParticleFrameworkCUDA
A unified particle framework similar to NVIDIA FleX. It uses CUDA to accelerate simulation of fluids, rigid bodies, deformable bodies and granular flows on the GPU.

References:

[1] P. Goswami, P. Schlegel, B. Solenthaler, et al. Interactive SPH simulation and rendering on the GPU[C] Proceedings of the 2010 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA ’10).

[2] X. Nie, L. Chen, T. Xiang. Real-Time Incompressible Fluid Simulation on the GPU[J]. International Journal of Computer Games Technology, 2015.

[3] N. Akinci, M. Ihmsen, G. Akinci, et al. Versatile rigid-fluid coupling for incompressible SPH[J]. ACM Transactions on Graphics (Proceedings SIGGRAPH) 30, 4 (2012).

[4] N. Akinci, G. Akinci, M. Teschner. Versatile surface tension and adhesion for SPH fluids[J]. ACM Transactions on
Graphics (Proc. SIGGRAPH Asia) 32, 6 (2013)

Portfolio:

YouTube Link: https://www.youtube.com/user/niexiao2008/videos?view_as=public

Youku Link: http://i.youku.com/u/UMzg0NDExODQ=/videos
    
Development Environment：

Windows 7 & Visual C++ 2010 & CUDA Toolkit v7.0 (The default CUDA toolkit installation location is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0) & Intel 3770(CPU) & GTX 780(GPU)

Coding style:

Coding style for this project generally follows the Google C++ Style Guide 

Note:

I use GTX 780 for testing. Since It has compute capability 3.5, I set code generation as compute_35 & sm_35. Also, the header file "sm_35_atomic_functions.h" has been included in particlues_kernel.cuh. You might need to slightly change these settings if you use different GPU with earlier compute capability. But any devices from Fermi to maxwell would work with the code.

