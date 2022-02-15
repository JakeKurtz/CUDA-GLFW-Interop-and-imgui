/***************************************************************************************
    *    Title: A tiny example of CUDA + OpenGL interop with write-only surfaces and CUDA kernels. Uses GLFW+GLAD. 
    *    Author: Allan MacKinnon 
    *    Availability: https://gist.github.com/allanmac/4ff11985c3562830989f
    *    
    ***************************************************************************************/

#pragma once
#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

//#include "CudaHelpers.cuh"

class Interop
{
private:
    bool multi_gpu;     // split GPUs?

    // number of fbo's
    int count;
    int index;

    int width;
    int height;

    // GL buffers
    GLuint* fbo_list;
    GLuint* rbo_list;

    // CUDA resources
    cudaGraphicsResource_t* cgr_list;
    cudaArray_t* ca_list;

public:
    Interop(const bool multi_gpu, const int fbo_count);
    ~Interop();

    cudaError_t set_size(const int width, const int height);
    void get_size(int* const width, int* const height);
    cudaError_t map(cudaStream_t stream);
    cudaError_t unmap(cudaStream_t stream);
    cudaError_t array_map();
    cudaArray_const_t array_get();
    void swap();
    void clear();
    void blit();
};

