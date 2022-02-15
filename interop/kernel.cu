#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glad/glad.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <cuda_gl_interop.h>

#include "Interop.h"
#include "CudaHelpers.cuh"

#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

const char* glsl_version;
bool show_demo_window = true;
bool show_another_window = false;
ImVec4 clear_color;

surface<void, cudaSurfaceType2D> surf;

union pxl_rgbx_24
{
    uint1       b32;

    struct {
        unsigned  r : 8;
        unsigned  g : 8;
        unsigned  b : 8;
        unsigned  na : 8;
    };
};

__device__ float remap(float value, float min, float max)
{
    return (value - min) / (max - min);
}

__global__ void test_kernel(int w, int h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int x = idx % w;
    const int y = idx / w;
    
    union pxl_rgbx_24 rgbx;

    rgbx.r = (int)255 * remap(x, 0.f, (float)w);
    rgbx.g = (int)255 * remap(y, 0.f, (float)h);
    rgbx.b = 0;
    rgbx.na = 255;

    surf2Dwrite(
        rgbx.b32,
        surf,
        x * sizeof(rgbx),
        y,
        cudaBoundaryModeZero
    );
}

static void glfw_error_callback(int error, const char* description)
{
    fputs(description, stderr);
}

static void glfw_window_size_callback(GLFWwindow* window, int width, int height)
{
    // get context
    Interop* interop = (Interop*)glfwGetWindowUserPointer(window);
    interop->set_size(width, height);
}

static void glfw_init(GLFWwindow** window, const int width, const int height)
{
    //
    // INITIALIZE GLFW/GLAD
    //

    glfwSetErrorCallback(glfw_error_callback);
    //glfwSetKeyCallback(*window, glfw_key_callback);
    glfwSetFramebufferSizeCallback(*window, glfw_window_size_callback);
    //glfwSetCursorPosCallback(*window, glfw_mouse_callback);
    //glfwSetMouseButtonCallback(*window, glfw_mouse_button_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glsl_version = "#version 330";

    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef PXL_FULLSCREEN
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    *window = glfwCreateWindow(mode->width, mode->height, "GLFW / CUDA Interop", monitor, NULL);
#else
    *window = glfwCreateWindow(width, height, "GLFW / CUDA Interop", NULL, NULL);
#endif

    if (*window == NULL)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(*window);

    if (glewInit() != GLEW_OK)
        exit(EXIT_FAILURE);

    // set up GLAD
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // ignore vsync for now
    glfwSwapInterval(0);

    // only copy r/g/b
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);
}

static void imgui_init(GLFWwindow** window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    //ImGui::StyleColorsDark();
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(*window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    show_demo_window = true;
    show_another_window = false;
    ImVec4 clear_color;
}

cudaError_t kernel_launcher(
    cudaArray_const_t array,
    const int         width,
    const int         height,
    cudaEvent_t       event,
    cudaStream_t      stream)
{
    cudaError_t cuda_err;

    // cuda_err = cudaEventRecord(event,stream);

    cuda_err = cudaBindSurfaceToArray(surf, array);

    if (cuda_err)
        return cuda_err;

    const int blocks = (width * height + 256 - 1) / 256;

    // cuda_err = cudaEventRecord(event,stream);

    if (blocks > 0)
        test_kernel <<< blocks, 256, 0, stream >>> (width, height);

    // cuda_err = cudaStreamWaitEvent(stream,event,0);

    return cudaSuccess;
}

int main()
{
    int width = 1024;
    int height = 1024;

    GLFWwindow* window;
    glfw_init(&window, width, height);
    imgui_init(&window);

    //
    // CREATE CUDA STREAM & EVENT
    //
    cudaStream_t stream;
    cudaEvent_t  event;

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));   // optionally ignore default stream behavior
    checkCudaErrors(cudaEventCreateWithFlags(&event, cudaEventBlockingSync)); // | cudaEventDisableTiming);

    //
    // CREATE INTEROP
    //
    // TESTING -- DO NOT SET TO FALSE, ONLY TRUE IS RELIABLE
    Interop* interop = new Interop(true, 2);

    checkCudaErrors(interop->set_size(width, height));

    //
    // SET USER POINTER AND CALLBACKS
    //
    glfwSetWindowUserPointer(window, interop);
    glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

    while (!glfwWindowShouldClose(window))
    {
        //
        // EXECUTE CUDA KERNEL ON RENDER BUFFER
        //
        int width, height;
        cudaArray_t cuda_array;

        interop->get_size(&width, &height);
        checkCudaErrors(interop->map(stream));
        checkCudaErrors(kernel_launcher(interop->array_get(),width,height,event,stream));
        checkCudaErrors(interop->unmap(stream));

        //
        // BLIT & SWAP FBO
        // 
        interop->blit();
        //interop->clear();
        interop->swap();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window); 
        glfwPollEvents();
    }

    //
    // CLEANUP
    //
    delete interop;
    glfwDestroyWindow(window);
    glfwTerminate();

    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
