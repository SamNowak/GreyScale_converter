#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include "stb_image.h"
#include "stb_image_write.h"


//each thread will transform a pixel into a grayscale versions of itself
__global__ void grayscale_kernel(unsigned char* img_In, unsigned char* img_Out, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int index = row * width + col;
        int rgb_offset = index * channels;

        //getting each color and 'averaging' them
        unsigned char r = img_In[rgb_offset];
        unsigned char g = img_In[rgb_offset + 1];
        unsigned char b = img_In[rgb_offset + 2];
        img_Out[index] = 0.299*r + 0.587*g + 0.114*b;
    }
}

//simple edge detection kernel, each thread will compute its own pixel with the surrounding pixels
__global__ void first_edge_kernel(unsigned char* img_In, unsigned char* img_Out, float * edge_kernel,int r, int width, int height, int padded_width) {
    int kernel_size = 2 * r + 1;

    //thread position
    int col = blockIdx.x * blockDim.x + threadIdx.x+r;
    int row = blockIdx.y * blockDim.y + threadIdx.y+r;

    float pixel_value = 0.0f;

    //getting the surrounding pixels
    if (col < width+r && row < height+r) {
        for (int krow = 0; krow < kernel_size; krow++) {
            for (int kcol = 0; kcol < kernel_size; kcol++) {
                int sRow = row-r + krow;
                int sCol = col-r +kcol;
                pixel_value += edge_kernel[krow*kernel_size + kcol]*img_In[sRow*padded_width + sCol];
            }
        }
        //write back
        img_Out[(row-r) * width + (col-r)] = min(255.0f, max(0.0f, pixel_value));
    }
}

//optimized convolution kernel, using shared memory and tiling.
//shared memory and tiling uses the Spatial locality, don't have to bring every pixel from img from global memory
//used cuda constants, since the filter is small we can fit it in memory cache too
#define TILE_DIM 16
#define RADIUS 1
#define FILTER_SIZE (2*RADIUS+1)
__constant__ float d_filter[FILTER_SIZE * FILTER_SIZE];
__global__ void edge_detection_kernel(const unsigned char* __restrict__ img_in, unsigned char* img_out, int width, int height) {

    extern __shared__ unsigned char tile[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int column = bx * TILE_DIM + tx;
    int row = by * TILE_DIM + ty;

    const int tileW = TILE_DIM+2;

    //loading values into the tile
    for(int y = ty; y < tileW; y+= blockDim.y) {
        for(int x = tx; x < tileW; x+= blockDim.x) {
            int input_x = bx * TILE_DIM + x;
            int input_y = by * TILE_DIM + y;
            tile[y*tileW + x] = img_in[input_y * (width+2) + input_x];
        }
    }

    //wait for all threads to input their tile inputs. making sure all values are in shared memory
    __syncthreads();

    //the convolution step
    if (column < width && row < height) {
        float sum = 0.0f;

        for (int filter_y = 0; filter_y < FILTER_SIZE; ++filter_y) {
            for (int filter_x = 0; filter_x < FILTER_SIZE; ++filter_x) {
                unsigned char pixel = tile[(ty+filter_y) * tileW + (tx+filter_x)];
                sum += d_filter[filter_y * FILTER_SIZE + filter_x]*pixel;
            }
        }
        int index = row * width + column;
        img_out[index] = (unsigned char)min(255.0f, max(0.0f, sum));
    }
}


//combining the two edges together to set a full edge picture. aka combining horizontal edges and vertical edges
__global__ void combine_edges(unsigned char* vertical_img, unsigned char* horizontal_img, unsigned char* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float x = vertical_img[index];
        float y = horizontal_img[index];
        output[index] = min(255.0f, sqrtf(x*x + y*y));
    }
}

int main() {

    //user input which type of kernel to run
    std::cout << "Simple:1 or Optimized: 2: ";
    int version; std::cin >> version;


    const char* input_img = "C:/Users/nowak/CLionProjects/EdgeDection/woodenBridge.JPG";
    const char* output_img = "edges_output.jpg";

    int width, height, channels;
    unsigned char* img = stbi_load(input_img, &width, &height, &channels, 0);

    if (!img) {
        std::cerr << "Failed to load image: " << stbi_failure_reason() << "\n";
        return 1;
    }

    //creating a new img which will be used as a grayscale version of the orginal
    size_t size = width * height;
    unsigned char* img_gray_h = new unsigned char[size];

    //allocating memory
    unsigned char *img_d, *img_gray_d;
    cudaMalloc(&img_d, width * height * channels);
    cudaMalloc(&img_gray_d, size);
    cudaMemcpy(img_d, img, width * height * channels, cudaMemcpyHostToDevice);

    //dimension
    dim3 dimGrid(ceil(width/16.0),ceil(height/16.0), 1);
    dim3 dimBlock(16,16,1);

    //grayscale timming
    auto t0 = std::chrono::high_resolution_clock::now();
    //calling kernel to make our image gray scale
    grayscale_kernel<<<dimGrid,dimBlock>>>(img_d, img_gray_d, width, height, channels);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    //copying output
    cudaMemcpy(img_gray_h, img_gray_d, size, cudaMemcpyDeviceToHost);
    auto t1b = std::chrono::high_resolution_clock::now();

    //This part is just padding the image(surrounding it by a duplicate pixel based off the edge pixel
    //padding the image by one pixel, since our convolution kernel is a 3x3 matrix
    int r = 1;
    int padded_width = width + 2;
    int padded_height = height + 2;

    unsigned char* padded_img_h = new unsigned char[padded_width * padded_height]();
    //memset(padded_img_h, 0, padded_width * padded_height);
    // Copy original grayscale image to the center
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            padded_img_h[(y + 1) * padded_width + (x + 1)] = img_gray_h[y * width + x];
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    //allocating new image pad and outputs
    unsigned char *padded_img_d, *v_img_d, *h_img_d, *edge_img_d;
    cudaMalloc(&padded_img_d, padded_width * padded_height);
    cudaMemcpy(padded_img_d, padded_img_h, padded_width * padded_height, cudaMemcpyHostToDevice);
    cudaMalloc(&v_img_d, size);
    cudaMalloc(&h_img_d, size);
    cudaMalloc(&edge_img_d, size);

    //setting our convoluation kernels
    float h_kernel_h[9] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    float v_kernel_h[9] = {
         1,  1,  1,
         0,  0,  0,
        -1, -1, -1
    };

    //getting the time it takes for each kernel
    std::chrono::high_resolution_clock::time_point edge_start, edge_end;
    //regular kernel version
    if (version == 1) {
        edge_start = std::chrono::high_resolution_clock::now();
        //allocate memory for kernels
        float* h_kernel_d;
        float* v_kernel_d;
        cudaMalloc(&h_kernel_d, 9 * sizeof(float));
        cudaMalloc(&v_kernel_d, 9 * sizeof(float));
        cudaMemcpy(h_kernel_d, h_kernel_h, 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(v_kernel_d, v_kernel_h, 9 * sizeof(float), cudaMemcpyHostToDevice);

        //allocate memory for image
        first_edge_kernel<<<dimGrid,dimBlock>>>(padded_img_d, v_img_d, v_kernel_d, r, width, height, padded_width);
        first_edge_kernel<<<dimGrid,dimBlock>>>(padded_img_d, h_img_d, h_kernel_d, r, width, height, padded_width);

        //dimension
        dim3 gridDim(ceil(width/16.0),ceil(height/16.0), 1);
        dim3 blockDim(16,16,1);
        combine_edges<<<ceil(size/256.0), 256>>>(v_img_d, h_img_d,edge_img_d, size);

        cudaFree(v_kernel_d);
        cudaFree(h_kernel_d);

        //end time
        cudaDeviceSynchronize();
        edge_end = std::chrono::high_resolution_clock::now();
    }
    else { //optimized version
        edge_start = std::chrono::high_resolution_clock::now();
        cudaMemcpyToSymbol(d_filter,h_kernel_h,9 * sizeof(float));
        edge_detection_kernel<<<dimGrid,dimBlock,(TILE_DIM+2)*(TILE_DIM+2)>>>(padded_img_d, v_img_d, width, height);
        cudaMemcpyToSymbol(d_filter,v_kernel_h,9 * sizeof(float));
        edge_detection_kernel<<<dimGrid,dimBlock,(TILE_DIM+2)*(TILE_DIM+2)>>>(padded_img_d, h_img_d, width, height);
        combine_edges<<<ceil(size/256.0),256>>>(v_img_d, h_img_d, edge_img_d, size);
        cudaDeviceSynchronize();
        edge_end = std::chrono::high_resolution_clock::now();
    }

    //Saving image
    unsigned char* edge_final_h = new unsigned char[size];
    cudaMemcpy(edge_final_h, edge_img_d, size, cudaMemcpyDeviceToHost);
    stbi_write_png(output_img, width, height, 1, edge_final_h, width);

    //free memory
    std::cout << "Saved image to " << output_img << "\n";
    stbi_image_free(img);
    delete[] img_gray_h;
    delete[] padded_img_h;
    delete[] edge_final_h;
    cudaFree(img_d);
    cudaFree(img_gray_d);
    cudaFree(padded_img_d);
    cudaFree(v_img_d);
    cudaFree(h_img_d);
    cudaFree(edge_img_d);

    //displaying the
    auto gray_ms   = std::chrono::duration<double,std::milli>(t1 - t0).count();
    auto copy_ms   = std::chrono::duration<double,std::milli>(t1b - t1).count();
    auto pad_ms    = std::chrono::duration<double,std::milli>(t2 - t1b).count();
    auto edge_ms   = std::chrono::duration<double,std::milli>(edge_end - edge_start).count();
    auto total_ms  = gray_ms + copy_ms + pad_ms + edge_ms;

    std::cout
      << "Timings (ms):\n"
      << "  grayscale kernel: " << gray_ms   << "\n"
      << "  host copy back:   " << copy_ms   << "\n"
      << "  padding:          " << pad_ms    << "\n"
      << "  edge detect:      " << edge_ms   << "\n"
      << "  ---------------------------\n"
      << "  total:            " << total_ms  << "\n"
      << std::endl;


    return 0;
}
