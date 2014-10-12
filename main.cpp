#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl.h>
#include <string.h>
#include <assert.h>


using namespace std;

void convolution(float * A, float * B, float * C, size_t n, size_t m){
    memset(C, 0, n*n* sizeof(float));

    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem a_memobj = NULL;
    cl_mem b_memobj = NULL;
    cl_mem c_memobj = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    ifstream vector_add_file("convolution.cl");
    if (vector_add_file.fail()) {
        cout << "Can't open data file" << endl;
    }
    std::string vector_add_src((std::istreambuf_iterator<char>(vector_add_file)),
            std::istreambuf_iterator<char>());

    char const *src[] = {vector_add_src.c_str()};
    size_t const src_length[] = {(size_t const) vector_add_src.size()};

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    command_queue = clCreateCommandQueueWithProperties(context, device_id, nullptr, &ret);

    a_memobj = clCreateBuffer(context, CL_MEM_READ_ONLY, n * n * sizeof(float), NULL, &ret);
    b_memobj = clCreateBuffer(context, CL_MEM_READ_ONLY, m * m * sizeof(float), NULL, &ret);
    c_memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(float), NULL, &ret);


    ret = clEnqueueWriteBuffer(command_queue, a_memobj, (cl_bool) true, 0, n * n * sizeof(float), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_memobj, (cl_bool) true, 0, m * m * sizeof(float), B, 0, NULL, NULL);

    program = clCreateProgramWithSource(context, 1, src, src_length, &ret);

    ret = clBuildProgram(program, ret_num_devices, &device_id, NULL, NULL, NULL);

    kernel = clCreateKernel(program, "convolution", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &a_memobj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &b_memobj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &c_memobj);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *) &n);
    ret = clSetKernelArg(kernel, 4, sizeof(int), (void *) &m);

    size_t const global_work_size = n*n;
    size_t const local_work_size = 1;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    assert(ret == 0);

    ret = clEnqueueReadBuffer(command_queue, c_memobj, (cl_bool) true, 0, n * n * sizeof(float), C, 0, NULL, NULL);



    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(c_memobj);
    ret = clReleaseMemObject(b_memobj);
    ret = clReleaseMemObject(a_memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
}


int main() {

    ifstream in("input.txt");
    size_t sizeA, sizeB;
    in >> sizeA;
    in >> sizeB;

    float *A = new float[sizeA * sizeA];
    for(int i = 0; i < sizeA; ++i){
        for(int j = 0; j < sizeA; ++j){
            in >> A[j + i*sizeA];
        }
    }
    float *B = new float[sizeB * sizeB];
    for(int i = 0; i < sizeA; ++i){
        for(int j = 0; j < sizeA; ++j){
            in >> B[j + i*sizeA];
        }
    }
    float *C = new float[sizeA * sizeA];

    convolution(A, B, C, sizeA, sizeB);


    ofstream out("output.txt");
    for (int i = 0; i < sizeA; ++i) {
        for (int j = 0; j < sizeA; ++j) {
            out << C[j + i * sizeA] << ' ';
        }
        out << endl;
    }

    return 0;
}