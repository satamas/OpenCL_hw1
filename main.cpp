#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl.hpp>

using namespace std;

inline size_t pow2roundup(size_t x) {
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

void prefix_sum(float *input_array, float *output_array, size_t n) {
    memset(output_array, 0, n * sizeof(float));
    size_t buffers_size = pow2roundup(n);
    vector<cl::Platform> platforms;
    vector<cl::Device> devices;

    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);
        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0]);

        ifstream vector_add_file("prefix_sum.cl");
        if (vector_add_file.fail()) {
            cout << "Can't open data file" << endl;
        }
        std::string source((std::istreambuf_iterator<char>(vector_add_file)),
                std::istreambuf_iterator<char>());

        cl::Program program(context, source);
        program.build(devices);

        cl::Buffer data(context, CL_MEM_READ_ONLY, sizeof(float) * buffers_size);

        queue.enqueueWriteBuffer(data, (cl_bool) true, 0, n * sizeof(float), input_array);


        std::vector<cl::Event> step_complete_events;

        for (int offset = 1; buffers_size / (offset * 2) >= 256; offset *= 2) {
            cl::CommandQueue queue1(context, devices[0]);
            cl::Event step_complete_event;
            cl::Kernel kernel(program, "prefix_sum_reduction");
            kernel.setArg(0, data);
            kernel.setArg(1, sizeof(int), &buffers_size);
            kernel.setArg(2, sizeof(int), &offset);
            queue1.enqueueNDRangeKernel(kernel, NULL, buffers_size / offset, 256, &step_complete_events, &step_complete_event);
            step_complete_events.push_back(step_complete_event);
        }

        if (buffers_size < 512) {
            int offset = 1;
            cl::CommandQueue queue1(context, devices[0]);
            cl::Event step_complete_event;
            cl::Kernel kernel(program, "prefix_sum_reduction");
            kernel.setArg(0, data);
            kernel.setArg(1, sizeof(int), &buffers_size);
            kernel.setArg(2, sizeof(int), &offset);
            queue1.enqueueNDRangeKernel(kernel, NULL, 256, 256, &step_complete_events, &step_complete_event);
            step_complete_events.push_back(step_complete_event);
        }

        {
            int offset = buffers_size / 2;
            cl::CommandQueue queue1(context, devices[0]);
            cl::Event step_complete_event;
            cl::Kernel kernel(program, "prefix_sum_down_sweep");
            kernel.setArg(0, data);
            kernel.setArg(1, sizeof(int), &buffers_size);
            kernel.setArg(2, sizeof(int), &offset);
            queue1.enqueueNDRangeKernel(kernel, NULL, 256, 256, &step_complete_events, &step_complete_event);
            step_complete_events.push_back(step_complete_event);
        }

        for (int offset = buffers_size / 1024; offset > 0; offset /= 2) {
            cl::CommandQueue queue1(context, devices[0]);
            cl::Event step_complete_event;
            cl::Kernel kernel(program, "prefix_sum_down_sweep");
            kernel.setArg(0, data);
            kernel.setArg(1, sizeof(int), &buffers_size);
            kernel.setArg(2, sizeof(int), &offset);
            queue1.enqueueNDRangeKernel(kernel, NULL, buffers_size / offset, 256, &step_complete_events, &step_complete_event);
            step_complete_events.push_back(step_complete_event);
        }


        queue.enqueueReadBuffer(data, (cl_bool) true, 0, n * sizeof(float), output_array, &step_complete_events);
        queue.finish();
        cout << endl;
    } catch (cl::Error e) {
        cout << endl << e.what() << " : " << e.err() << endl;
    }
}


int main() {

    ifstream in("input.txt");
    size_t size;
    in >> size;

    float *input_array = new float[size];
    for (int i = 0; i < size; ++i) {
        in >> input_array[i];
    }
    cout << endl;

    float *output_array = new float[size];

    prefix_sum(input_array, output_array, size);


    //ofstream out("output.txt");
    for (int i = 0; i < size; ++i) {
        cout << output_array[i] << ' ';
    }
    cout << endl;

    delete[] input_array;
    delete[] output_array;

    return 0;
}