#include <stdio.h>
#include <stdlib.h>

#include "net.h"

void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

void convolution_3x3_boxblur_RGB_lowlevel(const ncnn::Mat& rgb, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 2;

    ncnn::Layer* op = ncnn::create_layer("ConvolutionDepthWise");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 3);// num_output
    pd.set(1, 3);// kernel_w
    pd.set(5, 0);// bias_term
    pd.set(6, 3*3*3);// weight_data_size
    pd.set(7, 3);// group

    op->load_param(pd);

    // set weights
    ncnn::Mat weights[1];
    weights[0].create(3*3*3);// weight_data

    for (int i=0; i<3*3*3; i++)
    {
        weights[0][i] = 1.f / 9;
    }

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    // forward
    op->forward(rgb, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

void convolution_3x3_boxblur_RGB_highlevel(const ncnn::Mat& rgb, ncnn::Mat& out)
{
    const char param_str[] =
        "7767517\n"
        "2 2\n"
        "Input input 0 1 in\n"
        "ConvolutionDepthWise convdw 1 1 in out 0=3 1=3 5=0 6=27 7=3\n";

    float model_mem[1 + 27];
    model_mem[0] = 0.f; // tag
    for (int i=0; i<3*3*3; i++)
    {
        model_mem[1 + i] = 1.f / 9;
    }

    ncnn::Net net;
    net.opt.num_threads = 2;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)model_mem);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in", rgb);
    ex.extract("out", out);
}

int main(int argc, char **argv)
{
    ncnn::Mat input(6, 6, 3);

    // fill random
    for (int i = 0; i < input.total(); i++)
    {
        input[i] = rand() % 10;
    }
    
    ncnn::Mat out1;
    convolution_3x3_boxblur_RGB_lowlevel(input, out1);
    
    ncnn::Mat out2;
    convolution_3x3_boxblur_RGB_highlevel(input, out2);

    printf("Use low level API...\n");
    pretty_print(out1);
    printf("Use high level API...\n");
    pretty_print(out2);

    return 0;
}
