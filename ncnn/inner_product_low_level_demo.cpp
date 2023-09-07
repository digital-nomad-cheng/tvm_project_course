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

void inner_product_lowlevel(const ncnn::Mat& rgb, ncnn::Mat& out, bool use_bias=false)
{
    ncnn::Option opt;
    opt.num_threads = 2;

    ncnn::Layer* op = ncnn::create_layer("InnerProduct");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 3);// num_output
    if (use_bias)
    {
      pd.set(1, 1);// use bias_term
    }
    else 
    {
      pd.set(1, 0);// no bias_term
    }
    pd.set(2, 3*6);// weight_data_size

    op->load_param(pd);

    // set weights
    ncnn::Mat weights[2];
    weights[0].create(3*6);// weight_data
    weights[1].create(3);// bias data
    // demo show how weight data is organized in memory
    for (int j=0; j<3; j++) {
        for (int i=0; i<6; i++)
        {
            weights[0][j * 6 + i] = 1.f / 6 + j * 1.f;
        }
    }
    
    for (int i=0; i<3; i++) 
    {
        weights[1][i] = 1.f;
    }


    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    // forward
    op->forward(rgb, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

int main(int argc, char **argv)
{
    ncnn::Mat input(6);

    // fill random
    for (int i = 0; i < input.total(); i++)
    {
        input[i] = 1.0; // rand() % 10;
    }
    
    ncnn::Mat out1;
    inner_product_lowlevel(input, out1);
    
    ncnn::Mat out2;
    inner_product_lowlevel(input, out2, true);

    printf("Use low level API...\n");
    pretty_print(out1);
    printf("Use low level API with bias...\n");
    pretty_print(out2);
    return 0;
}
