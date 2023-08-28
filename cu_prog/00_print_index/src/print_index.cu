#include <cuda.h>
#include <stdio.h>

__global__ void print_idx_kernel(){
  printf("block idx: (%3d, %3d, %3d), thread idx: (%3d, %3d, %3d)\n",
       blockIdx.z, blockIdx.y, blockIdx.x,
       threadIdx.z, threadIdx.y, threadIdx.x);
}

__global__ void print_dim_kernel(){
  printf("grid dimension: (%3d, %3d, %3d), block dimension: (%3d, %3d, %3d)\n",
     gridDim.z, gridDim.y, gridDim.x,
     blockDim.z, blockDim.y, blockDim.x);
}

void print_idx_dim() {
  dim3 block_dim{3, 3, 3};
  dim3 grid_dim{2, 2, 2};
  
  print_dim_kernel<<<grid_dim, block_dim>>>();
  
  cudaDeviceSynchronize();
  
  print_idx_kernel<<<grid_dim, block_dim>>>();
  
  cudaDeviceSynchronize();
}

int main() {

  print_idx_dim();

}
