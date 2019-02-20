#include<THC/THC.h>

#include "nms_cuda.h"

extern THCState *state;

void fill_size(int64_t* input_size,THCudaTensor* input,int dims){
  for(int i=0;i<dims;i++){
    input_size[i]=THCudaTensor_size(state,input,i);
  }
}

void fill_long_size(int64_t* input_size,THCudaLongTensor* input,int dims){
  for(int i=0;i<dims;i++){
    input_size[i]=THCudaLongTensor_size(state,input,i);
  }
}



void nms(THCudaTensor* bbox, THCudaTensor* mask,float thresh){
  float* bbox_data = THCudaTensor_data(state,bbox);
  float* mask_data=THCudaTensor_data(state,mask);
  cudaStream_t stream = THCState_getCurrentStream(state);
  int64_t bbox_size[3];
  fill_long_size(bbox_size,bbox,3);
  int64_t mask_size[2];
  fill_size(mask_size,mask,2);
  nms_cuda(bbox_data,bbox_size,mask_data,mask_size,thresh,stream);
}
