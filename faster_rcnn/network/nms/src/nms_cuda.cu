#include "nms_cuda.h"

__device__ int get_index(){
  int blockId=blockIdx.y*gridDim.x+blockIdx.x;
  int threadId=blockId*blockDim.x+threadIdx.x;
  return threadId;
}

__device__ int get_block_prefix(){
    int blockId=blockIdx.y*gridDim.x+blockIdx.x;
    int block_prefix=blockId*blockDim.x;
    return block_prefix;
}

__device__ int get_thread(){
  return threadIdx.x;
}

__global__ void nms_cuda_imp(float* bbox, int64_t* bbox_size,
   float* mask,int64_t* mask_size,float thresh){
     int index=get_index();
     int block_prefix=get_block_prefix();
     for(int i=0;i<bbox_size[1];++i){
       __syncthreads();
       if(index>=bbox_size[0]*bbox_size[1]){
         continue;
       }
       if(i>=get_thread()){
         continue;
       }
       if(mask[block_prefix+i]==0){
         continue;
       }
       if(mask[index]==0){
         continue;
       }
       if(bbox[7*(block_prefix+i)+6]!=bbox[7*(index)+6]){
         continue;
       }
       float x11=bbox[7*(block_prefix+i)+0];
       float y11=bbox[7*(block_prefix+i)+1];
       float x12=bbox[7*(block_prefix+i)+2];
       float y12=bbox[7*(block_prefix+i)+3];
       float x21=bbox[7*(index)+0];
       float y21=bbox[7*(index)+1];
       float x22=bbox[7*(index)+2];
       float y22=bbox[7*(index)+3];
       float areas_u=(x12-x11)*(y12-y11)+(x22-x21)*(y22-y21);
       float max_x1=((x11>=x21)?x11:x21);
       float max_y1=((y11>=y21)?y11:y21);
       float min_x2=((x12<=x22)?x12:x22);
       float min_y2=((y12<=y22)?y12:y22);
       float w=min_x2-max_x1;
       w=(w>=0?w:0);
       float h=min_y2-max_y1;
       h=(h>=0?h:0);
       float areas_n=w*h;
       if(areas_u-areas_n==0){
         continue;
       }
       float iou=areas_n/(areas_u-areas_n);
       if(iou>thresh){
         mask[index]=0;
       }
     }
   }

void cuda_cpy(int64_t* from,int64_t** to,int size){
  cudaMalloc((void**)to,size*sizeof(int64_t));
  cudaMemcpy(*to,from,size*sizeof(int64_t),cudaMemcpyHostToDevice);
}

void nms_cuda(float* bbox, int64_t* bbox_size,
   float* mask,int64_t* mask_size,float thresh,cudaStream_t stream){
  int d1=1;
  int d2=1;
  if(bbox_size[0]>512){
    d2=(bbox_size[0]+511)/512;
    d1=512;
  }else{
    d1=bbox_size[0];
    d2=1;
  }
  dim3 batch(d1,d2,1);
  dim3 thread(bbox_size[1],1,1);
  int64_t* bbox_size_cuda;
  int64_t* mask_size_cuda;
  cuda_cpy(bbox_size,&bbox_size_cuda,3);
  cuda_cpy(mask_size,&mask_size_cuda,2);
  nms_cuda_imp<<<batch,thread,0,stream>>>(bbox,bbox_size_cuda,mask,mask_size_cuda,thresh);

  cudaFree(bbox_size_cuda);
  cudaFree(mask_size_cuda);
}
