#include "THC/THC.h"

#ifdef __cplusplus
extern "C" {
#endif

void nms_cuda(float* bbox, int64_t* bbox_size,float* mask,int64_t* mask_size,float thresh, cudaStream_t stream);

#ifdef __cplusplus
}

#endif
