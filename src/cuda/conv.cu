#define K 5
#define intile 8
#define outtile 4

__global__ void convNaive(float *Y, float *X, float *F, int x0, int x1, int f0, int f1){

    // Same convolution

    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if(row<x0 && col<x1){

        float res = 0.0f;

        for(int i=0; i<f0; i++){
            for(int j=0; j<f1; j++){
                
                int rid = row - f0/2 + i;
                int cid = col - f1/2 + j;

                if(rid>=0 && rid<x0 && cid>=0 && cid<x1){
                    res += X[rid*x1+ cid]*F[i*f1+j];
                }

            }
        }
        Y[row*x1 + col] = res;
    }

}

__global__ convShared(float *Y, float *X, float *F, int x0, int x1, int f0, int f1){
    
    // indices based on output tile
    int row = threadIdx.y + blockIdx.y*outtile;
    int col = threadIdx.x + blockIdx.x*outtile;

    __shared__ float sm[intile*intile];
    int rid = row-k/2;
    int cid = col-k/2;
    // load into shared memory based on intile
    if(rid>=0 && rid<x0 && cid>=0 && cid<x1){
        sm[threadIdx.y*intile + threadIdx.x] = X[rid*x1 + cid];
    }
    else{
        sm[threadIdx.y*intile + threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    //compute conv from sm
    if(threadIdx.x<outtile && threadIdx.y<outtile && row<x0 && col<x1){
                
        float res = 0.0f;
            for(int i=0; i<f0; i++){
                for(int j=0; j<f1; j++){
                    res += F[i*f1+j]*sm[(row+i)*x1 + col+j];
                }
            }
        Y[row*x1+col] = res;
    } 

}