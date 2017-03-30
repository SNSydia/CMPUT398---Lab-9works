#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define BLOCK_SIZE 512 //TODO: You can change this

#define wbCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void blockadd(int* g_aux, int* g_odata, int n){
	int id = blockIdx.x*blockDim.x + threadIdx.x; //Id of the thread within the block

	if (blockIdx.x > 0 && id < n){
		g_odata[id] += g_aux[blockIdx.x];
	}

}

__global__ void split(int *in_d, int *out_d, int length, int shamt) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
	int bit = 0;

	if (index < length) {
		bit = in_d[index] & (1 << shamt);
		//bit = ~bit;

		if (bit > 0)
            bit = 1;
        else
            bit = 0;

		__syncthreads();

		out_d[index] = 1 -bit;

	}

}

__global__ void split2(int *in_d,  int *lb_d,  int length) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	int x = in_d[length - 1] + lb_d[length - 1];
	__syncthreads();

	if (index < length) {
		if (lb_d[index] == 0) {
			__syncthreads();
			int val = in_d[index];
			in_d[index] = index - val + x;

		}
	}

}

__global__ void scatter( int *in_d,  int *index_d,  int *out_d,  int length) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index < length) {

		int val = index_d[index];
		__syncthreads();

		out_d[val] = in_d[index];

	}
}

__global__ void scan(int *g_odata, int *g_idata, int *g_aux, int n){

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float temp[BLOCK_SIZE]; 

	if (i < n){
		temp[threadIdx.x] = g_idata[i];
	}

	for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2){
		__syncthreads();
		float in1 = 0.0;

		if (threadIdx.x >= stride){
			in1 = temp[threadIdx.x - stride];
		}
		__syncthreads();
		temp[threadIdx.x] += in1;
	}

	__syncthreads();

	if (i + 1 < n) g_odata[i + 1] = temp[threadIdx.x];
	g_odata[0] = 0;//MUST DO THIS TO MAKE IT WORK!!!


	if (g_aux != NULL && threadIdx.x == blockDim.x - 1){

		g_aux[blockIdx.x] = g_odata[i + 1];
		g_odata[i + 1] = 0;
	}
}
void swap(int* in, int* out){

    int *tmp;
    tmp = in;
    in = out;
    out = tmp;
}

void recursive_scan(int* deviceOutput, int* deviceInput, int numElements){
	int numBlocks = (numElements / BLOCK_SIZE) + 1;
	if (numBlocks == 1){ 
		dim3 block(BLOCK_SIZE, 1);
		dim3 grid(numBlocks, 1);

		scan << <grid, block >> >(deviceOutput, deviceInput, NULL, numElements);
		cudaDeviceSynchronize();
	}
	else{ 
		int* deviceAux;
		cudaMalloc((void**)&deviceAux, (numBlocks*sizeof(int)));

		int *deviceAuxPass;
		cudaMalloc((void**)&deviceAuxPass, (numBlocks*sizeof(int)));

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid(numBlocks, 1);

		scan << <grid, block >> >(deviceOutput, deviceInput, deviceAux, numElements);
		wbCheck(cudaDeviceSynchronize());


		dim3 grid2(1, 1);
		dim3 block2(numBlocks, 1, 1);

		scan << <grid2, block2 >> >(deviceAuxPass, deviceAux, NULL, numBlocks);
		wbCheck(cudaDeviceSynchronize());

		recursive_scan(deviceAuxPass, deviceAux, numBlocks);

		blockadd << <block2, block >> >(deviceAuxPass, deviceOutput, numElements);
		wbCheck(cudaDeviceSynchronize());

		cudaFree(deviceAux);
		cudaFree(deviceAuxPass);
	}

}

void sort(int* deviceInput, int *deviceOutput, int numElements, int* hostInput)
{
	//TODO: Modify this to complete the functionality of the sort on the deivce
	int numBlocks = (numElements / BLOCK_SIZE) + 3;
	int *help; int *help2;


	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(numBlocks, 1);

	dim3 grid2(1, 1);
	dim3 block2(numBlocks, 1, 1);

	cudaMalloc(&help, sizeof(int)*numElements);
	cudaMalloc(&help2, sizeof(int)*numElements);

	for (int bit = 0; bit < 15; bit++){

		split << <grid, block >> >(deviceInput, deviceOutput, numElements, bit);
		cudaDeviceSynchronize();		
		
		recursive_scan(help2, deviceOutput, numElements);
		//scan << <grid, block >> >(help2, deviceOutput, NULL, numElements);
		cudaDeviceSynchronize();

		/**blockadd << <block2, block >> >(help2, deviceOutput, numElements);
		cudaDeviceSynchronize();**/

		split2 << <grid, block >> >(help2, deviceOutput, numElements);
		cudaDeviceSynchronize();

		scatter << <grid, block >> >(deviceInput, help2, deviceOutput, numElements);
		cudaDeviceSynchronize();

		//swap(deviceInput, deviceOutput);
		int *tmp;
		tmp = deviceInput;
		deviceInput = deviceOutput;
		deviceOutput= tmp;
	}

	/**int *tmp;
	tmp = deviceInput;
	deviceInput = deviceOutput;
	deviceOutput = tmp;**/
}


int main(int argc, char **argv) {
	wbArg_t args;
	int *hostInput;  // The input 1D list
	int *hostOutput; // The output list
	int *deviceInput;
	int *deviceOutput;
	int numElements; // number of elements in the list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (int *)wbImport(wbArg_getInputFile(args, 0), &numElements, "integral_vector");
	cudaHostAlloc(&hostOutput, numElements * sizeof(int), cudaHostAllocDefault);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ", numElements);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(int)));
	wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(int)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Clearing output memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(int)));
	wbTime_stop(GPU, "Clearing output memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(int),
		cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	wbTime_start(Compute, "Performing CUDA computation");
	sort(deviceInput, deviceOutput, numElements, hostInput);
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
		cudaMemcpyDeviceToHost));
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, numElements);

	free(hostInput);
	cudaFreeHost(hostOutput);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
