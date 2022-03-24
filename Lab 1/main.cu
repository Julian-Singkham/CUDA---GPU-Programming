#include <stdio.h>

int main(int argc, char* argv[]){

	cudaError_t error;
	int count;	//stores the number of CUDA compatible devices

	error = cudaGetDeviceCount(&count);	//get the number of devices with compute capability >= 2.0

	if(error != cudaSuccess){	//if there is an error getting the device count
		printf("\nERROR calling cudaGetDeviceCount()\n");	//display an error message
		return error;	//return the error
	}

	printf("%s", argv[1]);
	FILE *f = fopen(argv[1], "w");


	fprintf(f, "\nNumber of CUDA devices: %d\n", count);
	
	fprintf(f, "\n-------------------------Device Properties-------------------------\n");
	
	for (int dev = 0; dev < count; dev++){
		fprintf(f, "\n**************************** GPU %d ****************************\n\n", dev);
		cudaDeviceProp prop;
    		//get the properties for the first CUDA device
		error = cudaGetDeviceProperties(&prop, dev);	
		
		if(error != cudaSuccess){
			//if there is an error getting the device properties
		        fprintf(f, "ERROR calling cudaGetDeviceProperties()\n");	//display an error message
			return error;	//return the error
		}

		fprintf(f, "Name:                  %s\n", prop.name);
		fprintf(f, "Global Memory:         %f Gb\n", (double)prop.totalGlobalMem/1024/1000000);
       		fprintf(f, "Shared Memory/block:   %f Kb\n", (double)prop.sharedMemPerBlock/1024);
		fprintf(f, "Registers/block:       %d\n", prop.regsPerBlock);
		fprintf(f, "Warp Size:             %d\n", prop.warpSize);
		fprintf(f, "Max Threads/block:     %d\n", prop.maxThreadsPerBlock);
		fprintf(f, "Max Block Dimensions:  [ %d x %d x %d ]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		fprintf(f, "Max Grid Dimensions:   [ %d x %d x %d ]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		fprintf(f, "Constant Memory:       %f Kb\n", (double)prop.totalConstMem/1024);
		fprintf(f, "Compute Capability:    %d.%d\n", prop.major, prop.minor);
	        fprintf(f, "Clock Rate:            %f GHz\n", (double)prop.clockRate/1000000);
       }

	return 0;
}


