/**
  ***********************************************************************************
  * @file   : TiledMatrixMultiplication.cu
  * @brief  : Main program body
  *         : Lab 4: Tiled Matrix Multiplication
  *         : CS-4981/031 
  * @date   : SEP 29 2021
  * @author : Julian Singkham
  ***********************************************************************************
  * @attention
  *
  * This program conducts matrix multiplication with dimensions specified by the user
  * in the program arguments and each element is set to 1. 
  *
  * Inputs:
  *    m = # rows in matrix A
  *    n = # columns in matrix A and rows in matrix B
  *    k = # columns in matrix B
  *
  * Matrix A and B are created and initialized to 1 in the kernel. The GPU
  * implementation conducts matrix multiplication in global memory. The tiled
  * version conducts matrix multiplicatio using shared memory. Between the two
  * implementations the output matrix is zeroed. In both implementations, only the 
  * multiplicaiton kernel call is timed.
  *
  ***********************************************************************************
**/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const int TILE_WIDTH = 32;
#define RGB_COMPONENT_COLOR 255
typedef struct {
	unsigned char red,green,blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel *data;
} PPMImage;

__constant__ float filter[9] = {-1,-1,-1,
								-1,8,-1,
								-1,-1,-1};
//====================================CUDA Kernel=====================================
/**
  * @brief Multiplies two vectors to an output vector using shared memory
  * 
  * @param output: Pointer to the the device vector that stores the multiplicaiton.
  * @param mata: Pointer to the the device vector that holds matrix A.
  * @param matb: Pointer to the the device vector that holds matrix B.
  * @param m: Number of rows in matrix A.
  * @param n: Number of cols in matrix A and rows in matrix B.
  * @param k: Number of cols in matrix B.
  * 
  * @retval NONE
  */
  __global__ void VectorTiledConvKernel(PPMImage* conv, PPMImage* img){
	__shared__ int sharePixelBox[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; 
	int by = blockIdx.y;
    int tx = threadIdx.x; 
	int ty = threadIdx.y;
    int y = by * blockDim.y + ty;
    int x = bx * blockDim.x + tx;
    for(int i = 0; i < ceil((float) img->x/TILE_WIDTH); ++i){
		// Must check for boundary condition in each memory call
		if(y < img->y && x * TILE_WIDTH+tx < img->x){
			// Generate neighbors list
			int neighbors[9] = {(y-1)*img->x+(x-1), (y-1)*img->x+x, (y-1)*img->x+(x+1),
								 y*img->x+(x-1),     y*img->x+x,     y*img->x+(x+1),
								(y+1)*img->x+(x-1), (y+1)*img->x+x, (y+1)*img->x+(x+1)};
			//__syncthreads();

			// Edge detection
			for(int i=0; i<9; i++){
				// Left bound check
				if(x == 0){
					neighbors[0] = -1;
					neighbors[3] = -1;
					neighbors[6] = -1;
				}
				// Right bound check
				if(x == (img->x-1)){
					neighbors[2] = -1;
					neighbors[5] = -1;
					neighbors[8] = -1;
				}
				// Upper bound check
				if(y == 0){
					neighbors[0] = -1;
					neighbors[1] = -1;
					neighbors[2] = -1;
				}
				// Lower bound check
				if(y == (img->y-1)){
					neighbors[6] = -1;
					neighbors[7] = -1;
					neighbors[8] = -1;
				}	
			}

			// Apply filter to neighbors
			int sum_red = 0;
			int sum_green = 0;
			int sum_blue = 0;
			for(int j=0; j<9; j++){
				if(neighbors[j] != -1){
					sum_red   += img->data[neighbors[j]].red*filter[j];
					sum_green += img->data[neighbors[j]].green*filter[j];
					sum_blue  += img->data[neighbors[j]].blue*filter[j];
				}
			}

			// Clamp RGB values to 0-255
			if(sum_red > 255)
				sum_red = 255;
			if(sum_red < 0)
				sum_red = 0;
			if(sum_green > 255)
				sum_green = 255;
			if(sum_green < 0)
				sum_green = 0;
			if(sum_blue > 255)
				sum_blue = 255;
			if(sum_blue < 0)
				sum_blue = 0;
			
			// Set pixel color
			conv->data[neighbors[4]].red = sum_red;
			conv->data[neighbors[4]].green = sum_green;
			conv->data[neighbors[4]].blue = sum_blue;
			}
		}
    }
}

//======================================Methods=======================================
PPMImage* readPPM(const char *filename){
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;

	//open PPM file for reading
	fp = fopen(filename, "r");
	

	//read image format
	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	//check the image format
	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	//alloc memory form image
	img = (PPMImage *)malloc(sizeof(PPMImage));

	//check for comments
	c = getc(fp);
	while (c == '#') {
	while (getc(fp) != '\n') ;
	c = getc(fp);
	}

	ungetc(c, fp);

	//read image size information
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	//read rgb component
	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
		exit(1);
	}

	//check rgb component depth
	if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n') ;
	//memory allocation for pixel data
	img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//read pixel data from file
	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}
	

void writePPM(const char *filename, PPMImage *img)
{
	FILE *fp;
	//open file for output
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//write the header file
	//image format
	fprintf(fp, "P6\n");

	//image size
	fprintf(fp, "%d %d\n",img->x,img->y);

	// rgb component depth
	fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

	// pixel data
	fwrite(img->data, 3 * img->x, img->y, fp);
	fclose(fp);
}

PPMImage* convolution(PPMImage *img){
	// Create the modified image
	PPMImage *conv;
	conv = (PPMImage *)malloc(sizeof(PPMImage));
	conv->x = img->x;
	conv->y = img->y;
	conv->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));
	
	// row
	for(int y=0; y<img->y; y++){
		// col
		for(int x=0; x<img->x; x++){
			// Get pixel index
			int neighbors[9] = {(y-1)*img->x+(x-1), (y-1)*img->x+x, (y-1)*img->x+(x+1),
								y*img->x+(x-1),    y*img->x+x,    y*img->x+(x+1),
								(y+1)*img->x+(x-1), (y+1)*img->x+x, (y+1)*img->x+(x+1)};
			// Edge detection
			for(int i=0; i<9; i++){
				// Left bound check
				if(x == 0){
					neighbors[0] = -1;
					neighbors[3] = -1;
					neighbors[6] = -1;
				}
				// Right bound check
				if(x == (img->x-1)){
					neighbors[2] = -1;
					neighbors[5] = -1;
					neighbors[8] = -1;
				}
				// Upper bound check
				if(y == 0){
					neighbors[0] = -1;
					neighbors[1] = -1;
					neighbors[2] = -1;
				}
				// Lower bound check
				if(y == (img->y-1)){
					neighbors[6] = -1;
					neighbors[7] = -1;
					neighbors[8] = -1;
				}	
			}
			
			// Apply filter to neighbors
			int sum_red = 0;
			int sum_green = 0;
			int sum_blue = 0;
			for(int i=0; i<9; i++){
				if(neighbors[i] != -1){
					sum_red   += img->data[neighbors[i]].red*filter[i];
					sum_green += img->data[neighbors[i]].green*filter[i];
					sum_blue  += img->data[neighbors[i]].blue*filter[i];
				}
			}

			// Output filter to pixel
			if(sum_red > 255)
				sum_red = 255;
			if(sum_red < 0)
				sum_red = 0;
			if(sum_green > 255)
				sum_green = 255;
			if(sum_green < 0)
				sum_green = 0;
			if(sum_blue > 255)
				sum_blue = 255;
			if(sum_blue < 0)
				sum_blue = 0;
			
			conv->data[neighbors[4]].red = sum_red;
			conv->data[neighbors[4]].green = sum_green;
			conv->data[neighbors[4]].blue = sum_blue;
		}
	}
	return conv;
}

/**
  * @brief Calculates how long the CPU was operating
  * 
  * @param start: Pointer to the cpu time the operation started.
  * @param end: Pointer to the cpu time the operation ended.
  * 
  * @retval The length of time the cpu was operating in milliseconds
  */
  float cpu_time(timespec* start, timespec* end){
	return ((1e9*end->tv_sec + end->tv_nsec) - 
			(1e9*start->tv_sec + start->tv_nsec))/1e6;
}

/**
  * @brief Conducts matrix multiplication between two matrices and saves the output
  * to another matrix. This also compares the compute time between the GPU and GPU 
  * with tiled multiplication.
  * 
  * @param argv: Pointer to the given arguements
  *     args[1] = Matrix A row count
  *     args[2] = Matrix A col/Matrix B row count
  *     args[3] = Matrix B col count
  * 
  * @retval NONE
  */
int main(int argc, char* argv[]){
	printf("**************Initializing CPU Variables**************\n");
	PPMImage *image;
	PPMImage *conv;
	timespec ts, te; // Used to calculate CPU implementation
	image = readPPM(argv[1]);
	int image_size = sizeof(PPMImage);
	int data_size = image->x * image->y * sizeof(PPMPixel);
	int coordinate_size = sizeof(int);
	

	printf("******************CPU Implementation******************\n");
	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	conv = convolution(image);
	clock_gettime(CLOCK_MONOTONIC_RAW, &te);
	printf("\nCPU ellapsed time: %f milliseconds\n\n", cpu_time(&ts, &te));
    writePPM(argv[2], conv);

	printf("**************Initializing GPU Variables**************\n");
	// Create CUDA start and stop events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	PPMImage *d_image;
	PPMImage *d_conv;
	PPMImage *output;
	
	// Create GPU output image
	output = (PPMImage *)malloc(sizeof(PPMImage));
	output->x = image->x;
	output->y = image->y;
	output->data = (PPMPixel*)malloc(data_size);

	// Allocate memory on the GPU
	cudaMalloc((void **)&d_image, image_size);
	cudaMalloc((void **)&d_image->data, data_size);
	cudaMalloc((void **)&d_conv, image_size);
	cudaMalloc((void **)&d_conv->data, data_size);

	// Copy host variables to GPU FIXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
	cudaMemcpy(d_image->x, image->x, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_image->y, image->y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_image->data, image->data, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_conv->x, conv->x, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_conv->y, conv->y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_conv->data, conv->data, data_size, cudaMemcpyHostToDevice);

	// Calculate grid size and block dimension
	unsigned int grid_rows = ceil((float) image->y/TILE_WIDTH);
    unsigned int grid_cols = ceil((float) image->x/TILE_WIDTH);
    dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	printf("******************GPU Implementation******************");
	cudaEventRecord(start);
	VectorTiledConvKernel<<<dimGrid, dimBlock>>>(d_conv, d_image);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0; //declare a variable to store runtime
 	cudaEventElapsedTime(&milliseconds, start, stop); //get the elapsed time

	// Copy device output to GPU
	cudaMemcpy(output->data, d_conv->data, data_size, cudaMemcpyDeviceToHost);

	writePPM(argv[3], output);
	printf("\nGPU ellapsed time: %f milliseconds\n\n", milliseconds);

	// Free all host allocated memory
	free(image->data);
	free(image);
	free(conv->data);
	free(conv);
	free(output->data);
	free(output);

	// Free all GPU allocated memory
	cudaFree(d_image->data);
	cudaFree(d_image);
	cudaFree(d_conv->data);
	cudaFree(d_conv);
	
	return 0;
}





