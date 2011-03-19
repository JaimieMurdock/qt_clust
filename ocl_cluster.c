#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "readline.h"
#include <omp.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

#define GENES 1024 
#define THRESH 259 
#define POPSIZE 28000 
#define OFFSET 1000
#define PRINT_ELTS 1
#define DEBUG 0

float std2[GENES];
int G[POPSIZE*GENES];

float distance(int agent) {
    // Create the two input vectors
    int i, genesMem = GENES / 4, popMem = POPSIZE;

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("vector_add_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
            &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            GENES * sizeof(int), NULL, &ret);
    printf("allocated a\n");
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            GENES * POPSIZE * sizeof(int), NULL, &ret);
    printf("allocated b\n");
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            POPSIZE * sizeof(float), NULL, &ret);
    printf("allocated c\n");
    cl_mem std2_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            GENES * sizeof(float), NULL, &ret);
    printf("allocated d\n");
    cl_mem genes_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            sizeof(int), NULL, &ret);
    printf("allocated e\n");

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            GENES *  sizeof(int), &G[agent], 0, NULL, NULL);
    printf("allocated %d\n", (int)ret);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
            GENES * POPSIZE * sizeof(int), G, 0, NULL, NULL);
    printf("allocated %d\n", (int)ret);
    ret = clEnqueueWriteBuffer(command_queue, std2_mem_obj, CL_TRUE, 0, 
            GENES * sizeof(float), std2, 0, NULL, NULL);
    printf("allocated %d\n", (int)ret);
    ret = clEnqueueWriteBuffer(command_queue, genes_mem_obj, CL_TRUE, 0, 
            sizeof(int), &genesMem, 0, NULL, NULL);
    printf("allocated %d\n", (int)ret);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);
    printf("program %d\n", (int)ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    printf("build\n");

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    printf("build\n");

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&std2_mem_obj);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&genes_mem_obj);
    
    // Execute the OpenCL kernel on the list
    size_t global_item_size = (size_t)(GENES * POPSIZE / 4); // Process the entire lists
    printf("global %d\n", (int)global_item_size);
    size_t local_item_size = 1024; // Process one item at a time
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);

    // Read the memory buffer C on the device to the local variable C
    float *C = (float*)malloc(sizeof(float) * POPSIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
            POPSIZE * sizeof(float), C, 0, NULL, NULL);

    // Display the result to the screen
    float output = 0; int gene, a;
    for(a = 0; a < POPSIZE; a++) {
        /*
        output = 0;
        for (gene = 0; gene < GENES; gene++) {
            // tmp = (a * GENES) + gene;
             printf("(GENE %d) %d - %d = %f\n", tmp,
                G[tmp], G[(GENES * agent) + gene], C[tmp]);
            output += C[(a * GENES) + gene];
        }
        */
        printf("%d -> %d: %f\n", agent, a, C[a]);
    }

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(C);
    return output;
}

void print_vector(int v[], int size) {
	int i;
	printf("| [");
	for (i =0; i < size; i++) {
		printf("%3d ", v[i]);
	}
	printf("]\n");
}

void print_float_vector(float v[], int size) {
	int i;
	printf("(");
	for (i =0; i < size; i++) {
		printf(" %f ", v[i]);
	}
	printf(")\n");
}

int sum_arr(int a[], int ELTS) { 
	int i, sum; sum = 0;
	for (i = 0; i < ELTS; i++)
		sum += a[i];
	return sum;
}


int min_indexf(float a[], int ELTS) { 
	int i, max; max = 0;
	for (i = 1; i < ELTS; i++)
		if ((-1 * a[i]) > (-1* a[max])) {
			max = i;
        }
	return max;
}

bool clust[POPSIZE][POPSIZE];

int cluster_length(int n, int ELTS) { 
	int i, sum; sum = 0;
	for (i = 0; i < ELTS; i++)
		if (clust[n][i] == true) sum++;
	return sum;
}

void build_cluster(int i, int POP) {
    bool *AC = clust[i];
	int j, lastPick;
    float max_dist[POP]; 
	for (j = 0; j < POP; j++) {
		max_dist[j] = 0; AC[j] = false;
	}
	lastPick = i; AC[i] = true; max_dist[i] = THRESH + 1;
    if (DEBUG) printf("%dC ", i);
	
	bool flag, flag2; float d; int pick; flag = true;
	while (flag) {
		for (j = 0; j < POP; j++) {
			if (max_dist[j] < THRESH) {
				//d = distance();
                if (DEBUG) printf("%d -> %d: %f (cur: %f) \n", lastPick, j, d, max_dist[j]);
				if (d > max_dist[j]) max_dist[j] = d;
                if (DEBUG) printf("%d -> %d: %f (cur: %f) \n", i, j, d, max_dist[j]);
			}
		}

		pick = min_indexf(max_dist, POP);
		if (max_dist[pick] > THRESH) {
			flag = false;
		} else {
			lastPick = pick; AC[pick] = true; max_dist[pick] = THRESH + 1;
			
			flag2 = true;
			for (j = 0; j < POP; j++) {
				if (flag2 && (max_dist[j] < THRESH)) {
					flag2 = false; break;
				}
			}
			flag = !flag2;
		}
	}
    if (DEBUG) printf("%dC | (len %d)\n", i, cluster_length(i, POP));
}

int ids[POPSIZE];

int cluster_all(int POP, int clusterId) {
	int i, j, size, k, big = 0, bigi = 0;

    if (DEBUG) printf("beginning cluster %d. genome size: %d\n", clusterId, POP);
    
    #pragma omp parallel for
	for (i = 0; i < POP; i++) {
		build_cluster(i, POP);
	}

    for (i=0; i < POP; i++) {
		size = cluster_length(i, POP);
		if (size > big) {
			bigi = i;
			big = size;
		}
    }
	
	printf("cluster %d (%d elts)", clusterId, cluster_length(bigi, POP));
    if (PRINT_ELTS) {
        printf(" : ");
        for (i=0; i<POP; i++) {
            if (clust[bigi][i]) printf("%d ", ids[i]);
        }
    }
    printf("\n");

	//remove biggest cluster's elements from the pool of genomes
    j=0;
    /*
	for (i = 0; i < POP; i++) {
        if (DEBUG) printf("%d || moving agent %d -> %d\n", clusterId, i, j);
        if (!clust[bigi][i]) {
            memmove(G[j], G[i], GENES*sizeof(float));
            k = ids[i];
            ids[j] = k;
            if (DEBUG) printf("%d || success!\n", clusterId);
            j++;
        }    
    }
    */
    return POP-big;
    /*
	for (i = 0; i < POP; i++) {
		if (!overlap(clust[bigi], clust[i]))
			
			printf("%d > %d | %d\n", bigi, i, overlap(clust[bigi], clust[i], POP));
	}	
    */
}



int overlap(bool x[], bool y[], int POP) {
	int i;
	for (i=0; i < POP; i++){
		//printf("%d || %d & %d\n", i, x[i], y[i]);
		if (x[i] && y[i]) return 1;
	}	
	return 0;
}

void load_genome(int id, int ind) {
    FILE *fp;
    char filename[100];
    sprintf(filename, "/home/jaimie/polyworld/run/genome/genome_%d.txt", id);

    fp = fopen(filename, "r");
    if (!fp) {
    	printf("Unable to open file \"%s\"\n", filename);
    	exit(1);
    }


    char* s; int i;
    for (i=0;  i< GENES; i++) {
        s = readline(fp);
        G[(GENES*ind) + i] = atoi(s);
        ids[ind] = id;
    }

    fclose(fp);
}

int *mean(int SIZE) {
    int i, j;

    static int avg[GENES];
    for (j=0; j<GENES; j++)
        avg[j] = 0;

    int offset = 0;
    for (i=0; i<SIZE; i++) {
        for (j=0; j<GENES; j++)
            avg[j] += G[offset + j];
        offset += GENES;
    }

    for (i=0; i < GENES; i++)
        avg[i] = avg[i] / SIZE;

    return avg;
}

float *stddev(int SIZE, int avg[GENES]) {
    int i, j;

    static float std[GENES];
    for (j=0; j<GENES; j++)
        std[j] = 0;

    int offset = 0;
    for (i=0; i<SIZE; i++) {
        for (j=0; j<GENES; j++)
            std[j] += (G[offset + j] - avg[j]) * (G[offset + j] - avg[j]);
        offset += GENES;
    }

    for (i=0; i < GENES; i++)
        std[i] = std[i] / SIZE;

    //print_float_vector(std, GENES);
    return std;
}
/*
float *stdscore(int genome[GENES], int avg[GENES], float std[GENES]) {
    int i, j;

    static float score[GENES];
    for (j=0; j<GENES; j++)
        score[j] = 0;

    for (j=0; j<GENES; j++)
        score[j] = (genome[j] - avg[j]) / std[j];

    //print_float_vector(score, GENES);
    return score;
}
*/

int main() {
    int POP = POPSIZE;
	int  i, j;

    int *genome;
	for (i = 0; i < POP; i++) {
        load_genome(i+OFFSET, i);
        // printf("loading %d (into %d)\n", i+OFFSET,i);
	}

    // print_vector(G, GENES * POPSIZE);
    int *avg = mean(POP);
    float *std = stddev(POP, avg);
    for (i =0; i < GENES; i++)
        std2[i] = std[i];
    print_vector(avg, GENES);
    print_float_vector(std2, GENES);
    
    distance(3);
    
    /*
    i=0;
    while (POP > 0) {
        POP = cluster_all(POP, i);
        i++;
    }
    */

    return POP;
}
