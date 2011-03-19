cluster.out : cluster.c
	gcc cluster.c -lm -fopenmp -mcmodel=medium -O2 -pipe -o cluster.out

# OCL build for ATI Hardware 
ocl_host : ocl_cluster.c
	gcc -c -I /opt/stream/include/ ocl_cluster.c -o ocl_cluster.o
	gcc -L /opt/stream/lib/x86_64 -l OpenCL ocl_cluster.o -o ocl_host
