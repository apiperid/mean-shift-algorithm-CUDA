/*
       Parallel And Distributed
     
       PIPERIDIS ANESTIS 
       AEM : 8689

*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include "cuda_functions_with_shared_memory.h"

#define MAX_ELEMENTS 600
#define MAX_DIMENSIONS 2
#define EPSILON 0.0001
#define SIGMA 1

/*
  in this program we save the arrays as 1D this because of the cudaMalloc and the cudaMemcpy
*/

double *X,*Y;
int elements,dimensions,threads;
double error_cpu[1] = { INFINITY };

struct timeval startwtime, endwtime;
double seq_time;


void allocate_memory();
void check_input(int argc,char **argv);
void read_data_file(char *file);
void print_X();
void print_Y();
void free_memory();
void create_output_file();
void test_with_matlab();
void test_with_sequential();
double dist(double *array1,double *array2,int length);

int main(int argc,char **argv)
{
  check_input(argc,argv);
  allocate_memory();
  read_data_file(argv[4]);

   /*
     Allocating memory for variables and arrays in GPU using cudaMalloc
     x is just the x array which contains the data 
     y  is the same Y as in sequential
     y_n  means Y_NEW
     m  is exactly the m in sequential
     x_p means x_pointers
     y_p means y_pointers
     y_n_p means y_new_pointers
     m_p means m_pointers
     error_gpu contains the frobenius norm of m
  
     the pointer arrays contains the adrresses of each "row" of array to its is referred

     For example:
     x_p contains the adrresses of array x as:
     x_p[0] = &x[0]
     x_p[1] = &x[2]  if dimensions of each element are 2 ... etc

     so we can have access easier to the arrays
     
  */

  double *error_gpu;
  double *x,*y,*y_n,*m;
  double **x_p,**y_p,**y_n_p,**m_p;
  cudaError_t cuda_error;  

  //ALLOCATING MEMORY AND ALSO CHECKING FOR PROBLEMS
  cuda_error=cudaMalloc((void **)&error_gpu,sizeof(double));
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMalloc\n");
    exit(1);
  }
  cuda_error=cudaMalloc((void **)&x,elements*dimensions*sizeof(double));
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMalloc\n");
    exit(1);
  }
  cuda_error=cudaMalloc((void **)&y,elements*dimensions*sizeof(double));
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMalloc\n");
    exit(1);
  }
  cuda_error=cudaMalloc((void **)&y_n,elements*dimensions*sizeof(double));
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMalloc\n");
    exit(1);
  }
  cuda_error=cudaMalloc((void **)&m,elements*dimensions*sizeof(double));
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMalloc\n");
    exit(1);
  }
  cuda_error=cudaMalloc((void **)&x_p,elements*sizeof(double *));
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMalloc\n");
    exit(1);
  }
  cuda_error=cudaMalloc((void **)&y_p,elements*sizeof(double *));
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMalloc\n");
    exit(1);
  }
  cuda_error=cudaMalloc((void **)&y_n_p,elements*sizeof(double *));
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMalloc\n");
    exit(1);
  }
  cuda_error=cudaMalloc((void **)&m_p,elements*sizeof(double *));
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMalloc\n");
    exit(1);
  }

  //MOVE DATA FROM HOST TO DEVICE
  cuda_error=cudaMemcpy(x,X,elements*dimensions*sizeof(double), cudaMemcpyHostToDevice);
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMemcpy 1\n");
    exit(1);
  }

  //print_X();

  initialize_x_pointers<<<1,1>>>(x_p,x,elements,dimensions);
  initialize_y_pointers<<<1,1>>>(y_p, y_n_p, y, y_n,elements,dimensions);
  initialize_m_pointers<<<1,1>>>(m_p, m,elements,dimensions);

  
  int N=elements*dimensions;
  int M=threads;
  //START COUNTING TIME
  gettimeofday (&startwtime, NULL);
  int iteration=0;
  /*
     in parallel we are doing:
     clearing the y_new
     the formula : running 1 block for each element and argv[3] threads per block
     updating the m array
     updating the y array
     norm_of_m

     EXTRA EXPLANATION:
   
     FOR THE FUNCTIONS:
     clear_y_new
     update_m
     update
     norm_of_m
     WE ARE RUNNING SO MANY BLOCKS AS ELEMENTS
     AND argv[3] threads per block
     
   
     calculate : elements blocks and dimensions thread per block

  */
  set_y<<<(N + M-1) / M,M>>>(y,x,N);
  while(error_cpu[0]>EPSILON)
  {
    iteration++;
    clear_y_new<<<(N + M-1) / M,M>>>(y_n,N);
    calculate<<<elements,dimensions>>>(y_n_p,y_p,x_p,elements,N);
    update_m<<<(N + M-1) / M,M>>>(y_n,y,m,N);
    update<<<(N + M-1) / M,M>>>(y_n,y,N);
    update_array_for_norm_m<<<(N + M-1) / M,M>>>(m,elements,dimensions);
    get_norm_m<<<1,1>>>(error_gpu,elements);
    //TRANSFER THE ERROR TO HOST AND CHECK IT
    cuda_error=cudaMemcpy(error_cpu,error_gpu,sizeof(double), cudaMemcpyDeviceToHost);
    if(cuda_error!=cudaSuccess)
    {
      printf("error in cudaMemcpy 2\n");
      exit(1);
    }
    printf("iteration : %d , Error : %f \n",iteration,error_cpu[0]);
  }
  gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
  printf("\nmean shift CUDA with Shared Memory time = %f\n\n", seq_time);
  //MOVE DATA FROM DEVICE TO HOST
  cuda_error=cudaMemcpy(Y,y_n,elements*dimensions*sizeof(double), cudaMemcpyDeviceToHost);
  if(cuda_error!=cudaSuccess)
  {
    printf("error in cudaMemcpy 3\n");
    exit(1);
  }
  //FREE DEVICE MEMORY
  cudaFree(x);
  cudaFree(y);
  cudaFree(y_n);
  cudaFree(m);
  cudaFree(x_p);
  cudaFree(y_p);
  cudaFree(y_n_p);
  cudaFree(m_p);
  //print_Y();
  create_output_file();
  printf("START TESTING\n\n");
  test_with_matlab();
  test_with_sequential();
  free_memory();

}

/*
#######################################################################################################
#######################################################################################################
################# FUNCTIONS WHICH WILL BE EXCECUTED ON HOST ###########################################
#######################################################################################################
#######################################################################################################


     the explanation of each function is to the mean_shift_sequential.c
*/


  
void check_input(int argc,char **argv)
{
  if(argc!=5)
  {
    printf("Arguments must be 5\n");
    exit(1);
  }
  elements=atoi(argv[1]);
  dimensions=atoi(argv[2]);
  threads=atoi(argv[3]);
  if(elements>MAX_ELEMENTS||elements<=0)
  {
    printf("Elements must be in range [1,%d]\n",MAX_ELEMENTS);
    exit(1);
  }
  if(dimensions<=0||dimensions>MAX_DIMENSIONS)
  {
    printf("Dimensions must be in the range : [1,%d] \n",MAX_DIMENSIONS);
    exit(1);
  }
  if(threads!=dimensions)
  {
    printf("Threads per block be :%d in this case\n",dimensions);
    exit(1);
  }
}

void allocate_memory()
{
  //printf("elements = %d , dimensions = %d , threads = %d \n",elements,dimensions,threads);
  X=(double *)malloc(elements*dimensions*sizeof(double));
  if(X==NULL)
  {
    printf("error allocating memory in cpu\n");
    exit(1); 
  }
  Y=(double *)malloc(elements*dimensions*sizeof(double));
  if(Y==NULL)
  {
    printf("error allocating memory in cpu\n");
    exit(1); 
  }
}

void read_data_file(char *file)
{
  int return_value;
  FILE *fp=fopen(file,"rb");
  if(fp==NULL)
  {
    printf("cannot open file : %s\n",file);
    exit(1);
  }
  fseek(fp,0,SEEK_SET);
  return_value=fread(X,sizeof(double),elements*dimensions,fp);
  if(return_value!=elements*dimensions)
  {
    printf("error reading from file : %s\n",file);
    exit(1);
  }
}

void create_output_file(void)
{
  double temp;
  int write_value;
  FILE *fp=fopen("result_with_shared_memory.bin","wb");
  if(fp==NULL)
  {
    printf("Error opening the file : result_with_shared_memory.bin\n");
    exit(1);
  }
  for(int i=0;i<elements;i++)
  {
    for(int j=0;j<dimensions;j++)
    {
      temp=Y[(i*dimensions)+j];
      write_value=fwrite(&temp,sizeof(double),1,fp); 
      if(write_value!=1)
      {
        printf("error writing in file : result_with_shared_memory.bin\n");
        exit(1);
      }
    }
  }
  fclose(fp);
}

/*
  FUNCTION double dist(double *array1,double *array2,int length)

  calculates the distance between two points

*/

double dist(double *array1,double *array2,int length)
{
  double sum=0.0;
  for(int i=0;i<length;i++)
    sum+=(array1[i]-array2[i])*(array1[i]-array2[i]);
  return sqrt(sum);
}

void test_with_matlab()
{
  FILE *c,*matlab;
  int return_value_c,return_value_matlab,spots;
  double c_array[dimensions],matlab_array[dimensions],average_distance,tmp_distance;
  spots=0;
  average_distance=0.0;
  c=fopen("result_with_shared_memory.bin","rb");
  if(c==NULL)
  { 
    printf("error opening the file : result_with_shared_memory.bin\n");
    exit(1);
  }
  matlab=fopen("result_matlab.bin","rb");
  if(matlab==NULL) 
  {
    printf("error opening the file : result_matlab.bin\n");
    exit(1);
  }
  fseek(c,0,SEEK_END);
  fseek(matlab,0,SEEK_END);
  if(ftell(c)!=ftell(matlab))
  {
    printf("result_with_shared_memory.bin and result_matlab.bin Do not have the same size\n");
    exit(1);
  }
  rewind(c);
  rewind(matlab);
  for(int i=0;i<elements;i++)
  {
    return_value_c=fread(c_array,sizeof(double),dimensions,c);
    return_value_matlab=fread(matlab_array,sizeof(double),dimensions,matlab);
    if((return_value_c!=dimensions)||(return_value_matlab!=dimensions))
    {
      printf("error reading from binary files\n");
      exit(1);
    }
    if((tmp_distance=dist(c_array,matlab_array,dimensions))<=EPSILON)
      spots++;
    average_distance+=tmp_distance;
  } 
  fclose(c);
  fclose(matlab);
  printf("Testing with Matlab \n ");
  printf("Success = %f percent\n",((double)spots/elements)*100);
  printf("Average error comparing with matlab = %f \n\n",average_distance/elements);
}

void test_with_sequential()
{
  FILE *c_seq,*c_cuda;
  int return_value_seq,return_value_cuda,spots;
  double seq_array[dimensions],cuda_array[dimensions],average_distance,tmp_distance;
  spots=0;
  average_distance=0.0;
  c_seq=fopen("result_sequential.bin","rb");
  if(c_seq==NULL)
  { 
    printf("error opening the file : result_sequential.bin\n");
    exit(1);
  }
  c_cuda=fopen("result_with_shared_memory.bin","rb");
  if(c_cuda==NULL) 
  {
    printf("error opening the file : result_with_shared_memory.bin\n");
    exit(1);
  }
  fseek(c_seq,0,SEEK_END);
  fseek(c_cuda,0,SEEK_END);
  if(ftell(c_seq)!=ftell(c_cuda))
  {
    printf("result_sequential.bin and result_with_shared_memory.bin Do not have the same size\n");
    exit(1);
  }
  rewind(c_seq);
  rewind(c_cuda);
  for(int i=0;i<elements;i++)
  {
    return_value_seq=fread(seq_array,sizeof(double),dimensions,c_seq);
    return_value_cuda=fread(cuda_array,sizeof(double),dimensions,c_cuda);
    if((return_value_seq!=dimensions)||(return_value_cuda!=dimensions))
    {
      printf("error reading from binary files\n");
      exit(1);
    }
    if((tmp_distance=dist(seq_array,cuda_array,dimensions))<=EPSILON)
      spots++;
    average_distance+=tmp_distance;
  } 
  fclose(c_seq);
  fclose(c_cuda);
  printf("Testing with Sequential\n");
  printf("Success = %f percent\n",((double)spots/elements)*100);
  printf("Average error comparing with sequential = %f \n\n",average_distance/elements);

}

void print_X()
{
  printf("X array is :\n");
  printf("------------------\n");
  for(int i=0;i<elements;i++)
  { 
    for(int j=0;j<dimensions;j++)
      printf("%f ",X[(i*dimensions)+j]);
    printf("\n");
  }
}

void print_Y()
{
  printf("Y array is :\n");
  printf("------------------\n");
  for(int i=0;i<elements;i++)
  { 
    for(int j=0;j<dimensions;j++)
      printf("%f ",Y[(i*dimensions)+j]);
    printf("\n");
  }
}

void free_memory()
{
  free(X);
  free(Y);
}

