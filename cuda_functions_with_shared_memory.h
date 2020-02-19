/*
         Parallel And Distributed
  
         PIPERIDIS ANESTIS
         AEM : 8689

*/

/*
#######################################################################################################
#######################################################################################################
################# FUNCTIONS WHICH WILL BE EXCECUTED ON DEVICE #########################################
#######################################################################################################
#######################################################################################################
*/

//          this .h file must be included to file mean_shift_with_shared_memory.cu

//            NOTE!!!!! IF YOU WANT TO USE ANOTHER DATASET YOU HAVE TO CHANGE THE DEFINED
//            VARIABLES MAX_DIMENSIONS AND MAX_ELEMENTS HERE AND IN THE PROGRAM
//            mean_shift_with_shared_memory.cu


#define SIGMA 1
#define MAX_DIMENSIONS 2
#define MAX_ELEMENTS 600



__device__ double array_for_norm[MAX_ELEMENTS];

/*
  THE NEXT 3 FUNCTIONS ARE JUST INITIALIZATION OF POINTERS AND ARRAYS
*/

__global__ void initialize_x_pointers(double **x_p,double *x,int el,int dim)
{
  for(int i=0;i<el;i++)
    x_p[i]=&x[i*dim];
}

__global__ void initialize_y_pointers(double **y_p,double **y_n_p,double *y,double *y_n,int el,int dim)
{
  for(int i=0;i<el;i++)
  {
    y_p[i]=&y[i*dim];
    y_n_p[i]=&y_n[i*dim];
    for(int j=0;j<dim;j++)
    {
      y_p[i][j]=0.0;
      y_n_p[i][j]=0.0;
    }
  }
}

__global__ void initialize_m_pointers(double **m_p,double *m,int el,int dim)
{
  for(int i=0;i<el;i++)
  {
    m_p[i]=&m[i*dim];
    for(int j=0;j<dim;j++)
      m_p[i][j]=INFINITY;
  }
}

/*
   FUNCTION __global__ void clear_y_new(double *y_n,int size)
  
   getting as input the y_new array and the size of the array
   we are are running so many blocks as the elements and the user give us
   the threads per block
   So each thread clears a specific value of array y_new
   
*/

__global__ void clear_y_new(double *y_n,int size)
{
    int index=threadIdx.x + blockIdx.x * blockDim.x;
    if(index<size)
      y_n[index]=0.0;
}

/*
   FUNCTION __global__ void update(double *y_n,double *y,int size)
  
   getting as input the y_new and y array and the size of the arrays
   we are are running so many blocks as the elements and the user give us
   the threads per block
   So each thread is responsible for one element
   
*/

__global__ void update(double *y_n,double *y,int size)
{
  int index=threadIdx.x + blockIdx.x * blockDim.x;
  if(index<size)
    y[index]=y_n[index];
}

/*
   FUNCTION __global__ void update_m(double *y_n,double *y,double *m,int size)
  
   getting as input the y_new , y and m array and the size of the arrays
   we are are running so many blocks as the elements and the user give us
   the threads per block
   So each thread is responsible for one element and does the calculation
   y_new - y_n
   
*/

__global__ void update_m(double *y_n,double *y,double *m,int size)
{
  int index=threadIdx.x + blockIdx.x * blockDim.x;
  if(index<size)
    m[index]=y_n[index]-y[index];
}

/*
   FUNCTION __global__ void set_y(double *y,double *x,int size)
  
   getting as input the y and x array and the size of the arrays
   blocks are equal to elements and user gives the threads per block
   so each thread copies a value of array x and puts it in a specific
   cell in array y.
   
*/

__global__ void set_y(double *y,double *x,int size)
{
  int index=threadIdx.x + blockIdx.x * blockDim.x;
  if(index<size)
    y[index]=x[index];
}

/*
  FUNCTION __global__ void get_norm_m(double *error_gpu,int el)

  get access to global array : array_for_norm and getting the sum of all its elements
  returning the square root of the sum.
  that is the frobenius norm of array m

  NOTE!!! we run this function sequentially .We gained time by parallelizing the function
  __global__ void update_array_for_norm_m(double *m,int el,int dim)

*/

__global__ void get_norm_m(double *error_gpu,int el)
{
  error_gpu[0]=0.0;
  for(int i=0;i<el;i++)
    error_gpu[0]+=array_for_norm[i];
  error_gpu[0]=sqrt(error_gpu[0]);
}

/*
  FUNCTION __global__ void update_array_for_norm_m(double *m,int el,int dim)

  in this version of program we have the comfort to run the calculating of norm m in parallel.
  run so many blocks as the elements and dimensions threads per block.
  each thread calculates the m(i)^2 and saves it to a shared array.
  After all threads per block do the same the "master thread"( threadIdx.x=0 ) saves the sum 
  of all this to a global array named array_for_norm.

*/
__global__ void update_array_for_norm_m(double *m,int el,int dim)
{
  __shared__ double temp[MAX_DIMENSIONS];
  double sum;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < (el*dim) )
  {
    sum=0.0;
    temp[threadIdx.x]=(m[index]*m[index]);
    __syncthreads();
    //BE SURE WE GOT THE INFORMATION WE NEED
    //THEN ONLY ONE THREAD UPDATES THE GLOBAL ARRAY
    if( threadIdx.x== 0)
    {
      for(int i=0;i<MAX_DIMENSIONS;i++)
        sum+=temp[i];
      //UPDATE THE GLOBAL ARRAY
      array_for_norm[blockIdx.x]=sum;
    }
  }
}

/*
  FUNCTION __device__ double distance_and_kernel(double **y_p,double **x_p,int y_id,int y_offset,int x_id,int x_offset)

  getting as input arrays y_pointers and x_pointers 
  and also it takes which element in y array thread refers to ( variables y_id and y_offset ) 
  and the same with element in x array ( variables x_id and x_offset ).
  this function returns the square of difference between two dimensions .
  we save this result to a __shared__ array.

  we also use it for the kernel calculation because it returns the same value.
  in the first case we take the square root of the result 
  in the second case we take the exp(-x/(2*SIGMA^2))

*/

__device__ double distance_and_kernel(double **y_p,double **x_p,int y_id,int y_offset,int x_id,int x_offset)
{
  return (y_p[y_id][y_offset]-x_p[x_id][x_offset])*(y_p[y_id][y_offset]-x_p[x_id][x_offset]);
}
/*
  FUNCTION __device__ double add(double *array,int size)
   
  just getting an array and its size and return the sum of all its elements
  useful for calucating the distance and the exponential of the kernel function

*/

__device__ double add(double *array,int size)
{
  double sum=0.0;
  for(int i=0;i<size;i++)
    sum+=array[i];
  return sum;
}

/*
  FUNCTION __global__ void calculate(double **y_n_p,double **y_p,double **x_p,int el,int dim,int size,double *error_gpu)


  each thread takes only one dimension and of its block and calculating the difference of its dimension and a dimension
  of a point Xj and saves the result to a shared array with name temp.
  After we check if the Xj point is closer than SIGMA^2 we calculating the numenator for all Xjs and updating 
  the y_new dimension
  After checking all the Xjs it normalizes its dimension

*/
__global__ void calculate(double **y_n_p,double **y_p,double **x_p,int el,int size)
{
  __shared__ double temp[MAX_DIMENSIONS];
  double numenator,denumenator,sum;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < size )
  {
    denumenator=0.0;
    for(int j=0;j<el;j++)
    {
      temp[threadIdx.x]=distance_and_kernel(y_p,x_p,blockIdx.x,threadIdx.x,j,threadIdx.x);
      __syncthreads();
      //WE ARE SURE THAT WE HAVE THE INFORMATION WE NEED TO CONTINUE
      //IF THE Xj POINT IS CLOSE ENOUGH , CONTINUE TO KERNEL
      sum=add(temp,MAX_DIMENSIONS);
      if(sqrt(sum)<=(SIGMA*SIGMA))
      {
        numenator=exp((-sum)/(2*SIGMA*SIGMA));
        denumenator+=numenator;
        y_n_p[blockIdx.x][threadIdx.x]+=(numenator*x_p[j][threadIdx.x]);
      }     
    }
    //NORMALIZE THE DIMENSION
    y_n_p[blockIdx.x][threadIdx.x]=y_n_p[blockIdx.x][threadIdx.x]/denumenator;
  }
}
