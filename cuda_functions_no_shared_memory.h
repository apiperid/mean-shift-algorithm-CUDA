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

//          this .h file must be included to file mean_shift_without_shared_memory.cu

#define SIGMA 1

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
   FUNCTION __device__ double Distance(double **y_p,double **x_p,int index,int dim,int x_id)

   getting as input the y_pointers and x_pointers , the index of the thread ,the dimensions of the block 
   and the id of x element in array x

   calculating the distance between y and the x element and returning it.
*/

__device__ double Distance(double **y_p,double **x_p,int index,int dim,int x_id)
{
  double sum=0.0;
  for(int i=0;i<dim;i++)
    sum+=(y_p[index][i]-x_p[x_id][i])*(y_p[index][i]-x_p[x_id][i]);
  return sqrt(sum);
}

/*
   FUNCTION __device__ double kernel_function(double **y_p,double **x_p,int index,int dim,int x_id)

   getting as input the y_pointers and x_pointers , the index of the thread ,the dimensions of the block 
   and the id of x element in array x

   calculating the distance_square between y and the x element and returning the exponential formula exp(-x/2*sigma^2).
*/

__device__ double kernel_function(double **y_p,double **x_p,int index,int dim,int x_id)
{
  double norm_square=0.0;
  for(int i=0;i<dim;i++)
    norm_square+=(y_p[index][i]-x_p[x_id][i])*(y_p[index][i]-x_p[x_id][i]);
  return exp(-norm_square/(2*(SIGMA*SIGMA)));
}

/*
   FUNCTION __device__ void normalize(double **y_n_p,int index,int dim,double denumenator)
  
   getting as input the y_new_pointers, the index of the thread ,the dimensions of the block 
   and the denumenator we will divide with.
   We divide each dimension with the denumenator
   
*/

__device__ void normalize(double **y_n_p,int index,int dim,double denumenator)
{
  for(int i=0;i<dim;i++)
    y_n_p[index][i]=y_n_p[index][i]/denumenator;
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
   FUNCTION __global__ void calculate(double **x_p,double **y_p,double **y_n_p,int el,int dim,int size)
  
   as input we have :
   arrays : x_pointers , y_pointers , y_new_pointers , elements , dimensions , size
   Each thread is responsible for its block ( each thread is a point of array y ).
   for those x which are closer than SIGMA^2 from this point the thread calculates
   the kernel function and updates the dimensions of the elements 
   after that it normalizes the dimensions with the sum of the exponnetials ( numenators ).

*/

__global__ void calculate(double **x_p,double **y_p,double **y_n_p,int el,int dim,int size)
{
  int index=blockIdx.x;
  double numenator;
  double denumenator;
  if(index<el)
  {  
    denumenator=0.0;
    for(int j=0;j<el;j++)
    {
      if(Distance(y_p,x_p,index,dim,j)<=(SIGMA*SIGMA))
      {
        numenator=kernel_function(y_p,x_p,index,dim,j);
        denumenator+=numenator;
        for(int d=0;d<dim;d++)
        {
          y_n_p[index][d]+=numenator*x_p[j][d];
        }
      }
    }
    normalize(y_n_p,index,dim,denumenator);
  }
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
   FUNCTION __global__ void norm_of_m(double *m,int el,int dim,double *error_gpu)
  
   getting as input the m array and the size of the array
   we are calculating the frobenius of the array and putting in to variable error_gpu

   NOTE!!! we do this calculation sequentially

*/

__global__ void norm_of_m(double *m,int el,int dim,double *error_gpu)
{
  error_gpu[0]=0.0;
  for(int i=0;i<el*dim;i++)
    error_gpu[0]+=(m[i]*m[i]);
  error_gpu[0]=sqrt(error_gpu[0]);
}
