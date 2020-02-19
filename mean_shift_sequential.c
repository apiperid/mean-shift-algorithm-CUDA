/*
       Parallel And Distributed
     
       PIPERIDIS ANESTIS 
       AEM : 8689

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define MAX_ELEMENTS 600
#define MAX_DIMENSIONS 2
#define EPSILON 0.0001
#define SIGMA 1
#define error_check 0.0001


/*
  X array contains the data from the file
  Y array contains the data in iteration k
  Y_NEW contains the data in iteration k+1
  m array contains values of Y_NEW-Y
  elements are the number of elements user gives us
  dimensions is the number of dimensions of each element 
  ERROR is the norm of m ( ||m|| )
*/

double **X,**Y,**Y_NEW;
double **m;
int elements,dimensions;
double ERROR;

struct timeval startwtime, endwtime;
double seq_time;


void allocate_memory(void);
void read_from_file(char *file);
void initialize_Y(void);
void print_X(void);
void print_Y(void);
void print_Y_NEW(void);
void calculate(void);
double norm_of_m(void);
void update(void);
double kernel_function(double *Y_i,double *X_j,int length);
void free_memory(void);
double Distance(double *Y_i,double *X_j,int length);
void normalize(double denumenator,int i);
void clear_Y_NEW(void);
void create_output_file(void);
void test_with_matlab(void);
void update_m(void);


int main(int argc,char **argv)
{
  //CHECKING THE INPUTS
  if(argc!=4)
  {
    printf("Arguments must be 4\n");
    exit(1);
  }
  elements=atoi(argv[1]);
  dimensions=atoi(argv[2]);
  if(elements>MAX_ELEMENTS||elements<=0)
  {
    printf("Elements must be in the range [1,%d]\n",MAX_ELEMENTS);
    exit(1);
  }
  if(dimensions>MAX_DIMENSIONS||dimensions<=0)
  {
    printf("Dimensions must be in the range [1,%d]\n",MAX_DIMENSIONS);
    exit(1);
  }
  //ALLOCATING MEMORY
  allocate_memory();
  //READING FROM BINARY FILE
  read_from_file(argv[3]);
  //START THE ALGORITHM
  gettimeofday (&startwtime, NULL);
  initialize_Y();
  calculate();
  //END OF ALGORITHM
  gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
  printf("\nmean shift Sequential time = %f\n\n", seq_time);
  //print_X();
  //print_Y(); 
  //print_Y_NEW();
  create_output_file();
  printf("TESTING WITH MATLAB\n");
  test_with_matlab();
  free_memory();
}
 
/*
   FUNCTION void calculate(void)

   here we have the algorithm.
   we stop the loop when we get error less than EPSILON

   fisrt we clear the Y_NEW array

   after, for each element of Y array we are searching which elements of 
   X array are close to it with distance less than SIGMA^2.
   for those we are calulating the formula gave us.

   We are updating the Y_NEW array

   normalize it
 
   updating the m array

   finding the frobenius norm
  
   updating the context of Y array

   and going to start again till we have error less than EPSILON

*/

void calculate(void)
{
  double denumenator;
  double numenator;
  int iteration=0;
  ERROR=INFINITY;
  while(ERROR>EPSILON)
  {
    iteration++;
    clear_Y_NEW();
    for(int i=0;i<elements;i++)
    {
      denumenator=0.0;
      for(int j=0;j<elements;j++)
      {
        if(Distance(Y[i],X[j],dimensions)<=(SIGMA*SIGMA))
        {
          numenator=kernel_function(Y[i],X[j],dimensions);
          denumenator+=numenator;
          for(int d=0;d<dimensions;d++)
          {
            Y_NEW[i][d]+=numenator*X[j][d];
          }
        }
      }
      normalize(denumenator,i);
    }
    update_m();
    ERROR=norm_of_m();
    update();
    printf("iterarion %d error %f\n",iteration,ERROR);
  }
}

/*
   FUNCTION void normalize(double denumenator,int i)

   dividing a specific element of array Y_NEW with a denumenator

*/


void normalize(double denumenator,int i)
{
    for(int d=0;d<dimensions;d++)
      Y_NEW[i][d]=Y_NEW[i][d]/denumenator;
}

/*
   FUNCTION double Distance(double *Y_i,double *X_j,int length)

   Getting as input two 1D arrays calculating and returning
   the distance of the two elements.

*/

double Distance(double *Y_i,double *X_j,int length)
{
  double result=0.0;
  for(int i=0;i<length;i++)
    result+=(Y_i[i]-X_j[i])*(Y_i[i]-X_j[i]);
  //printf("Distance is : %f \n",sqrt(result));
  return sqrt(result);
}

/*
   FUNCTION double kernel_function(double *Y_i,double *X_j,int length)

   Getting as input two 1D arrays calculating the euclidean square norm
   and return the exponential.

*/

double kernel_function(double *Y_i,double *X_j,int length)
{
  double norm_square=0.0;
  for(int i=0;i<length;i++)
    norm_square+=(Y_i[i]-X_j[i])*(Y_i[i]-X_j[i]);
  //printf("%f\n",norm_square);
  return exp(-norm_square/(2*(SIGMA*SIGMA)));

}
/*
   FUNCTION void update_m(void)

   giving m array the values of 
   Y_NEW - Y .

*/

void update_m(void)
{
  for(int i=0;i<elements;i++)
    for(int j=0;j<dimensions;j++) 
      m[i][j]=Y_NEW[i][j]-Y[i][j];
}

/*
   FUNCTION double norm_of_m(void)

   finding the "frobenius norm" of array m
   we just getting the square of all elements
   we return the square_root of the sum.

*/

double norm_of_m(void)
{
  double sum_square=0.0;
  for(int i=0;i<elements;i++)
    for(int j=0;j<dimensions;j++)
      sum_square+=(m[i][j]*m[i][j]);
  return sqrt(sum_square);
}

/*
   FUNCTION void update(void)

   changing the context of Y array with this of Y_NEW array 
   in order to continue the algorithm

*/

void update(void)
{
  for(int i=0;i<elements;i++)
  {
    for(int j=0;j<dimensions;j++)
    {
      Y[i][j]=Y_NEW[i][j];
    }
  }
}

/*
   FUNCTION void clear_Y_NEW()

   giving Y_NEW array value 0.0 to all elements.

*/

void clear_Y_NEW()
{
  for(int i=0;i<elements;i++)
    for(int j=0;j<dimensions;j++)
      Y_NEW[i][j]=0.0;
}

/*
   FUNCTION void initialize_Y(void)

   Giving Y array the X array
   this is the first step of the algorithm.

*/

void initialize_Y(void)
{
  for(int i=0;i<elements;i++)
  {
    for(int j=0;j<dimensions;j++)
    {
      Y[i][j]=X[i][j];
    }
  }
}

/*
   FUNCTION void read_from_file(char *file)

   reading the file and saving the data in array X.
*/

void read_from_file(char *file)
{
  FILE *fp=fopen(file,"rb");
  if(fp==NULL)
  {
    printf("error opening the file : %s\n",file);
    exit(1);
  }
  int return_value;
  for(int i=0;i<elements;i++)
  {
    fseek(fp,i*(sizeof(double)*MAX_DIMENSIONS),SEEK_SET);
    return_value=fread(X[i], sizeof(double),dimensions, fp);
    if(return_value!=dimensions)
    {
      printf("error reading from file :%s\n",file);
      exit(1);
    }
  }
  fclose(fp);
}
/*
  FUNCTION void create_output_file(void)

  Creating a binary file named result_sequential.bin and putting into it
  the result of this algorithm.

*/

void create_output_file(void)
{
  double temp;
  int write_value;
  FILE *fp=fopen("result_sequential.bin","wb");
  if(fp==NULL)
  {
    printf("Error opening the file result_sequential.bin\n");
    exit(1);
  }
  for(int i=0;i<elements;i++)
  {
    for(int j=0;j<dimensions;j++)
    {
      temp=Y_NEW[i][j];
      write_value=fwrite(&temp,sizeof(double),1,fp); 
      if(write_value!=1)
      {
        printf("error writing in file : result_sequential.bin\n");
        exit(1);
      }
    }
  }
  fclose(fp);
}

/*
   FUNCTION void test_with_matlab(void)

   in this function we are testing the sequential result with the same result of matlab.
   we take each element from the sequential file and calculating the distance with the same element 
   of the matlab file.
   if distance is less than the error_check we count this element

   the result of this function is a precent ofhow well the algotihm works
   also printing the average error between this file and the matlab file.  
  
*/

void test_with_matlab(void)
{
  FILE *c,*matlab;
  double input_c[dimensions],input_matlab[dimensions],average_distance,tmp_distance;
  average_distance=0.0;
  int return_value_c,return_value_matlab,spots;
  spots=0;
  c=fopen("result_sequential.bin","rb");
  if(c==NULL)
  {
    printf("error opening the file : result_sequential.bin\n");
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
    printf("result_sequential.bin and result_matlab.bin Do not have the same size\n");
    exit(1);
  }
  rewind(c);
  rewind(matlab);
  for(int i=0;i<elements;i++)
  {
    return_value_c=fread(input_c,sizeof(double),dimensions,c);
    return_value_matlab=fread(input_matlab,sizeof(double),dimensions,matlab);
    if((return_value_c!=dimensions)||(return_value_matlab!=dimensions))
    {
      printf("error reading from binary files\n");
      exit(1);
    }
    if((tmp_distance=Distance(input_c,input_matlab,dimensions))<=EPSILON)
      spots++;
    average_distance+=tmp_distance;
  } 
  fclose(c);
  fclose(matlab);
  printf("Success = %f percent\n",((double)spots/elements)*100);
  printf("Average error = %f \n\n",average_distance/elements);
}

/*
  FUNCTION void allocate_memory(void)

  allocating memory for all arrays we will need in this program

*/


void allocate_memory(void)
{
  X=(double **)malloc(elements*sizeof(double *));
  if(X==NULL)
  {
    printf("Cannot Allocate memory\n");
    exit(1);
  }
  for(int i=0;i<elements;i++)
  {
    X[i]=(double *)malloc(dimensions*sizeof(double));
    if(X[i]==NULL)
    {
      printf("Cannot allocate memory\n");
      exit(1);
    }
  }
  Y=(double **)malloc(elements*sizeof(double *));
  if(Y==NULL)
  {
    printf("Cannot Allocate memory\n");
    exit(1);
  }
  for(int i=0;i<elements;i++)
  {
    Y[i]=(double *)malloc(dimensions*sizeof(double));
    if(Y[i]==NULL)
    {
      printf("Cannot Allocate memory\n");
      exit(1);
    }
  }
  Y_NEW=(double **)malloc(elements*sizeof(double *));
  if(Y_NEW==NULL)
  {
    printf("Cannot Allocate memory\n");
    exit(1);
  }
  for(int i=0;i<elements;i++)
  {
    Y_NEW[i]=(double *)malloc(dimensions*sizeof(double));
    if(Y_NEW[i]==NULL)
    {
      printf("Cannot Allocate memory\n");
      exit(1);
    }
  }
  m=(double **)malloc(elements*sizeof(double *));
  if(m==NULL)
  {
    printf("Cannot Allocate memory\n");
    exit(1);
  }
  for(int i=0;i<elements;i++)
  {
    m[i]=(double *)malloc(dimensions*sizeof(double));
    if(m[i]==NULL)
    {
      printf("Cannot Allocate memory\n");
      exit(1);
    }
  }
}

/*
   FUNCTION void print_X(void)

   printing the X array

*/

void print_X(void)
{
  printf("X array is:\n");
  for(int i=0;i<elements;i++)
  {
    for(int j=0;j<dimensions;j++)
    {
      printf("%f   ",X[i][j]);
    }
    printf("\n");
  }
}

/*
   FUNCTION void print_Y(void)

   printing the Y array

*/

void print_Y(void)
{
  printf("Y array is:\n");
  for(int i=0;i<elements;i++)
  {
    for(int j=0;j<dimensions;j++)
    {
      printf("%f   ",Y[i][j]);
    }
    printf("\n");
  }
}

/*
   FUNCTION void print_Y_NEW(void)

   printing the Y_NEW array

*/

void print_Y_NEW(void)
{
  printf("Y_NEW array is:\n");
  for(int i=0;i<elements;i++)
  {
    for(int j=0;j<dimensions;j++)
    {
      printf("%f  ",Y_NEW[i][j]);
    }
    printf("\n");
  }
}

/*
   FUNCTION void free_memory(void)

   free memory we dont need anymore

*/


void free_memory(void)
{
  for(int i=0;i<elements;i++)
  {
    free(X[i]);
    free(Y[i]);
    free(Y_NEW[i]);
    free(m[i]);
  }
  free(X);
  free(Y);
  free(Y_NEW);
  free(m);
}
