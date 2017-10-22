#include <stdio.h>
#include <stdlib.h>
#include <string.h>


FILE* getMatrixSize(char *filename, int *size);
void read_file(FILE *file_p, int *matrix, int size);
void usage(void);
int main(int argc, char* argv[]){
   if (argc != 2) usage();
   printf("the file name is %s\n", argv[1]);
   int * buf;
   int *size = malloc(sizeof(int));
   FILE *file_p = getMatrixSize(argv[1], size);
   printf("%d\n", *size);
   buf = malloc(sizeof(int)*(*size)*(*size));
   read_file(file_p, buf, *size);
   for (int i = 0; i < *size; ++i){
     for (int j = 0; j < *size; ++j){
       printf("%d ", buf[i*(*size)+j]);
     }
     printf("\n");
   }
}
void usage(void)
{
  fprintf(stderr, "Please specify the input filename\n");
  exit(1);
}
FILE* getMatrixSize(char *filename, int *size)
{
    FILE *file_p = fopen(filename, "r");
    char *a;
    a = malloc(10*sizeof(char));
    fgets(a, 10, file_p);
    //printf("%s\n", a);
    *size = atoi(a);
    return file_p;
}
void read_file(FILE *file_p, int *matrix, int size)
{
  //FILE *file_p = fopen(filename, w);
  for (int i = 0; i < size; ++i){
    for (int j = 0; j < size; ++j){
      fscanf(file_p, "%d ", &matrix[i*size + j]);
    }
    //getchar();
  }
}
