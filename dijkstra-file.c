/* File:     mpi_io.c
 * Purpose:  Implement I/O functions that will be useful in an
 *           an MPI implementation of Dijkstra's algorithm.
 *           In particular, the program creates an MPI_Datatype
 *           that can be used to implement input and output of
 *           a matrix that is distributed by block columns.  It
 *           also implements input and output functions that use
 *           this datatype.  Finally, it implements a function
 *           that prints out a process' submatrix as a string.
 *           This makes it more likely that printing the submatrix
 *           assigned to one process will be printed without
 *           interruption by another process.
 *
 * Compile:  mpicc -g -Wall -o mpi_io mpi_io.c
 * Run:      mpiexec -n <p> ./mpi_io (on lab machines)
 *           csmpiexec -n <p> ./mpi_io (on the penguin cluster)
 *
 * Input:    n:  the number of rows and the number of columns
 *               in the matrix
 *           mat:  the matrix:  note that INFINITY should be
 *               input as 1000000
 * Output:   The submatrix assigned to each process and the
 *           complete matrix printed from process 0.  Both
 *           print "i" instead of 1000000 for infinity.
 *
 * Notes:
 * 1.  The number of processes, p, should evenly divide n.
 * 2.  You should free the MPI_Datatype object created by
 *     the program with a call to MPI_Type_free:  see the
 *     main function.
 * 3.  Example:  Suppose the matrix is
 *
 *        0 1 2 3
 *        4 0 5 6
 *        7 8 0 9
 *        8 7 6 0
 *
 *     Then if there are two processes, the matrix will be
 *     distributed as follows:
 *
 *        Proc 0:  0 1    Proc 1:  2 3
 *                 4 0             5 6
 *                 7 8             0 9
 *                 8 7             6 0
 */
 //read in stdin is too slow, so create another version for reading files
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_STRING 10000
#define INFINITY 1000000

int Read_n(int my_rank, MPI_Comm comm);
int read_n_from_file(int my_rank, MPI_Comm comm, char *filename);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void Read_matrix(int loc_mat[], int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);
void read_matrix_from_file(int loc_mat[], int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm, char *filename);
void Print_local_matrix(int loc_mat[], int n, int loc_n, int my_rank);
void Print_matrix(int loc_mat[], int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);
double Dijkstra(int loc_mat [], int loc_dist[], int loc_known[], int loc_pred[], int my_min[], int glb_min[],
        int n, int loc_n, int my_rank, MPI_Comm comm);
void Find_min_dist(int loc_dist[], int loc_known[], int my_min[], int loc_n, int my_rank);
void Print_paths(int pred[], int n);
void Print_dists(int dist[], int n);
void usage(void);
int main(int argc, char* argv[]) {
   if (argc != 2) usage();
   int *loc_mat, *loc_dist, *loc_known, *loc_pred;
   int my_min[2];
   int *glb_dist, *glb_pred;
   int glb_min[2];
   double loc_start, loc_end, loc_elapsed;
   double all_start, all_end, all_elapsed;
   int n, loc_n, p, my_rank;
   //FILE *file_p;
   MPI_Comm comm;
   MPI_Datatype blk_col_mpi_t;


   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &p);
   MPI_Comm_rank(comm, &my_rank);

   //count the total time
   if (my_rank == 0) all_start = MPI_Wtime();
   //n = Read_n(my_rank, comm);
   n = read_n_from_file(my_rank, comm, argv[1]);
   //printf("%d\n", n);
   loc_n = n/p;
   loc_mat = malloc(n*loc_n*sizeof(int));
   loc_known = malloc(loc_n*sizeof(int));
   loc_dist = malloc(loc_n*sizeof(int));
   loc_pred = malloc(loc_n*sizeof(int));

   /* Build the special MPI_Datatype before doing matrix I/O */
   blk_col_mpi_t = Build_blk_col_type(n, loc_n);

   //Read_matrix(loc_mat, n, loc_n, blk_col_mpi_t, my_rank, comm);
   read_matrix_from_file(loc_mat, n, loc_n, blk_col_mpi_t, my_rank, comm, argv[1]);
   //Print_local_matrix(loc_mat, n, loc_n, my_rank);
   //Print_matrix(loc_mat, n, loc_n, blk_col_mpi_t, my_rank, comm);

   MPI_Barrier(comm);
   if (my_rank == 0) loc_start = MPI_Wtime();
//*****************************************************************************
   double allReduce_time = Dijkstra(loc_mat, loc_dist, loc_known, loc_pred, my_min, glb_min, n,loc_n,my_rank, comm);
//*****************************************************************************
   if (my_rank == 0){
     loc_end = MPI_Wtime();
     loc_elapsed = loc_end - loc_start;
   }
   //MPI_Reduce(&loc_elapsed, &glb_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

   if (my_rank == 0) all_end = MPI_Wtime();
   //Recording results
   if(my_rank==0){
      glb_dist = malloc(n*sizeof(int));
      glb_pred = malloc(n*sizeof(int));
   }
   MPI_Gather(loc_dist,loc_n,MPI_INT,glb_dist,loc_n,MPI_INT,0,comm);
   MPI_Gather(loc_pred,loc_n,MPI_INT,glb_pred,loc_n,MPI_INT,0,comm);
   if (my_rank ==0){
      Print_dists(glb_dist, n);
      Print_paths(glb_pred, n);
    }
   free(loc_mat); free(loc_dist);free(loc_pred); free(loc_known);
   if (my_rank==0){
      free(glb_dist); free(glb_pred);
      all_elapsed  = all_end - all_start;
      printf("Elapsed time is %f seconds\n",loc_elapsed);
      printf("allReduce_time %f\n", allReduce_time);
      printf("global_time %f\n", all_elapsed);


      FILE *file_out = fopen("result.out.new","a");
      fprintf(file_out, "%s %d %f %f %f\n", argv[1], p, loc_elapsed, allReduce_time, all_elapsed);
      fclose(file_out);
    }
   /* When you're done with the MPI_Datatype, free it */
   MPI_Type_free(&blk_col_mpi_t);

   MPI_Finalize();
   return 0;
}  /* main */

void Find_min_dist(int loc_dist[], int loc_known[], int my_min[], int loc_n, int my_rank)
{
   int loc_u = -1;
   int loc_min = INFINITY;
   for (int i = 0; i < loc_n; ++i){
     if (loc_known[i] == 1) continue;
     if (loc_dist[i] < loc_min){
        loc_min = loc_dist[i];
        loc_u = i;
     }
   }
   my_min[0] = loc_min;
   my_min[1] = loc_n*my_rank + loc_u;
}
double Dijkstra(int loc_mat [], int loc_dist[], int loc_known[], int loc_pred[], int my_min[], int glb_min[],
        int n, int loc_n, int my_rank, MPI_Comm comm)
{
    //*Initialization
    double loc_start, loc_end;
    for(int i = 0; i < loc_n; i++) {
      loc_dist[i] = loc_mat[0*loc_n+i];
      loc_pred[i] = 0;
      loc_known[i] = 0;
    }
    if(my_rank==0) loc_known[0]=1;
    for (int i = 1; i < n; ++i){
      //iterate for n-1 times
      Find_min_dist(loc_dist, loc_known, my_min, loc_n, my_rank);
      loc_start = MPI_Wtime();
      MPI_Allreduce(my_min,glb_min,1,MPI_2INT,MPI_MINLOC,comm);
      loc_end = MPI_Wtime();
      //we have to update known[], but we don't know which part it belongs
      if (my_rank == glb_min[1]/loc_n){
          loc_known[glb_min[1]%loc_n] = 1;
      }
      //if (my_rank == 0) printf("Round %d: choose vertex %d\n",i,glb_min[1]);
      //now update dist and known;
      for (int i = 0; i < loc_n; ++i){
          if (loc_known[i]) continue;
          if (loc_dist[i] > glb_min[0] + loc_mat[glb_min[1]*loc_n + i]){
            //if the distance 0->updated-one->i smaller than 0->i
            loc_dist[i] = glb_min[0] + loc_mat[glb_min[1]*loc_n + i];
            loc_pred[i] = glb_min[1];
          }
      }
    }
    return loc_end - loc_start;
}
void Print_dists(int dist[], int n) {
   int v;

   printf("  v    dist 0->v\n");
   printf("----   ---------\n");

   for (v = 1; v < n; v++)
      printf("%3d       %4d\n", v, dist[v]);
   printf("\n");
} /* Print_dists */
void Print_paths(int pred[], int n) {
   int v, w, *path, count, i;

   path =  malloc(n*sizeof(int));

   printf("  v     Path 0->v\n");
   printf("----    ---------\n");
   for (v = 1; v < n; v++) {
      printf("%3d:    ", v);
      count = 0;
      w = v;
      while (w != 0) {
         path[count] = w;
         count++;
         w = pred[w];
      }
      printf("0 ");
      for (i = count-1; i >= 0; i--)
         printf("%d ", path[i]);
      printf("\n");
   }

   free(path);
}  /* Print_paths */

/*---------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Read in the number of rows in the matrix on process 0
 *            and broadcast this value to the other processes
 * In args:   my_rank:  the calling process' rank
 *            comm:  Communicator containing all calling processes
 * Ret val:   n:  the number of rows in the matrix
 */
int Read_n(int my_rank, MPI_Comm comm) {
   int n;

   if (my_rank == 0)
      scanf("%d", &n);
   MPI_Bcast(&n, 1, MPI_INT, 0, comm);
   return n;
}  /* Read_n */
int read_n_from_file(int my_rank, MPI_Comm comm, char *filename)
{
  int n;

  if (my_rank == 0){
    FILE *file_p = fopen(filename, "r");
    char *a;
    a = malloc(10*sizeof(char));
    fgets(a, 10, file_p);
    //printf("%s\n", a);
    n = atoi(a);
    fclose(file_p);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, comm);

  return n;
}

/*---------------------------------------------------------------------
 * Function:  Build_blk_col_type
 * Purpose:   Build an MPI_Datatype that represents a block column of
 *            a matrix
 * In args:   n:  number of rows in the matrix and the block column
 *            loc_n = n/p:  number cols in the block column
 * Ret val:   blk_col_mpi_t:  MPI_Datatype that represents a block
 *            column
 */
MPI_Datatype Build_blk_col_type(int n, int loc_n) {
   MPI_Aint lb, extent;
   MPI_Datatype block_mpi_t;
   MPI_Datatype first_bc_mpi_t;
   MPI_Datatype blk_col_mpi_t;

   MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
   MPI_Type_get_extent(block_mpi_t, &lb, &extent);

   MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);
   MPI_Type_create_resized(first_bc_mpi_t, lb, extent,
         &blk_col_mpi_t);
   MPI_Type_commit(&blk_col_mpi_t);

   MPI_Type_free(&block_mpi_t);
   MPI_Type_free(&first_bc_mpi_t);

   return blk_col_mpi_t;
}  /* Build_blk_col_type */

/*---------------------------------------------------------------------
 * Function:  Read_matrix
 * Purpose:   Read in an nxn matrix of ints on process 0, and
 *            distribute it among the processes so that each
 *            process gets a block column with n rows and n/p
 *            columns
 * In args:   n:  the number of rows in the matrix and the submatrices
 *            loc_n = n/p:  the number of columns in the submatrices
 *            blk_col_mpi_t:  the MPI_Datatype used on process 0
 *            my_rank:  the caller's rank in comm
 *            comm:  Communicator consisting of all the processes
 * Out arg:   loc_mat:  the calling process' submatrix (needs to be
 *               allocated by the caller)
 */
void Read_matrix(int loc_mat[], int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) {
   int* mat = NULL, i, j;

   if (my_rank == 0) {
      mat = malloc(n*n*sizeof(int));
      for (i = 0; i < n; i++)
         for (j = 0; j < n; j++)
            scanf("%d", &mat[i*n + j]);
   }

   MPI_Scatter(mat, 1, blk_col_mpi_t,
           loc_mat, n*loc_n, MPI_INT, 0, comm);

   if (my_rank == 0) free(mat);
}  /* Read_matrix */

void read_matrix_from_file(int loc_mat[], int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm, char *filename) {
   int* mat = NULL, i, j;

   if (my_rank == 0) {
      mat = malloc(n*n*sizeof(int));
      FILE* file_q = fopen(filename, "r");
      char *a;

      a = malloc(10*sizeof(char));
      fgets(a, 10, file_q);
      //printf("%s\n", a);
      free(a);
      for (i = 0; i < n; i++){
         for (j = 0; j < n; j++){
            //scanf("%d", &mat[i*n + j]);

            fscanf(file_q, "%d ", &mat[i*n + j]);
            //printf("%d %d %d\n", i, j,mat[i*n + j]);
          }
        }
        /*
        for (int i = 0; i < n; ++i){
          for (int j = 0; j < n; ++j){
            printf("%d ",mat[i*n + j]);
          }
          printf("\n");
        }
        */
        fclose(file_q);
   }
   MPI_Scatter(mat, 1, blk_col_mpi_t,
           loc_mat, n*loc_n, MPI_INT, 0, comm);

   if (my_rank == 0) free(mat);
}  /* Read_matrix */
/*---------------------------------------------------------------------
 * Function:  Print_local_matrix
 * Purpose:   Store a process' submatrix as a string and print the
 *            string.  Printing as a string reduces the chance
 *            that another process' output will interrupt the output.
 *            from the calling process.
 * In args:   loc_mat:  the calling process' submatrix
 *            n:  the number of rows in the submatrix
 *            loc_n:  the number of cols in the submatrix
 *            my_rank:  the calling process' rank
 */
void Print_local_matrix(int loc_mat[], int n, int loc_n, int my_rank) {
   char temp[MAX_STRING];
   char *cp = temp;
   int i, j;

   sprintf(cp, "Proc %d >\n", my_rank);
   cp = temp + strlen(temp);
   for (i = 0; i < n; i++) {
      for (j = 0; j < loc_n; j++) {
         if (loc_mat[i*loc_n + j] == INFINITY)
            sprintf(cp, " i ");
         else
            sprintf(cp, "%2d ", loc_mat[i*loc_n + j]);
         cp = temp + strlen(temp);
      }
      sprintf(cp, "\n");
      cp = temp + strlen(temp);
   }

   printf("%s\n", temp);
}  /* Print_local_matrix */


/*---------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print the matrix that's been distributed among the
 *            processes.
 * In args:   loc_mat:  the calling process' submatrix
 *            n:  number of rows in the matrix and the submatrices
 *            loc_n:  the number of cols in the submatrix
 *            blk_col_mpi_t:  MPI_Datatype used on process 0 to
 *               receive a process' submatrix
 *            my_rank:  the calling process' rank
 *            comm:  Communicator consisting of all the processes
 */
void Print_matrix(int loc_mat[], int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) {
   int* mat = NULL, i, j;

   if (my_rank == 0) mat = malloc(n*n*sizeof(int));
   MPI_Gather(loc_mat, n*loc_n, MPI_INT,
         mat, 1, blk_col_mpi_t, 0, comm);
   if (my_rank == 0) {
      for (i = 0; i < n; i++) {
         for (j = 0; j < n; j++)
            if (mat[i*n + j] == INFINITY)
               printf(" i ");
            else
               printf("%2d ", mat[i*n + j]);
         printf("\n");
      }
      free(mat);
   }
}  /* Print_matrix */
void usage(void)
{
  fprintf(stderr, "Please specify the input filename\n");
  exit(1);
}
