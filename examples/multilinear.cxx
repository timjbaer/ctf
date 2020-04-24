/** \addtogroup examples 
  * @{ 
  * \defgroup Multilinear 
  * @{ 
  * \brief w_i = f(T_ij, a_i, b_j)
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;


/**
 */
int multl(int     m,
          int     n,
          World & dw){

  int nd = m;
  Vector<int> X(nd, dw); 
  Vector<int> Y(nd, dw); 
  X.fill_random(1, 65);
  Y.fill_random(1, 25);
  Vector<int> W(nd, dw, "W");
  W.fill_random(6, 10);
  int lens[2] = {nd, nd};
  Tensor<int> T(2, lens, dw);
  T.fill_random(1, 105);

  T.print();
  Y.print();
  X.print();

  Tensor<int> * vec_list[2] = {&Y, &X};

  Bivar_Function<int, int, int> fmv([](int t, int x) {
      return (x * t);
  });
  Multilinear<int>(&T, vec_list, &W, &fmv); 
  W.print();
  return 0;
} 


#ifndef TEST_SUITE
char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np, m, n;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 4;
  } else m = 4;

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Performing multilinear operation\n");
    }
    multl(m, n, dw);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
