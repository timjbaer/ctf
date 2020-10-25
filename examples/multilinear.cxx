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
int multl(int64_t     m,
          int64_t     n,
          World &     dw,
          double      sp_T=1.,
          bool        test=true,
          bool        bench=true,
          int         niter=10){
  assert(test || bench);

  int64_t nd = m;
  bool pass = true;
  Vector<int> X(nd, dw, "X"); 
  Vector<int> Y(nd, dw, "Y"); 
  X.fill_random(1, 25);
  Y["i"] = X["i"];
  // Y.fill_random(1, 20);
  Vector<int> W_in(nd, dw, "W_in");
  W_in.fill_random(6, 10);
  Vector<int> res(nd, dw, "res");
  Vector<int> W(nd, dw, "W");
  W["i"] = W_in["i"];
  int64_t lens[2] = {nd, nd};
  int sT = sp_T < 1. ? SP : 0;
  Tensor<int> T(2, sT, lens, dw);
  if (sp_T < 1.)
    T.fill_sp_random(1, 105, sp_T);
  else
    T.fill_random(1, 105);

  Tensor<int> * vec_list[2] = {&Y, &X};

  Bivar_Function<int, int, int> fmv([](int t, int x) {
      return (x * t);
  });

  std::function<int(int, int, int)> f = [](int a, int b, int c) {
    return (a * b * c);
  };
  
  if (test) {
    // Multilinear<int>(&T, vec_list, &W, &fmv);
    res["i"] = T["ij"] * X["j"];
    Multilinear1<int>(&T, vec_list, &W, f);
    int64_t npair;
    Pair<int> * pairs;
    res.get_local_pairs(&npair, &pairs, false, false);
    int *arr;
    arr = (int *)Y.sr->alloc(nd);
    Y.read_all(arr, true);
    for (int64_t i = 0; i < npair; i++) {
      pairs[i].d *= arr[pairs[i].k];
    }
    res.write(npair, pairs);
    res["i"] += W_in["i"];
    res["i"] -= W["i"];
    pass = res.norm2() <= 1.E-6;
    if (dw.rank == 0) {
      if (pass)
        printf("Multilinear test passed T(%lld), sparse: %d\n", nd, (sp_T < 1.));
    }
    Y.sr->dealloc((char *)arr);
    res.sr->pair_dealloc((char *)pairs);
  }
  if (bench) {
    double min_time = DBL_MAX;
    double max_time = 0.0;
    double tot_time = 0.0;
    double times[niter];
    if (dw.rank == 0){
      printf("Starting %d benchmarking multilinear\n", niter);
      initialize_flops_counter();
    }
    Timer_epoch multi("specified multilinear");
    multi.begin();
    for (int i = 0; i < niter; i++){
      double start_time = MPI_Wtime();
      Multilinear<int>(&T, vec_list, &W, &fmv);
      double end_time = MPI_Wtime();
      double iter_time = end_time-start_time;
      times[i] = iter_time;
      tot_time += iter_time;
      if (iter_time < min_time) min_time = iter_time;
      if (iter_time > max_time) max_time = iter_time;
    }
    multi.end();
    if (dw.rank == 0){
      printf("iterations completed, did %ld flops.\n", CTF::get_estimated_flops());
      printf("All iterations times: ");
      for (int i = 0; i < niter; i++){
        printf("%lf ", times[i]);
      }
      printf("\n");
      std::sort(times, times+niter);
      printf("T(%ld) sparse: %d Min time = %lf, Avg time = %lf, Med time = %lf, Max time = %lf\n", nd, (sp_T < 1.), min_time, tot_time/niter, times[niter/2], max_time);
    }
  }
  return pass;
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
  int rank, np, pass, niter, test, bench; 
  int64_t m, n;
  double sp_T;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoll(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 4;
  } else m = 4;

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoll(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;
  
  if (getCmdOption(input_str, input_str+in_num, "-sp_T")){
    sp_T = atof(getCmdOption(input_str, input_str+in_num, "-sp_T"));
    if (sp_T < 0.0 || sp_T > 1.0) sp_T = .2;
  } else sp_T = .2;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 10;
  } else niter = 1;

  if (getCmdOption(input_str, input_str+in_num, "-bench")){
    bench = atoi(getCmdOption(input_str, input_str+in_num, "-bench"));
    if (bench != 0 && bench != 1) bench = 1;
  } else bench = 1;
  
  if (getCmdOption(input_str, input_str+in_num, "-test")){
    test = atoi(getCmdOption(input_str, input_str+in_num, "-test"));
    if (test != 0 && test != 1) test = 1;
  } else test = 1;
  
  {
    World dw(argc, argv);
    if (rank == 0){
      printf("Performing multilinear operation\n");
    }
    pass = multl(m, n, dw, sp_T, test, bench, niter);
    assert(pass);
  }
  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
