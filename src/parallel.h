#pragma once

#include <mpi.h>
#include <omp.h>

class Parallel {
 public:
  static Parallel& get_instance() {
    static Parallel instance;
    return instance;
  }

  static int get_n_procs() { return get_instance().n_procs; }

  static int get_proc_id() { return get_instance().proc_id; }

  static int get_n_threads() { return get_instance().n_threads; }

  static void barrier() { MPI_Barrier(MPI_COMM_WORLD); }

  template <class T>
  static void broadcast(T& t);

 private:
  Parallel() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    n_threads = omp_get_max_threads();
  }

  ~Parallel() { MPI_Finalize(); }

  int n_procs;

  int proc_id;

  int n_threads;
};
