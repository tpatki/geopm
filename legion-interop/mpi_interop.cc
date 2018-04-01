/* Copyright 2018 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

////////////////////////////////////////////////////////////
// THIS EXAMPLE MUST BE BUILT WITH A VERSION
// OF GASNET CONFIGURED WITH MPI COMPATIBILITY
//
// NOTE THAT GASNET ONLY SUPPORTS MPI-COMPATIBILITY
// ON SOME CONDUITS. CURRENTLY THESE ARE IBV, GEMINI,
// ARIES, MXM, and OFI. IF YOU WOULD LIKE ADDITIONAL
// CONDUITS SUPPORTED PLEASE CONTACT THE MAINTAINERS
// OF GASNET.
//
// Note: there is a way to use this example with the
// MPI conduit, but you have to have a version of 
// MPI that supports MPI_THREAD_MULTIPLE. See the 
// macro GASNET_CONDUIT_MPI below.
////////////////////////////////////////////////////////////

#include <cstdio>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <geopm.h>
#include "legion.h"

using namespace Legion;

enum TaskID
{
  TOP_LEVEL_TASK_ID,
  MPI_INTEROP_TASK_ID,
  WORKER_TASK_ID,
};

// Here is our global MPI-Legion handshake
// You can have as many of these as you 
// want but the common case is just to
// have one per Legion-MPI rank pair
MPILegionHandshake handshake;

// Have a global static number of iterations for
// this example, but you can easily configure it
// from command line arguments which get passed 
// to both MPI and Legion
const int total_iterations = 5;
const int factorA = 10000000; // both of these should be larger to take any time
const int factorB = 10000000;

int worker_task(const Task *task, 
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
  int err = 0;
  uint64_t rid = *(const uint64_t*)task->args;
  printf("Legion Doing Work in Rank %lld\n", 
          task->parent_task->index_point[0]);
  err = geopm_prof_enter(rid);
  if (err) {std::cout<<"Something is wrong 5"<<std::endl;}
  int y = 1;
  for (int i=0;i<factorB*(1+task->parent_task->index_point[0]);i++) {
    y = 3+(y%100);
  }
  err = geopm_prof_exit(rid);
  if (err) {std::cout<<"Something is wrong 6"<<std::endl;}
  return y;
}

void mpi_interop_task(const Task *task, 
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  printf("Hello from Legion MPI-Interop Task %lld\n", task->index_point[0]);

  uint64_t rid;
  uint64_t Leg_rid;
  int err;

  char child_str[256];
  sprintf(child_str, "child_of_task_%lld", task->index_point[0]);
  err = geopm_prof_region(      child_str,
                                GEOPM_REGION_HINT_COMPUTE,
                                &rid);
  if (err) {std::cout<<"Something is wrong 0"<<std::endl;}

  char str[256];
  sprintf(str, "task_%lld", task->index_point[0]);
  err = geopm_prof_region(      str,
                                GEOPM_REGION_HINT_COMPUTE,
                                &Leg_rid);
  if (err) {std::cout<<"Something is wrong 1"<<std::endl;}

  for (int i = 0; i < total_iterations; i++)
  {
    // Launch our worker task
    TaskLauncher worker_launcher(WORKER_TASK_ID, TaskArgument(&rid,sizeof(rid)));
    Future f = runtime->execute_task(ctx, worker_launcher);
    f.get_result<int>();

//    err = geopm_prof_epoch();
//    if (err) {std::cout<<"Something is wrong 2"<<std::endl;}
    err = geopm_prof_enter(Leg_rid);
    if (err) {std::cout<<"Something is wrong 3"<<std::endl;}
    int x = 1;
    for (int i=0;i<factorA*(1+task->index_point[0]);i++) {
      x = (2*x+3) % 2037;
    }
    std::cout<<"use x for something: x="<<x<<std::endl;
    //sleep(2);  // to prevent epoch from being too short
    err = geopm_prof_exit(Leg_rid);
    if (err) {std::cout<<"Something is wrong 4"<<std::endl;}
    std::cout<<"End TS legion task: "<<task->index_point[0]<<std::endl;
  }
  //handshake.legion_handoff_to_mpi(); // this isn't actually required, MPI will wait for legion runtime to shutdown even if we don't have the mpi rank wait for this
}

void top_level_task(const Task *task, 
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  printf("Hello from Legion Top-Level Task\n");
  // Both the application and Legion mappers have access to
  // the mappings between MPI Ranks and Legion address spaces
  // The reverse mapping goes the other way
  const std::map<int,AddressSpace> &forward_mapping = 
    runtime->find_forward_MPI_mapping();
  for (std::map<int,AddressSpace>::const_iterator it = 
        forward_mapping.begin(); it != forward_mapping.end(); it++)
    printf("MPI Rank %d maps to Legion Address Space %d\n", 
            it->first, it->second);

  int rank = -1, size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // Do a must epoch launch to align with the number of MPI ranks
  MustEpochLauncher must_epoch_launcher;
  Rect<1> launch_bounds(0,size - 1);
  ArgumentMap args_map;
  IndexLauncher index_launcher(MPI_INTEROP_TASK_ID, launch_bounds, 
                               TaskArgument(NULL, 0), args_map);
  must_epoch_launcher.add_index_task(index_launcher);
  runtime->execute_must_epoch(ctx, must_epoch_launcher);
}

int main(int argc, char **argv)
{
#ifdef GASNET_CONDUIT_MPI
  // The GASNet MPI conduit requires special start-up
  // in order to handle MPI calls from multiple threads
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  // If you fail this assertion, then your version of MPI
  // does not support calls from multiple threads and you 
  // cannot use the GASNet MPI conduit
  if (provided < MPI_THREAD_MULTIPLE)
    printf("ERROR: Your implementation of MPI does not support "
           "MPI_THREAD_MULTIPLE which is required for use of the "
           "GASNet MPI conduit with the Legion-MPI Interop!\n");
  assert(provided == MPI_THREAD_MULTIPLE);
#else
  // Perform MPI start-up like normal for most GASNet conduits
  MPI_Init(&argc, &argv);
#endif

  int rank = -1, size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Hello from MPI process %d of %d\n", rank, size);

  // Configure the Legion runtime with the rank of this process
  Runtime::configure_MPI_interoperability(rank);
  // Register our task variants
  {
    TaskVariantRegistrar top_level_registrar(TOP_LEVEL_TASK_ID);
    top_level_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(top_level_registrar, 
                                                      "Top Level Task");
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  }
  {
    TaskVariantRegistrar mpi_interop_registrar(MPI_INTEROP_TASK_ID);
    mpi_interop_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<mpi_interop_task>(mpi_interop_registrar,
                                                        "MPI Interop Task");
  }
  {
    TaskVariantRegistrar worker_task_registrar(WORKER_TASK_ID);
    worker_task_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<int, worker_task>(worker_task_registrar,
                                                   "Worker Task");
  }
  // Create a handshake for passing control between Legion and MPI
  // Indicate that MPI has initial control and that there is one
  // participant on each side
  handshake = Runtime::create_handshake(false/*(Was true) MPI initial control*/,
                                        1/*MPI participants*/,
                                        1/*Legion participants*/);
  // Start the Legion runtime in background mode
  Runtime::start(argc, argv, true/*background*/);
  //handshake.mpi_wait_on_legion();
  printf("MPI Doing Work on rank %d\n", rank);
  // When you're done wait for the Legion runtime to shutdown
  Runtime::wait_for_shutdown();
#ifndef GASNET_CONDUIT_MPI
std::cout<<"This should not print if using the mpi conduit, but it seems to anyways. I'm commenting out the MPI_Finalize. If you are using a different conduit, you should uncomment it.\n";
  // Then finalize MPI like normal
  // Exception for the MPI conduit which does its own finalization
//  MPI_Finalize();
#endif

  return 0;
}
