/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2017 by Denis Davydov
 *
 * ---------------------------------------------------------------------

 *
 * matrix-free solution of GHEP in quantum mechanics via pArpack.
 *
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/lac/parpack_solver.h>

#include <iostream>
#include <fstream>

using namespace dealii;
using namespace MatrixFreeOperators;

#include "hamiltonian.h"


struct EigenvalueParameters
{

  EigenvalueParameters()
    :
    global_mesh_refinement_steps(5),
    number_of_eigenvalues(5)
  {}

  unsigned int global_mesh_refinement_steps;

  unsigned int number_of_eigenvalues;

};

template <int dim, int fe_degree=1,int n_q_points=fe_degree+1, typename NumberType = double>
class EigenvalueProblem
{
public:
  ~EigenvalueProblem();
  EigenvalueProblem(const EigenvalueParameters &parameters);
  void run();

private:
  typedef LinearAlgebra::distributed::Vector<NumberType> VectorType;

  void make_mesh();
  void setup_system();
  void solve();

  const EigenvalueParameters &parameters;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  std::ofstream output_fstream;
  ConditionalOStream   pcout;
  ConditionalOStream   plog;
  TimerOutput  computing_timer;

  parallel::distributed::Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;
  FE_Q<dim> fe;
  MappingQ<dim>  mapping;
  QGauss<1> quadrature_formula;

  ConstraintMatrix constraints;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  std::vector<LinearAlgebra::distributed::Vector<NumberType>> eigenfunctions;
  std::vector<NumberType> eigenvalues;

  std::shared_ptr<MatrixFree<dim,NumberType>> fine_level_data;

  HamiltonianOperator<dim,fe_degree,n_q_points,1,VectorType> hamiltonian_operator;
  MassOperator       <dim,fe_degree,n_q_points,1,VectorType> mass_operator;
};



template <int dim, int fe_degree, int n_q_points,typename NumberType>
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::~EigenvalueProblem()
{
  dof_handler.clear ();

  hamiltonian_operator.clear();
  mass_operator.clear();
}



template <int dim, int fe_degree, int n_q_points,typename NumberType>
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::EigenvalueProblem(const EigenvalueParameters &parameters)
  :
  parameters(parameters),
  mpi_communicator(MPI_COMM_WORLD),
  n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout(std::cout, this_mpi_process==0),
  plog(output_fstream, this_mpi_process==0),
  computing_timer(mpi_communicator,
                  pcout,
                  TimerOutput::summary,
                  TimerOutput::wall_times),
  triangulation(mpi_communicator,
                // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                Triangulation<dim>::limit_level_difference_at_vertices,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  dof_handler(triangulation),
  fe(fe_degree),
  mapping(fe_degree+1),
  quadrature_formula(n_q_points),
  eigenfunctions(parameters.number_of_eigenvalues),
  eigenvalues(parameters.number_of_eigenvalues)
{
  if (this_mpi_process==0)
    output_fstream.open("output",std::ios::out | std::ios::trunc);

  const int n_threads = dealii::MultithreadInfo::n_threads();
  pcout << "-------------------------------------------------------------------------" << std::endl
#ifdef DEBUG
        << "--     . running in DEBUG mode" << std::endl
#else
        << "--     . running in OPTIMIZED mode" << std::endl
#endif
        << "--     . running with " << n_mpi_processes << " MPI process" << (n_mpi_processes == 1 ? "" : "es") << std::endl;

  if (n_threads>1)
    pcout << "--     . using " << n_threads << " threads " << (n_mpi_processes == 1 ? "" : "each") << std::endl;

  pcout  << "--     . polynomial degree " << fe_degree << std::endl
         << "--     . quadrature order "  << n_q_points << std::endl
         << "-------------------------------------------------------------------------" << std::endl
         << std::endl;
}



template <int dim, int fe_degree, int n_q_points,typename NumberType>
void
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::make_mesh()
{
  TimerOutput::Scope t (computing_timer, "Make mesh");
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (parameters.global_mesh_refinement_steps);
}



template <int dim, int fe_degree, int n_q_points,typename NumberType>
void
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::setup_system()
{
  TimerOutput::Scope t (computing_timer, "Setup");
  dof_handler.distribute_dofs (fe);

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs (dof_handler,
                                           locally_relevant_dofs);

  constraints.reinit (locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints  (dof_handler, constraints);
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            Functions::ZeroFunction<dim> (),
                                            constraints);
  constraints.close ();

  // matrix-free data
  fine_level_data.reset();
  fine_level_data = std::make_shared<MatrixFree<dim,NumberType>>();
  typename MatrixFree<dim,NumberType>::AdditionalData data;
  data.tasks_parallel_scheme = MatrixFree<dim,NumberType>::AdditionalData::partition_color;
  data.mapping_update_flags = update_values | update_gradients | update_JxW_values;
  fine_level_data->reinit (mapping, dof_handler, constraints, quadrature_formula, data);

  // initialize matrix-free operators:
  mass_operator.initialize(fine_level_data);
  hamiltonian_operator.initialize(fine_level_data);

  // initialize vectors:
  for (unsigned int i=0; i<eigenfunctions.size (); ++i)
    fine_level_data->initialize_dof_vector (eigenfunctions[i]);

  // print out some data
  pcout << "Number of active cells:       "
        << triangulation.n_global_active_cells()
        << std::endl
        << "Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl;
}



template <int dim, int fe_degree, int n_q_points,typename NumberType>
void
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::solve()
{
  TimerOutput::Scope t (computing_timer, "Solve");
  std::vector<std::complex<NumberType>> lambda(parameters.number_of_eigenvalues);

  // set up iterative inverse
  static ReductionControl inner_control_c(dof_handler.n_dofs(), 0.0, 1.e-13);

  SolverCG<VectorType> solver_c(inner_control_c);
  PreconditionIdentity preconditioner;
  const auto shift_and_invert =
    inverse_operator(linear_operator<VectorType>(hamiltonian_operator),
                     solver_c,
                     preconditioner);

  const unsigned int num_arnoldi_vectors = 2*eigenvalues.size() + 2;
  typename PArpackSolver<VectorType>::AdditionalData
  additional_data(num_arnoldi_vectors,
                  PArpackSolver<VectorType>::largest_magnitude,
                  true);

  SolverControl solver_control(dof_handler.n_dofs(), 1e-9, /*log_history*/ false, /*log_results*/ false);
  PArpackSolver<VectorType> eigensolver(solver_control, mpi_communicator, additional_data);

  eigensolver.reinit(eigenfunctions[0]);
  // make sure initial vector is orthogonal to the space due to constraints
  {
    VectorType init_vector;
    fine_level_data->initialize_dof_vector(init_vector);
    init_vector = 1.;
    constraints.set_zero(init_vector);
    eigensolver.set_initial_vector(init_vector);
  }

  // avoid output of iterative solver:
  const unsigned int previous_depth = deallog.depth_file(0);
  eigensolver.solve (hamiltonian_operator,
                     mass_operator,
                     shift_and_invert,
                     lambda,
                     eigenfunctions,
                     eigenvalues.size());
  deallog.depth_file(previous_depth);

  for (unsigned int i = 0; i < lambda.size(); i++)
    eigenvalues[i] = lambda[i].real();

  pcout << "Eigenvalues:                  ";
  for (const auto ev : eigenvalues)
    pcout << ev << " ";
  pcout << std::endl;

  // log for tests
  for (const auto ev: eigenvalues)
    plog  << ev << std::endl;
}



template <int dim, int fe_degree, int n_q_points,typename NumberType>
void
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::run()
{
  make_mesh();
  setup_system();
  solve();
}



int main (int argc,char **argv)
{

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
      {
        EigenvalueParameters parameters;
        EigenvalueProblem<2> eigen_problem(parameters);
        eigen_problem.run();
      }

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };
}
