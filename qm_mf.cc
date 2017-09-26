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
#include <deal.II/base/function_parser.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

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

  EigenvalueParameters(const std::string parameter_file)
  {
    ParameterHandler parameter_handler;
    parameter_handler.declare_entry ("Global mesh refinement steps", "5",
                                     Patterns::Integer (0, 20),
                                     "The number of times the 1-cell coarse mesh should "
                                     "be refined globally for our computations.");
    parameter_handler.declare_entry ("Adaptive mesh refinement steps", "0",
                                     Patterns::Integer (0, 20),
                                     "The number of times to perform adaptive mesh refinement.");
    parameter_handler.declare_entry ("Number of eigenvalues/eigenfunctions", "5",
                                     Patterns::Integer (1, 100),
                                     "The number of eigenvalues/eigenfunctions "
                                     "to be computed.");
    parameter_handler.declare_entry ("Potential", "0",
                                     Patterns::Anything(),
                                     "A functional description of the potential.");
    parameter_handler.declare_entry ("Dimension", "2",
                                     Patterns::Selection("2|3"),
                                     "Space dimension of the problem.");
    parameter_handler.declare_entry ("Degree", "1",
                                     Patterns::Integer (1,4),
                                     "Polynomial degree of the FE basis.");
    parameter_handler.declare_entry ("Quadrature points", "2",
                                     Patterns::Integer (1),
                                     "Number of quadrature points in each dimension.");
    parameter_handler.declare_entry ("Size", "2.",
                                     Patterns::Double(),
                                     "Size of the computational domain");
    parameter_handler.declare_entry ("Shift", "0.",
                                     Patterns::Double(),
                                     "Value of shift in shift-and-invert transformation");
    parameter_handler.declare_entry ("Refinement parameter", "0.5",
                                     Patterns::Double(),
                                     "A parameter to mark cells for refinement");


    parameter_handler.parse_input (parameter_file);

    global_mesh_refinement_steps = parameter_handler.get_integer ("Global mesh refinement steps");
    adaptive_mesh_refinement_steps = parameter_handler.get_integer ("Adaptive mesh refinement steps");
    number_of_eigenvalues = parameter_handler.get_integer ("Number of eigenvalues/eigenfunctions");
    potential = parameter_handler.get ("Potential");
    dim = parameter_handler.get_integer ("Dimension");
    degree = parameter_handler.get_integer ("Degree");
    n_q_points = parameter_handler.get_integer ("Quadrature points");
    size = parameter_handler.get_double ("Size");
    shift = parameter_handler.get_double("Shift");
    refinement = parameter_handler.get_double("Refinement parameter");
  }

  unsigned int global_mesh_refinement_steps;

  unsigned int adaptive_mesh_refinement_steps;

  unsigned int number_of_eigenvalues;

  std::string potential;

  unsigned int dim;

  unsigned int degree;

  unsigned int n_q_points;

  double size;

  double shift;

  double refinement;

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
  void solve(const unsigned int cycle);
  void adjust_ghost_range(std::vector<LinearAlgebra::distributed::Vector<NumberType>> &eigenfunctions) const;
  void estimate_error(Vector<float> &error) const;
  void refine();
  void output(const unsigned int iteration) const;

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
  IndexSet locally_relevant_dofs;

  std::vector<LinearAlgebra::distributed::Vector<NumberType>> eigenfunctions;
  std::vector<NumberType> eigenvalues;

  std::shared_ptr<MatrixFree<dim,NumberType>> fine_level_data;

  HamiltonianOperator<dim,fe_degree,n_q_points,1,VectorType> hamiltonian_operator;
  MassOperator       <dim,fe_degree,n_q_points,1,VectorType> mass_operator;

  FunctionParser<dim> potential;

  std::shared_ptr<Table<2, VectorizedArray<NumberType>>> coefficient;

  Vector<float> estimated_error_per_cell;
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
  potential.initialize (FunctionParser<dim>::default_variable_names (),
                        parameters.potential,
                        typename FunctionParser<dim>::ConstMap());

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
  GridGenerator::hyper_cube (triangulation, -parameters.size/2., parameters.size/2.);
  triangulation.refine_global (parameters.global_mesh_refinement_steps);
}



template <int dim, int fe_degree, int n_q_points,typename NumberType>
void
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::setup_system()
{
  TimerOutput::Scope t (computing_timer, "Setup");
  dof_handler.distribute_dofs (fe);

  estimated_error_per_cell.reinit(triangulation.n_active_cells());
  estimated_error_per_cell =0.;

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
  data.mapping_update_flags = update_values | update_gradients | update_JxW_values | update_quadrature_points;
  fine_level_data->reinit (mapping, dof_handler, constraints, quadrature_formula, data);

  // initialize matrix-free operators:
  mass_operator.initialize(fine_level_data);
  hamiltonian_operator.initialize(fine_level_data);

  // initialize vectors:
  for (unsigned int i=0; i<eigenfunctions.size (); ++i)
    fine_level_data->initialize_dof_vector (eigenfunctions[i]);

  // evaluate potential
  coefficient.reset();
  coefficient = std::make_shared<Table<2, VectorizedArray<NumberType>>>();
  {
    FEEvaluation<dim,fe_degree,n_q_points,1,NumberType> fe_eval(*fine_level_data);
    const unsigned int n_cells = fine_level_data->n_macro_cells();
    const unsigned int nqp = fe_eval.n_q_points;
    coefficient->reinit(n_cells, nqp);
    for (unsigned int cell=0; cell<n_cells; ++cell)
      {
        fe_eval.reinit(cell);
        for (unsigned int q=0; q<nqp; ++q)
          {
            VectorizedArray<NumberType> val = make_vectorized_array<NumberType> (0.);
            Point<dim> p;
            for (unsigned int v = 0; v < VectorizedArray<NumberType>::n_array_elements; ++v)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  p[d] = fe_eval.quadrature_point(q)[d][v];
                val[v] = potential.value(p) - parameters.shift;
              }
            (*coefficient)(cell,q) = val;
          }
      }
  }

  hamiltonian_operator.set_coefficient(coefficient);

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
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::solve(const unsigned int cycle)
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
  eigensolver.set_shift(parameters.shift);

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
  if (cycle==0)
    plog << "# cycle cells dofs eigenvalues" << std::endl;
  plog << cycle << " "
       << triangulation.n_global_active_cells() << " "
       << dof_handler.n_dofs();
  for (const auto ev: eigenvalues)
    plog << " " << ev;
  plog << std::endl;
}



template <int dim, int fe_degree, int n_q_points,typename NumberType>
void
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::
adjust_ghost_range(std::vector<LinearAlgebra::distributed::Vector<NumberType>> &eigenfunctions) const
{
  // LA::distributed::Vector initialized by MatrixFree won't have
  // all the ghost values needed for Kelly estimator. So we need to
  // expand the set of ghost dofs here.
  for (unsigned int i = 0; i < eigenfunctions.size(); i++)
    {
      // follow implementation of MatrixFreeOperators::Base::adjust_ghost_range_if_necessary()
      VectorView<double> view_src_in(eigenfunctions[i].local_size(), eigenfunctions[i].begin());
      Vector<double> copy_vec = view_src_in;
      eigenfunctions[i].reinit(dof_handler.locally_owned_dofs(),
                               locally_relevant_dofs,
                               mpi_communicator);
      VectorView<double> view_src_out(eigenfunctions[i].local_size(), eigenfunctions[i].begin());
      static_cast<Vector<double>&>(view_src_out) = copy_vec;
      constraints.distribute (eigenfunctions[i]);
      eigenfunctions[i].update_ghost_values();
    }
}



template <int dim, int fe_degree, int n_q_points,typename NumberType>
void
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::
estimate_error(Vector<float> &error) const
{
  std::vector<Vector<float>> errors_per_cell(eigenfunctions.size());

  std::vector<Vector<float>*> err(eigenfunctions.size());
  std::vector<const LinearAlgebra::distributed::Vector<NumberType> *> sol(eigenfunctions.size());
  for (unsigned int i = 0; i < eigenfunctions.size(); i++)
    {
      errors_per_cell[i].reinit(triangulation.n_active_cells());
      sol[i] = &eigenfunctions[i];
      err[i] = &errors_per_cell[i];
    }

  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(fe_degree+1),
                                      /*neumann_bc:*/typename FunctionMap<dim>::type(),
                                      sol,
                                      err,
                                      ComponentMask(),
                                      0,
                                      numbers::invalid_unsigned_int,
                                      numbers::invalid_subdomain_id,
                                      numbers::invalid_material_id,
                                      KellyErrorEstimator<dim>::cell_diameter);

  // Note, that Kelly return error (not squared error)
  for (unsigned int c = 0; c < triangulation.n_active_cells(); c++)
    {
      error[c] = 0.;
      for (unsigned int i = 0; i < eigenfunctions.size(); i++)
        error[c] += Utilities::fixed_power<2>(errors_per_cell[i][c]);
      error[c] = std::sqrt(estimated_error_per_cell[c]);
    }

  double l2_squared = Utilities::fixed_power<2>(estimated_error_per_cell.l2_norm());
  l2_squared = Utilities::MPI::sum(l2_squared, mpi_communicator);
  l2_squared=std::sqrt(l2_squared);
  pcout << "L2 norm of the error " << l2_squared << std::endl;
}



template <int dim, int fe_degree, int n_q_points,typename NumberType>
void
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::
refine()
{
  const double threshold = parameters.refinement * Utilities::MPI::max(estimated_error_per_cell.linfty_norm(),mpi_communicator);
  GridRefinement::refine (triangulation, estimated_error_per_cell, threshold);

  for (typename Triangulation<dim>::active_cell_iterator
       cell = triangulation.begin_active();
       cell != triangulation.end(); ++cell)
    if (cell->subdomain_id() != triangulation.locally_owned_subdomain())
      {
        cell->clear_refine_flag ();
        cell->clear_coarsen_flag ();
      }

  triangulation.execute_coarsening_and_refinement();
}

template <int dim, int fe_degree, int n_q_points,typename NumberType>
void
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::
output (const unsigned int cycle) const
{
  pcout << "   " << "Output solution..." << std::flush;

  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  for (unsigned int i = 0; i < eigenfunctions.size(); i++)
    data_out.add_data_vector (eigenfunctions[i],
                              std::string("eigenfunction_") +
                              dealii::Utilities::int_to_string(i));

  Vector<float> subdomain (triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");
  data_out.add_data_vector (estimated_error_per_cell, "error_indicator");
  data_out.build_patches ();

  const std::string prefix = "output_" + Utilities::int_to_string(cycle);

  const auto get_filename = [&](const unsigned int p) -> std::string
  {
    return prefix + ".proc" + Utilities::int_to_string(p) + ".vtu";
  };

  const std::string filename = get_filename(triangulation.locally_owned_subdomain());

  std::ofstream output (filename.c_str());
  data_out.write_vtu (output);

  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      // first write out "pvtu" file which combines output from all MPI cores
      std::vector<std::string> filenames;
      for (unsigned int i=0; i<dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
        filenames.push_back (get_filename(i));

      const std::string pvtu_filename = prefix + ".pvtu";
      std::ofstream pvtu_output (pvtu_filename.c_str());
      data_out.write_pvtu_record (pvtu_output, filenames);

      // and then write a "pvd" file which combines the output of different cycles
      static std::vector<std::pair<double,std::string> > time_and_name_history;
      time_and_name_history.push_back (std::make_pair (cycle,pvtu_filename));

      const std::string pvd_filename = "output.pvd";
      std::ofstream pvd_output (pvd_filename.c_str());
      DataOutBase::write_pvd_record (pvd_output, time_and_name_history);
    }
}



template <int dim, int fe_degree, int n_q_points,typename NumberType>
void
EigenvalueProblem<dim,fe_degree,n_q_points,NumberType>::run()
{
  make_mesh();

  for (unsigned int cycle = 0; cycle <= parameters.adaptive_mesh_refinement_steps; ++cycle)
    {
      setup_system();
      solve(cycle);
      adjust_ghost_range(eigenfunctions);
      output(cycle);
      estimate_error(estimated_error_per_cell);
      refine();
    }
}


// macros for Boost to choose polynomial degree and quadrature points
// at run-time:
# include <boost/preprocessor/facilities/empty.hpp>
# include <boost/preprocessor/list/at.hpp>
# include <boost/preprocessor/list/for_each_product.hpp>
# include <boost/preprocessor/tuple/elem.hpp>
# include <boost/preprocessor/tuple/to_list.hpp>

#define MF_DIM BOOST_PP_TUPLE_TO_LIST(2,(2,3))
#define MF_DQ  BOOST_PP_TUPLE_TO_LIST(10,(\
                                          (1,2),\
                                          (2,3),(2,4),(2,5),(2,6),\
                                          (3,4),(3,5),(3,6),\
                                          (4,5),(4,6) \
                                         ))

// Accessors:
#define GET_D(L)   BOOST_PP_TUPLE_ELEM(2,0,BOOST_PP_TUPLE_ELEM(1,0,L))
#define GET_Q(L)   BOOST_PP_TUPLE_ELEM(2,1,BOOST_PP_TUPLE_ELEM(1,0,L))

#define DOIF3(R, L) \
  else if ( (parameters.degree == GET_D(L)) && (parameters.n_q_points == GET_Q(L) ) ) \
    { \
      EigenvalueProblem<3,GET_D(L),GET_Q(L)> eigen_problem(parameters); \
      eigen_problem.run (); \
    } \
   
#define DOIF2(R, L) \
  else if ( (parameters.degree == GET_D(L)) && (parameters.n_q_points == GET_Q(L) ) ) \
    { \
      EigenvalueProblem<2,GET_D(L),GET_Q(L)> eigen_problem(parameters); \
      eigen_problem.run (); \
    } \
   
int main (int argc, char *argv[])
{

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
      {
        AssertThrow(argc > 1,
                    ExcMessage("Parameter file is required as an input argument"));
        const std::string filename = argv[1];
        EigenvalueParameters parameters(filename);
        if (parameters.dim == 2)
          {
            // start with a dummy condition to use Boost PP 'else if' below.
            if (parameters.degree==0)
              {
                AssertThrow(false, ExcInternalError());
              }
            BOOST_PP_LIST_FOR_EACH_PRODUCT(DOIF2, 1, (MF_DQ))
            else
              {
                AssertThrow(false,
                            ExcMessage("Matrix-free calculations with degree="+
                                       std::to_string(parameters.degree)+
                                       " and n_q_points_1d="+
                                       std::to_string(parameters.n_q_points)+
                                       " are not supported."));
              }
          }
        else
          {
            // start with a dummy condition to use Boost PP 'else if' below.
            if (parameters.degree==0)
              {
                AssertThrow(false, ExcInternalError());
              }
            BOOST_PP_LIST_FOR_EACH_PRODUCT(DOIF3, 1, (MF_DQ))
            else
              {
                AssertThrow(false,
                            ExcMessage("Matrix-free calculations with degree="+
                                       std::to_string(parameters.degree)+
                                       " and n_q_points_1d="+
                                       std::to_string(parameters.n_q_points)+
                                       " are not supported."));
              }
          }
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
