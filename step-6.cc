/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>

using namespace dealii;


template <int dim>
class Step6
{
public:
  Step6 ();
  ~Step6 ();

  void run ();

private:
  void make_mesh();
  void setup_system ();
  void assemble_system ();
  void solve ();
  void refine_grid ();
  void output_results (const unsigned int cycle) const;

  Triangulation<dim>           triangulation;
  const SphericalManifold<dim> boundary;

  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;

  ConstraintMatrix     constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
};



template <int dim>
double coefficient (const Point<dim> &p)
{
  if (p.square() < 0.5*0.5)
    return 20;
  else
    return 1;
}





template <int dim>
Step6<dim>::Step6 ()
  :
  fe (2),
  dof_handler (triangulation)
{}



template <int dim>
Step6<dim>::~Step6 ()
{
  triangulation.clear ();
}



template <int dim>
void
Step6<dim>::make_mesh ()
{
   GridGenerator::hyper_ball (triangulation);

   triangulation.set_all_manifold_ids_on_boundary(0);
   triangulation.set_manifold (0, boundary);

   triangulation.refine_global (1);
}



template <int dim>
void Step6<dim>::setup_system ()
{
  // Distribute DoFs
  dof_handler.distribute_dofs (fe);

  // Optimise DoF numbering
  DoFRenumbering::Cuthill_McKee (dof_handler);

  // Zero Dirichlet BC and hanging-nodes constraints
  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            Functions::ZeroFunction<dim>(),
                                            constraints);
  constraints.close ();

  // Sparsity of the system matrix:
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);

  // Initialize matrix
  system_matrix.reinit (sparsity_pattern);

  // Initialize vectors
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}



template <int dim>
void Step6<dim>::assemble_system ()
{
  const QGauss<dim>  quadrature_formula(3);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);

      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          const double current_coefficient = coefficient<dim>
                                             (fe_values.quadrature_point (q_index));
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += (current_coefficient *
                                     fe_values.shape_grad(i,q_index) *
                                     fe_values.shape_grad(j,q_index) *
                                     fe_values.JxW(q_index));

              cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                              1.0 *
                              fe_values.JxW(q_index));
            }
        }

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    }
}




template <int dim>
void Step6<dim>::solve ()
{
  SolverControl      solver_control (1000, 1e-12);
  SolverCG<>         solver (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);

  constraints.distribute (solution);
}



template <int dim>
void Step6<dim>::refine_grid ()
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(3),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.3, 0.03);

  triangulation.execute_coarsening_and_refinement ();
}



template <int dim>
void Step6<dim>::output_results (const unsigned int cycle) const
{
  // Output object
  DataOut<dim> data_out;
  // Associate DoFHandler and solution vector
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  // Intermediate format
  data_out.build_patches();
  // Final output
  const std::string filename = "solution_" + Utilities::int_to_string(cycle) + ".vtk";
  std::ofstream output (filename.c_str());
  data_out.write_vtk(output);
}



template <int dim>
void Step6<dim>::run ()
{
  make_mesh();

  for (unsigned int cycle=0; cycle<8; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle > 0)
        refine_grid ();

      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells()
                << std::endl;

      setup_system ();

      std::cout << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

      assemble_system ();
      solve ();
      output_results (cycle);
    }
}



int main ()
{
  Step6<2> laplace_problem_2d;
  laplace_problem_2d.run ();
  return 0;
}
