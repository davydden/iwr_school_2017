/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2017 by Denis Davydov
 *
 * ---------------------------------------------------------------------
 */

template <int dim, int fe_degree, int n_q_points_1d = fe_degree+1, int n_components = 1, typename VectorType = LinearAlgebra::distributed::Vector<double> >
class HamiltonianOperator : public Base<dim, VectorType>
{
public:
  /**
   * Number typedef.
   */
  typedef typename Base<dim,VectorType>::value_type value_type;

  /**
   * size_type needed for preconditioner classes.
   */
  typedef typename Base<dim,VectorType>::size_type size_type;

  /**
   * Constructor.
   */
  HamiltonianOperator ();

  /**
   * The diagonal is approximated by computing a local diagonal matrix per element
   * and distributing it to the global diagonal. This will lead to wrong results
   * on element with hanging nodes but is still an acceptable approximation
   * to be used in preconditioners.
   */
  virtual void compute_diagonal ();

  /**
   * Set the heterogeneous scalar coefficient @p scalar_coefficient to be used at
   * the quadrature points. The Table should be of correct size, consistent
   * with the total number of quadrature points in <code>dim</code>-dimensions,
   * controlled by the @p n_q_points_1d template parameter. Here,
   * <code>(*scalar_coefficient)(cell,q)</code> corresponds to the value of the
   * coefficient, where <code>cell</code> is an index into a set of cell
   * batches as administered by the MatrixFree framework (which does not work
   * on individual cells, but instead of batches of cells at once), and
   * <code>q</code> is the number of the quadrature point within this batch.
   *
   * If this function is not called, the coefficient is assumed to be zero.
   */
  void set_coefficient(const std::shared_ptr<Table<2, VectorizedArray<value_type> > > &scalar_coefficient );

  virtual void clear();

  /**
   * Read/Write access to coefficients to be used in Hamiltonian operator.
   *
   * The function will throw an error if coefficients are not previously set
   * by set_coefficient() function.
   */
  std::shared_ptr< Table<2, VectorizedArray<value_type> > > get_coefficient();

private:
  /**
   * Applies the Hamiltonian matrix operation on an input vector. It is
   * assumed that the passed input and output vector are correctly initialized
   * using initialize_dof_vector().
   */
  virtual void apply_add (VectorType       &dst,
                          const VectorType &src) const;

  /**
   * Applies the Hamiltonian operator on a cell.
   */
  void local_apply_cell (const MatrixFree<dim,value_type>            &data,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int,unsigned int>  &cell_range) const;

  /**
   * Apply diagonal part of the Hamiltonian operator on a cell.
   */
  void local_diagonal_cell (const MatrixFree<dim,value_type>            &data,
                            VectorType                                  &dst,
                            const unsigned int &,
                            const std::pair<unsigned int,unsigned int>  &cell_range) const;

  /**
   * Apply Hamiltonian operator on a cell @p cell.
   */
  void do_operation_on_cell(FEEvaluation<dim,fe_degree,n_q_points_1d,n_components,value_type> &phi,
                            const unsigned int cell) const;

  /**
   * User-provided coefficient.
   */
  std::shared_ptr< Table<2, VectorizedArray<value_type> > > scalar_coefficient;
};


template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
HamiltonianOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
HamiltonianOperator ()
  :
  Base<dim, VectorType>()
{}



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
void
HamiltonianOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
clear ()
{
  Base<dim, VectorType>::clear();
  scalar_coefficient.reset();
}



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
void
HamiltonianOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
set_coefficient(const std::shared_ptr<Table<2, VectorizedArray<typename Base<dim,VectorType>::value_type> > > &scalar_coefficient_ )
{
  scalar_coefficient = scalar_coefficient_;
}



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
std::shared_ptr< Table<2, VectorizedArray< typename HamiltonianOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::value_type> > >
HamiltonianOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
get_coefficient()
{
  Assert (scalar_coefficient.get(),
          ExcNotInitialized());
  return scalar_coefficient;
}



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
void
HamiltonianOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
compute_diagonal()
{
  typedef typename Base<dim,VectorType>::value_type Number;
  Assert((Base<dim, VectorType>::data.get() != nullptr), ExcNotInitialized());

  unsigned int dummy = 0;
  this->inverse_diagonal_entries.
  reset(new DiagonalMatrix<VectorType>());
  VectorType &inverse_diagonal_vector = this->inverse_diagonal_entries->get_vector();
  this->initialize_dof_vector(inverse_diagonal_vector);

  this->data->cell_loop (&HamiltonianOperator::local_diagonal_cell,
                         this, inverse_diagonal_vector, dummy);
  this->set_constrained_entries_to_one(inverse_diagonal_vector);

  for (unsigned int i=0; i<inverse_diagonal_vector.local_size(); ++i)
    if (std::abs(inverse_diagonal_vector.local_element(i)) > std::sqrt(std::numeric_limits<Number>::epsilon()))
      inverse_diagonal_vector.local_element(i) = 1./inverse_diagonal_vector.local_element(i);
    else
      inverse_diagonal_vector.local_element(i) = 1.;

  inverse_diagonal_vector.update_ghost_values();
}



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
void
HamiltonianOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
apply_add (VectorType       &dst,
           const VectorType &src) const
{
  Base<dim, VectorType>::data->cell_loop (&HamiltonianOperator::local_apply_cell,
                                          this, dst, src);
}



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
void
HamiltonianOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
do_operation_on_cell(FEEvaluation<dim,fe_degree,n_q_points_1d,n_components,typename Base<dim,VectorType>::value_type> &phi,
                     const unsigned int cell) const
{
  typedef typename Base<dim,VectorType>::value_type number;
  // FIXME: switch to 1/2
  VectorizedArray<number> half = make_vectorized_array<number>(0.5);
  Assert (scalar_coefficient.get(),
          ExcMessage("Coefficients are not initialized"));

  phi.evaluate (true,true,false);
  for (unsigned int q=0; q<phi.n_q_points; ++q)
    {
      phi.submit_value ((*scalar_coefficient)(cell,q)*phi.get_value(q), q);
      phi.submit_gradient (half*phi.get_gradient(q), q);
    }
  phi.integrate (true,true);
}




template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
void
HamiltonianOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
local_apply_cell (const MatrixFree<dim,typename Base<dim,VectorType>::value_type> &data,
                  VectorType       &dst,
                  const VectorType &src,
                  const std::pair<unsigned int,unsigned int>  &cell_range) const
{
  typedef typename Base<dim,VectorType>::value_type Number;
  FEEvaluation<dim,fe_degree,n_q_points_1d,n_components,Number> phi (data, this->selected_rows[0]);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit (cell);
      phi.read_dof_values(src);
      do_operation_on_cell(phi,cell);
      phi.distribute_local_to_global (dst);
    }
}


template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename VectorType>
void
HamiltonianOperator<dim, fe_degree, n_q_points_1d, n_components, VectorType>::
local_diagonal_cell (const MatrixFree<dim,typename Base<dim,VectorType>::value_type> &data,
                     VectorType                                       &dst,
                     const unsigned int &,
                     const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  typedef typename Base<dim,VectorType>::value_type Number;
  FEEvaluation<dim,fe_degree,n_q_points_1d,n_components,Number> phi (data, this->selected_rows[0]);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit (cell);
      VectorizedArray<Number> local_diagonal_vector[phi.tensor_dofs_per_cell];
      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
            phi.begin_dof_values()[j] = VectorizedArray<Number>();
          phi.begin_dof_values()[i] = 1.;
          do_operation_on_cell(phi,cell);
          local_diagonal_vector[i] = phi.begin_dof_values()[i];
        }
      for (unsigned int i=0; i<phi.tensor_dofs_per_cell; ++i)
        phi.begin_dof_values()[i] = local_diagonal_vector[i];
      phi.distribute_local_to_global (dst);
    }
}
