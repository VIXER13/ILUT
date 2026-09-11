#pragma once

#include "preconditioner_base.hpp"

#include <metamath/linear/fixed_matrix.hpp>
#include <metamath/types/traits.hpp>
#include <metamath/utils/operators.hpp>

#include <vector>

namespace nonlocal::slae {

// Incomplete LDLT factorization preconditioner for symmetric sparse matrices stored in upper triangular format.
// T may be scalar or square_matrix<floating_point, N> for block structure.
// Factorization: A ≈ L D L^T (ILU0 - zero fill-in, sparsity pattern preserved).
// Upper entries store U_{ij} = L_{ji}^T (transpose of lower factor L).
// Diagonal stores D_i (block diagonal factors).
template<class T, std::integral I, std::integral J>
class ildlt_preconditioner final : public preconditioner_base<T> {
    metamath::linear::sparse_matrix<T, I, J> _matrix;
    std::vector<std::vector<size_t>> _col_preds; // TODO: consider using metamath::linear::sparse_matrix_portrait for column predecessors
                                                 // or reuse the existing portrait of the matrix to avoid extra memory allocation.

    // Computes incomplete LDLT factorization of the upper symmetric sparse matrix.
    // Sparsity pattern is preserved (ILU0: zero fill-in).
    void compute() {
        // TODO
    }

public:
    using typename preconditioner_base<T>::entity_t;

    explicit ildlt_preconditioner(metamath::linear::sparse_matrix<T, I, J>&& matrix)
        : _matrix{std::move(matrix)} {
        if (_matrix.rows() != _matrix.cols())
            throw std::invalid_argument{"Incomplete LDLT preconditioner requires a square matrix."};
        compute();
    }

    // Solves (L D L^T) x = rhs via three triangular sweeps.
    std::vector<entity_t> solve(const std::vector<entity_t>& rhs) const override {
        if (rhs.size() != _matrix.cols())
            throw std::invalid_argument{"Incomplete LDLT preconditioner requires rhs vector of the same size as the matrix."};
        return rhs;
    }
};

}
