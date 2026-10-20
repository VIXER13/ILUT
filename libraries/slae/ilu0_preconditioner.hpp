#pragma once

#include "preconditioner_base.hpp"

#include <metamath/linear/sparse_matrix.hpp>
#include <metamath/linear/fixed_matrix.hpp>
#include <metamath/types/traits.hpp>
#include <metamath/utils/operators.hpp>

#include <vector>

namespace nonlocal::slae {

// ILU0 (Incomplete LU with zero fill-in) preconditioner for general sparse matrices.
// T may be scalar or square_matrix<floating_point, N> for block structure.
// Factorization: A ≈ L U, sparsity pattern preserved.
// L is stored in the lower triangle (unit diagonal implicit), U in the upper triangle including the diagonal.
template<class T, std::integral I, std::integral J>
class ilu0_preconditioner final : public preconditioner_base<T> {
    using typename preconditioner_base<T>::entity_t;

    metamath::linear::sparse_matrix<T, I, J> _matrix;

    void compute() {
        const size_t n = _matrix.rows();
        for (size_t k = 0; k < n ; k++){
            
            if (!_matrix.portrait.contains(k, k)) {
            throw std::runtime_error{"Missing diagonal element"};
            }

            if (std::abs(_matrix(k, k)) <= 2.2e-16){
                throw std::runtime_error{"Zero diagonal element"};
            }

            //для каждой строки ниже
            for (const size_t s : _matrix.portrait.shifts_range(k)) {
                const size_t i = _matrix.portrait.indices[s]; //получили номера столбцов ненулевых элементов

                if (i > k) {
                    _matrix(i, k) /= _matrix(k, k);

                    for (const size_t t : _matrix.portrait.shifts_range(k)) {
                        const size_t j = _matrix.portrait.indices[t];

                        if (j > k && _matrix.portrait.contains(i, j)) {
                            _matrix(i, j) -= _matrix(i, k) * _matrix.values[t];
                        }
                    }
                }
            }
        }
    }
    
public:
    explicit ilu0_preconditioner(metamath::linear::sparse_matrix<T, I, J>&& matrix)
        : _matrix{std::move(matrix)} {
        if (_matrix.rows() != _matrix.cols())
            throw std::invalid_argument{"ILU0 preconditioner requires a square matrix."};
        validate_sparse_matrix(_matrix);
        compute();
    }

    // Solves (L U) x = rhs via forward and backward substitution.
    std::vector<entity_t> solve(const std::vector<entity_t>& rhs) const override {
        if (rhs.size() != matrix().cols())
            throw std::invalid_argument{"ILU0 preconditioner requires rhs vector of the same size as the matrix."};
        // LUx = b --> Ly=b --> Ux = y

        const size_t n = _matrix.rows();
        std::vector<entity_t> y = rhs;
        //решаем Ly = b 

        for (size_t i = 0; i < n ; i++){
            for (const size_t s : _matrix.portrait.shifts_range(i)) {
                const size_t j = _matrix.portrait.indices[s];
                if (j < i){
                    y[i] -= _matrix.values[s] * y[j];
                }
            }

        }
        // Ux = y
        std::vector<entity_t> x = y;

        for (size_t i = n - 1; i > 0; i--){
            for (const size_t s : _matrix.portrait.shifts_range(i)) {
            const size_t j = _matrix.portrait.indices[s];
            if(j > i){
                x[i] -= _matrix.values[s] * x[j];
                }
            }
            x[i] /= _matrix(i, i);

        }
        return x;
    }

    const metamath::linear::sparse_matrix<T, I, J>& matrix() const noexcept {
        return _matrix;
    };
};

}
