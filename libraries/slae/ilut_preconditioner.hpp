#pragma once

#include "preconditioner_base.hpp"

#include <metamath/linear/sparse_matrix.hpp>
#include <metamath/linear/fixed_matrix.hpp>
#include <metamath/types/traits.hpp>
#include <metamath/utils/operators.hpp>

#include <vector>

namespace nonlocal::slae {

template<class T, std::integral I, std::integral J>
class ilut_preconditioner final : public preconditioner_base<T> {
    using typename preconditioner_base<T>::entity_t;

    metamath::linear::sparse_matrix<T, I, J> _matrix;

    void compute() {
        const size_t n = _matrix.rows();
        std::vector<entity_t> w(n);
        std::vector<std::vector<entity_t>> lu_rows(n, std::vector<entity_t>(n, 0));
        float tau = 10e-3;
        const size_t p = 10;

        for(const size_t i : std::ranges::iota_view{0zu, n}){
            entity_t norm = 0;
            for(const size_t j : std::ranges::iota_view{0zu, n}){
                w[j] = _matrix(i, j);
                norm += w[j] * w[j];
                
            }
            norm = std::sqrt(norm);
            for(const size_t k : std::ranges::iota_view{0zu, i}){
                w[k] /= lu_rows[k][k];

                for(const size_t j : std::ranges::iota_view{k + 1, n}){
                    w[j] -= w[k] * lu_rows[k][j];
                }
            }
            for(const size_t j : std::ranges::iota_view{0zu, n}){
                if(j != i && std::abs(w[j]) < tau * norm){
                    w[j] = 0;
                }
            }
            std::vector<size_t> indices;
            for (const size_t j : std::ranges::iota_view{0zu, n}){
                if(i != j && w[j] !=0){
                    indices.push_back(j);
                }
            }
            // Сортируем
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                    return std::abs(w[a]) > std::abs(w[b]);
                    }
                );
            for (size_t j = p; j < indices.size(); j++) {
                w[indices[j]] = 0;
            }

            lu_rows[i] = w;

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

        for (const size_t i : std::ranges::iota_view{ 0zu, rhs.size() } ){
            for (const size_t s : _matrix.portrait.shifts_range(i)) {
                if (const size_t j = _matrix.portrait.indices[s]; j < i){
                    y[i] -= _matrix.values[s] * y[j];
                }
            }

        }
        // Ux = y

        for (const size_t i : std::ranges::iota_view{ 0zu, rhs.size() } | std::views::reverse) {
            for (const size_t s : _matrix.portrait.shifts_range(i)) {
            if(const size_t j = _matrix.portrait.indices[s]; j > i){
                y[i] -= _matrix.values[s] * y[j];
                }
            }
            y[i] /= _matrix(i, i);

        }
        return y;
    }

    const metamath::linear::sparse_matrix<T, I, J>& matrix() const noexcept {
        return _matrix;
    };
};

}

