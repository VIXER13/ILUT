#pragma once

#include <metamath/linear/sparse_matrix.hpp>
#include <slae/init_solver.hpp>

#include <filesystem>
#include <fstream>

namespace nonlocal {

template<class T>
metamath::linear::sparse_matrix<T> read_matrix(const std::filesystem::path& path) {
    std::ifstream file{path};
    size_t rows = 0;
    file >> rows;
    metamath::linear::sparse_matrix<T> matrix{rows, rows};
    for(const size_t s : std::ranges::iota_view{0zu, rows + 1})
        file >> matrix.portrait.shifts[s];
    matrix.portrait.allocate_indices();
    matrix.allocate_values();
    for(const size_t s : std::ranges::iota_view{0zu, matrix.non_zeros()}) {
        file >> matrix.portrait.indices[s];
        file >> matrix.values[s];
    }
    return matrix;
}

template<class T>
std::vector<T> read_vector(const std::filesystem::path& path) {
    std::ifstream file{path};
    size_t size = 0;
    file >> size;
    std::vector<T> result(size);
    for(const size_t i : std::ranges::iota_view{0zu, size})
        file >> result[i];
    return result;
}

template<class T>
std::vector<T> solve(const metamath::linear::sparse_matrix<T>& matrix, const std::vector<T>& right_part,
    metamath::linear::sparse_matrix<T>&& preconditioner_matrix = {}) {
static constexpr bool Is_Symmetric = false;
const auto solver = nonlocal::slae::init_iterative_solver(matrix, Is_Symmetric);
if (preconditioner_matrix.cols())
solver->preconditioner(nonlocal::slae::init_eigen_preconditioner(std::move(preconditioner_matrix), Is_Symmetric));
return solver->solve(right_part);
}

}