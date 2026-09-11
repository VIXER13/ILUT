#include "utils.hpp"

int main(int argc, char** argv)
try {
    if (argc < 4) {
        logger::info() << "Input format: [program name] /path/to/matrix.txt /path/to/preconditioner_matrix.txt /path/to/right_part.txt" << std::endl;
        return EXIT_FAILURE;
    }

    using T = double;

    logger::info() << "Read matrix" << std::endl;
    const auto matrix = nonlocal::read_matrix<T>(std::filesystem::path{argv[1]});
    logger::info() << "Read preconditioner matrix" << std::endl;
    auto preconditioner_matrix = nonlocal::read_matrix<T>(std::filesystem::path{argv[2]});
    logger::info() << "Read right part" << std::endl;
    const auto right_part = nonlocal::read_vector<T>(std::filesystem::path{argv[3]});

    logger::info() << "Solve system without preconditioner" << std::endl;
    const auto solution1 = nonlocal::solve<T>(matrix, right_part);
    logger::info() << "Solve system with preconditioner" << std::endl;
    const auto solution2 = nonlocal::solve<T>(matrix, right_part, std::move(preconditioner_matrix));

    using namespace metamath::operators;
    static constexpr T Threshold = 1e-10;
    if (const auto error = metamath::linear::norm(solution1 - solution2); error > Threshold)
        logger::error() << "Solutions with a preconditioner do not converge to solutions without a preconditioner."
                        << "The error between solutions: " << error << std::endl;

    return EXIT_SUCCESS;
} catch (const std::exception& e) {
    logger::error() << e.what() << std::endl;
    return EXIT_FAILURE;
} catch (...) {
    logger::error() << "Unknown error." << std::endl;
    return EXIT_FAILURE;
}