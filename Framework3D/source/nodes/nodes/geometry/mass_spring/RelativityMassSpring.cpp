#include "RelativityMassSpring.h"

#include <omp.h>

#include <cmath>
#include <iostream>

// per OpenMP standard, _OPENMP defined when compiling with OpenMP
#ifdef _OPENMP
#ifdef _MSC_VER
// must use MSVC __pragma here instead of _Pragma otherwise you get an internal
// compiler error. still an issue in Visual Studio 2022
#define OMP_PARALLEL_FOR __pragma(omp parallel for)
// any other standards-compliant C99/C++11 compiler
#else
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#endif  // _MSC_VER
// no OpenMP support
#else
#define OMP_PARALLEL_FOR
#endif  // _OPENMP

namespace USTC_CG::node_relativity_mass_spring {
MassSpring::MassSpring(const Eigen::MatrixXd& X, const EdgeSet& E)
{
    this->X = this->init_X = X;
    this->vel = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    this->E = E;

    std::cout << "number of edges: " << E.size() << std::endl;
    std::cout << "init mass spring" << std::endl;

    // Compute the rest pose edge length
    for (const auto& e : E) {
        Eigen::Vector3d x0 = X.row(e.first);
        Eigen::Vector3d x1 = X.row(e.second);
        this->E_rest_length.push_back((x0 - x1).norm());
    }

    // Initialize the mask for Dirichlet boundary condition
    dirichlet_bc_mask.resize(X.rows(), false);

    unsigned n_fix = sqrt(X.rows());  // Here we assume the cloth is square
    dirichlet_bc_mask[0] = true;
    dirichlet_bc_mask[n_fix - 1] = true;
}

void MassSpring::step()
{
    Eigen::Vector3d acceleration_ext = gravity + wind_ext_acc;

    unsigned n_vertices = X.rows();

    // The reason to not use 1.0 as mass per vertex: the cloth gets heavier as we increase the
    // resolution
    double mass_per_vertex = mass / n_vertices;

    Eigen::MatrixXd acceleration_collision =
        getSphereCollisionForce(sphere_center.cast<double>(), sphere_radius);
    
    if (time_integrator == IMPLICIT_EULER) {
        // Implicit Euler
        TIC(step)

        auto H = computeHessianSparse(stiffness);  // size = [nx3, nx3]
        // toSPD(H);

        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        // Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed!" << std::endl;
            return;
        }

        // compute Y
        Eigen::MatrixXd Y = X + h * vel;
        OMP_PARALLEL_FOR
        for (int i = 0; i < n_vertices; i++) {
            auto beta = vel.row(i) / speed_of_light;
            auto gamma = 1.0 / std::sqrt(1.0 - beta.squaredNorm());
            if (!dirichlet_bc_mask[i]) {
                Y.row(i) += h * h *
                            (acceleration_ext.transpose() - acceleration_ext.dot(beta) * beta) /
                            gamma;
                if (enable_sphere_collision) {
                    Y.row(i) += h * h *
                                (acceleration_collision.row(i) -
                                 acceleration_collision.row(i).dot(beta) * beta) /
                                gamma;
                }
            }
        }

        auto grad_g = computeGrad(stiffness);
        OMP_PARALLEL_FOR
        for (int i = 0; i < n_vertices; i++) {
            if (!dirichlet_bc_mask[i]) {
                auto beta = vel.row(i) / speed_of_light;
                auto gamma = 1.0 / std::sqrt(1.0 - beta.squaredNorm());
                grad_g.row(i) = gamma * mass_per_vertex * (X.row(i) - Y.row(i)) / h / h +
                                grad_g.row(i) - grad_g.row(i).dot(beta) * beta;
            }
        }
        auto grad_g_flatten = flatten(grad_g);

        // Solve Newton's search direction with linear solver
        // Delta x = H^(-1) * grad, H * Delta x = grad
        auto delta_X_flatten = solver.solve(grad_g_flatten);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solving failed!" << std::endl;
            return;
        }
        auto delta_X = unflatten(delta_X_flatten);

        if (enable_energy_correction) {
            // Energy correction
            double old_energy_p = computeEnergy(stiffness);
            double old_energy_k = 0;
            for (int i = 0; i < n_vertices; i++) {
                auto beta = vel.row(i) / speed_of_light;
                old_energy_k += mass_per_vertex * speed_of_light * speed_of_light /
                                std::sqrt(1 - beta.squaredNorm());
                old_energy_p -= mass_per_vertex * (acceleration_ext.dot(X.row(i)) +
                                                   acceleration_collision.row(i).dot(X.row(i)));
            }
            if (enable_debug_output)
                printf(
                    "old energy p: %6lf, old energy k: %6lf, old energy: %6lf\n",
                    old_energy_p,
                    old_energy_k,
                    old_energy_p + old_energy_k);
            OMP_PARALLEL_FOR
            for (int i = 0; i < n_vertices; i++) {
                X.row(i) -= delta_X.row(i);
            }

            double new_energy_p = computeEnergy(stiffness);
            double new_energy_k = 0;
            for (int i = 0; i < n_vertices; i++) {
                double beta = delta_X.row(i).norm() / h / speed_of_light;
                if (beta > 1 - 1e-3) {
                    beta = 1 - 1e-3;
                }
                new_energy_k +=
                    mass_per_vertex * speed_of_light * speed_of_light / std::sqrt(1 - beta * beta);
                new_energy_p -= mass_per_vertex * (acceleration_ext.dot(X.row(i)) +
                                                   acceleration_collision.row(i).dot(X.row(i)));
            }
            if (enable_debug_output)
                printf(
                    "new energy p: %6lf, new energy k: %6lf, new energy: %6lf\n",
                    new_energy_p,
                    new_energy_k,
                    new_energy_p + new_energy_k);
            // Potential energy transfer to kinetic energy
            const double aim_delta_energy_k = old_energy_p - new_energy_p;

            if (enable_energy_correction == 1) {
                // Method 1: Correct a along tangent direction
                double delta_energy_k = 0;
                for (int i = 0; i < n_vertices; i++) {
                    auto beta = vel.row(i) / speed_of_light;
                    delta_energy_k += beta.dot(delta_X.row(i) / h - vel.row(i)) * mass_per_vertex *
                                      speed_of_light / std::pow(1 - beta.squaredNorm(), 1.5);
                }
                if (enable_debug_output)
                    printf("delta energy k: %lf\n", delta_energy_k);
                double multiplier;
                if (delta_energy_k == 0) {
                    // For those who doesn't have a rising tangent direction. Not work in secant
                    // method.
                    multiplier = 1;
                }
                else if (delta_energy_k > 0) {
                    // Make energy less
                    if (aim_delta_energy_k > delta_energy_k) {
                        // Not exceed the aim
                        multiplier = 1;
                    }
                    else if (aim_delta_energy_k < 0) {
                        // Correction should not change the direction
                        multiplier = 0;
                    }
                    else {
                        multiplier = aim_delta_energy_k / delta_energy_k;
                    }
                }
                else if (delta_energy_k < 0) {
                    // Already make energy less
                    // If aim is less than the current energy, we do not set it to fit aim, because
                    // this way there will be huge a, v and x, which means explosion. To avoid fast
                    // change, just leave it the same.
                    multiplier = 1;
                }
                if (enable_debug_output)
                    printf("aim: %lf, multiplier: %lf\n", aim_delta_energy_k, multiplier);
                OMP_PARALLEL_FOR
                for (int i = 0; i < n_vertices; i++) {
                    X.row(i) += delta_X.row(i);
                    delta_X.row(i) *= multiplier;
                }
            }
            else if (enable_energy_correction == 2) {
                // Method 2: Correct a along secant direction
                double delta_energy_k = new_energy_k - old_energy_k;
                if (enable_debug_output)
                    printf("delta energy k: %lf\n", delta_energy_k);
                double multiplier;
                if (delta_energy_k == 0) {
                    // For those who doesn't have a rising tangent direction. Not work in secant
                    // method.
                    multiplier = 1;
                }
                else if (delta_energy_k > 0) {
                    // Make energy less
                    if (aim_delta_energy_k > delta_energy_k) {
                        // Not exceed the aim
                        multiplier = 1;
                    }
                    else if (aim_delta_energy_k < 0) {
                        // Correction should not change the direction
                        multiplier = 0;
                    }
                    else {
                        multiplier = aim_delta_energy_k / delta_energy_k;
                    }
                }
                else if (delta_energy_k < 0) {
                    // Already make energy less
                    // If aim is less than the current energy, we do not set it to fit aim, because
                    // this way there will be huge a, v and x, which means explosion. To avoid fast
                    // change, just leave it the same.
                    multiplier = 1;
                }
                if (enable_debug_output)
                    printf("aim: %lf, multiplier: %lf\n", aim_delta_energy_k, multiplier);
                OMP_PARALLEL_FOR
                for (int i = 0; i < n_vertices; i++) {
                    X.row(i) += delta_X.row(i);
                    delta_X.row(i) *= multiplier;
                }
            }
        }

        // update X and vel
        OMP_PARALLEL_FOR
        for (int i = 0; i < n_vertices; i++) {
            if (!dirichlet_bc_mask[i]) {
                vel.row(i) = -delta_X.row(i) / h;
                if (vel.row(i).norm() > speed_of_light) {
                    vel.row(i) = (1 - 1e-3) * speed_of_light * vel.row(i).normalized();
                }
                X.row(i) += vel.row(i) * h;
            }
            else {
                vel.row(i).setZero();
            }
        }

        TOC(step)
    }
    else if (time_integrator == SEMI_IMPLICIT_EULER) {
        // Semi-implicit Euler
        Eigen::MatrixXd force = -computeGrad(stiffness) / mass_per_vertex;
        OMP_PARALLEL_FOR
        for (int i = 0; i < n_vertices; i++) {
            if (!dirichlet_bc_mask[i]) {
                force.row(i) += acceleration_ext.transpose();
            }
        }
        if (enable_sphere_collision) {
            force += acceleration_collision;
        }
        Eigen::MatrixXd acceleration(n_vertices, 3);
        OMP_PARALLEL_FOR
        for (int i = 0; i < n_vertices; i++) {
            if (!dirichlet_bc_mask[i]) {
                auto beta = vel.row(i) / speed_of_light;
                double gamma = 1.0 / std::sqrt(1.0 - beta.squaredNorm());
                acceleration.row(i) = (force.row(i) - force.row(i).dot(beta) * beta) / gamma;
            }
            else {
                acceleration.row(i).setZero();
            }
        }

        if (enable_energy_correction) {
            // Energy correction
            double old_energy_p = computeEnergy(stiffness);
            double old_energy_k = 0;
            for (int i = 0; i < n_vertices; i++) {
                auto beta = vel.row(i) / speed_of_light;
                old_energy_k += mass_per_vertex * speed_of_light * speed_of_light /
                                std::sqrt(1 - beta.squaredNorm());
                old_energy_p -= mass_per_vertex * (acceleration_ext.dot(X.row(i)) +
                                                   acceleration_collision.row(i).dot(X.row(i)));
            }
            if (enable_debug_output)
                printf(
                    "old energy p: %6lf, old energy k: %6lf, old energy: %6lf\n",
                    old_energy_p,
                    old_energy_k,
                    old_energy_p + old_energy_k);

            OMP_PARALLEL_FOR
            for (int i = 0; i < n_vertices; i++) {
                X.row(i) += h * vel.row(i) + h * h * std::pow(damping, h) * acceleration.row(i);
            }

            double new_energy_p = computeEnergy(stiffness);
            double new_energy_k = 0;
            for (int i = 0; i < n_vertices; i++) {
                double beta = (vel.row(i) + h * std::pow(damping, h) * acceleration.row(i)).norm() /
                              speed_of_light;
                if (beta > 1 - 1e-3) {
                    beta = 1 - 1e-3;
                }
                new_energy_k +=
                    mass_per_vertex * speed_of_light * speed_of_light / std::sqrt(1 - beta * beta);
                new_energy_p -= mass_per_vertex * (acceleration_ext.dot(X.row(i)) +
                                                   acceleration_collision.row(i).dot(X.row(i)));
            }
            if (enable_debug_output)
                printf(
                    "new energy p: %6lf, new energy k: %6lf, new energy: %6lf\n",
                    new_energy_p,
                    new_energy_k,
                    new_energy_p + new_energy_k);
            // Potential energy transfer to kinetic energy
            const double aim_delta_energy_k = old_energy_p - new_energy_p;

            if (enable_energy_correction == 1) {
                // Method 1: Correct a along tangent direction
                double delta_energy_k = 0;
                for (int i = 0; i < n_vertices; i++) {
                    auto beta = vel.row(i) / speed_of_light;
                    delta_energy_k += h * beta.dot(acceleration.row(i)) * mass_per_vertex *
                                      speed_of_light / std::pow(1 - beta.squaredNorm(), 1.5);
                }
                if (enable_debug_output)
                    printf("delta energy k: %lf\n", delta_energy_k);
                double multiplier;
                if (delta_energy_k == 0) {
                    // For those who doesn't have a rising tangent direction. Not work in secant
                    // method.
                    multiplier = 1;
                }
                else if (delta_energy_k > 0) {
                    // Make energy less
                    if (aim_delta_energy_k > delta_energy_k) {
                        // Not exceed the aim
                        multiplier = 1;
                    }
                    else if (aim_delta_energy_k < 0) {
                        // Correction should not change the direction
                        multiplier = 0;
                    }
                    else {
                        multiplier = aim_delta_energy_k / delta_energy_k;
                    }
                }
                else if (delta_energy_k < 0) {
                    // Already make energy less
                    // If aim is less than the current energy, we do not set it to fit aim, because
                    // this way there will be huge a, v and x, which means explosion. To avoid fast
                    // change, just leave it the same.
                    multiplier = 1;
                }
                if (enable_debug_output)
                    printf("aim: %lf, multiplier: %lf\n", aim_delta_energy_k, multiplier);
                OMP_PARALLEL_FOR
                for (int i = 0; i < n_vertices; i++) {
                    X.row(i) -= h * vel.row(i) + h * h * std::pow(damping, h) * acceleration.row(i);
                    acceleration.row(i) *= multiplier;
                }
            }
            else if (enable_energy_correction == 2) {
                // Method 2: Correct a along secant direction
                double delta_energy_k = new_energy_k - old_energy_k;
                if (enable_debug_output)
                    printf("delta energy k: %lf\n", delta_energy_k);
                double multiplier;
                if (delta_energy_k == 0) {
                    // For those who doesn't have a rising tangent direction. Not work in secant
                    // method.
                    multiplier = 1;
                }
                else if (delta_energy_k > 0) {
                    // Make energy less
                    if (aim_delta_energy_k > delta_energy_k) {
                        // Not exceed the aim
                        multiplier = 1;
                    }
                    else if (aim_delta_energy_k < 0) {
                        // Correction should not change the direction
                        multiplier = 0;
                    }
                    else {
                        multiplier = aim_delta_energy_k / delta_energy_k;
                    }
                }
                else if (delta_energy_k < 0) {
                    // Already make energy less
                    // If aim is less than the current energy, we do not set it to fit aim, because
                    // this way there will be huge a, v and x, which means explosion. To avoid fast
                    // change, just leave it the same.
                    multiplier = 1;
                }
                if (enable_debug_output)
                    printf("aim: %lf, multiplier: %lf\n", aim_delta_energy_k, multiplier);
                OMP_PARALLEL_FOR
                for (int i = 0; i < n_vertices; i++) {
                    X.row(i) -= h * vel.row(i) + h * h * std::pow(damping, h) * acceleration.row(i);
                    acceleration.row(i) *= multiplier;
                }
            }
        }

        // Update X and vel
        OMP_PARALLEL_FOR
        for (int i = 0; i < n_vertices; i++) {
            vel.row(i) += h * std::pow(damping, h) * acceleration.row(i);
            if (vel.row(i).norm() > speed_of_light) {
                vel.row(i) = (1 - 1e-3) * speed_of_light * vel.row(i).normalized();
            }
            X.row(i) += h * vel.row(i);
        }
    }
    else {
        std::cerr << "Unknown time integrator!" << std::endl;
        return;
    }
}

// There are different types of mass spring energy:
// For this homework we will adopt Prof. Huamin Wang's energy definition introduced in GAMES103
// course Lecture 2 E = 0.5 * stiffness * sum_{i=1}^{n} (||x_i - x_j|| - l)^2 There exist other
// types of energy definition, e.g., Prof. Minchen Li's energy definition
// https://www.cs.cmu.edu/~15769-f23/lec/3_Mass_Spring_Systems.pdf
double MassSpring::computeEnergy(double stiffness)
{
    double sum = 0.;
    unsigned i = 0;
    for (const auto& e : E) {
        // For each edge
        auto diff = X.row(e.first) - X.row(e.second);
        // X (n x 3) stores the vertex positions
        // E (m x 2) stores the edge indices
        auto l = E_rest_length[i];
        // E_rest_length stores the rest length of each edge, num. i
        sum += 0.5 * stiffness * std::pow((diff.norm() - l), 2);
        i++;
    }
    return sum;
}

Eigen::MatrixXd MassSpring::computeGrad(double stiffness)
{
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    // g (n x 3) stores the gradient of the energy. The gradient is about the vertex i.
    unsigned i = 0;
    for (const auto& e : E) {
        const Eigen::Vector3d x = X.row(e.first) - X.row(e.second);
        g.row(e.first) += stiffness * (x.norm() - E_rest_length[i]) * x.normalized();
        g.row(e.second) += -stiffness * (x.norm() - E_rest_length[i]) * x.normalized();
        i++;
    }
    OMP_PARALLEL_FOR
    for (int j = 0; j < X.rows(); j++) {
        if (dirichlet_bc_mask[j]) {
            g.row(j).setZero();
        }
    }
    return g;
}

Eigen::SparseMatrix<double> MassSpring::computeHessianSparse(double stiffness)
{
    // Now compute gamma / h^2 * m + (I - beta beta^T) H
    unsigned n_vertices = X.rows();
    Eigen::SparseMatrix<double> H(n_vertices * 3, n_vertices * 3);

    unsigned i = 0;
    const auto I = Eigen::MatrixXd::Identity(3, 3);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.clear();
    for (const auto& e : E) {
        const Eigen::Vector3d x = X.row(e.first) - X.row(e.second);
        auto beta1 = vel.row(e.first) / speed_of_light;
        auto beta2 = vel.row(e.second) / speed_of_light;
        const Eigen::MatrixXd He = stiffness * (x * x.transpose() / x.squaredNorm() +
                                                (1 - E_rest_length[i] / x.norm()) *
                                                    (I - x * x.transpose() / x.squaredNorm()));
        Eigen::MatrixXd He1 = He, He2 = He;
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                for (int r = 0; r < 3; r++) {
                    He1(p, q) -= beta1(p) * beta1(r) * He(r, q);
                    He2(p, q) -= beta2(p) * beta2(r) * He(r, q);
                }
            }
        }
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                if (!dirichlet_bc_mask[e.first]) {
                    triplets.push_back(
                        Eigen::Triplet<double>(e.first * 3 + p, e.first * 3 + q, He1(p, q)));
                    if (!dirichlet_bc_mask[e.second]) {
                        triplets.push_back(
                            Eigen::Triplet<double>(e.first * 3 + p, e.second * 3 + q, -He2(p, q)));
                        triplets.push_back(
                            Eigen::Triplet<double>(e.second * 3 + p, e.first * 3 + q, -He1(p, q)));
                        triplets.push_back(
                            Eigen::Triplet<double>(e.second * 3 + p, e.second * 3 + q, He2(p, q)));
                    }
                }
                else if (!dirichlet_bc_mask[e.second]) {
                    triplets.push_back(
                        Eigen::Triplet<double>(e.second * 3 + p, e.second * 3 + q, He2(p, q)));
                }
            }
        }
        i++;
    }
    for (int j = 0; j < n_vertices; j++) {
        double gamma =
            1.0 / std::sqrt(1.0 - vel.row(j).squaredNorm() / speed_of_light / speed_of_light);
        for (int p = 0; p < 3; p++) {
            triplets.push_back(
                Eigen::Triplet<double>(j * 3 + p, j * 3 + p, gamma * mass / n_vertices / h / h));
        }
    }
    H.setFromTriplets(triplets.begin(), triplets.end());
    H.makeCompressed();
    return H;
}

bool MassSpring::checkSPD(const Eigen::SparseMatrix<double>& A)
{
    // Eigen::SimplicialLDLT<SparseMatrix_d> ldlt(A);
    // return ldlt.info() == Eigen::Success;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    auto eigen_values = es.eigenvalues();
    return eigen_values.minCoeff() >= 1e-10;
}

void MassSpring::toSPD(Eigen::SparseMatrix<double>& A)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    auto eigen_values = es.eigenvalues();
    printf("Minimal Eigen Value: %lf\n", eigen_values.minCoeff());
    if (eigen_values.minCoeff() < 0) {
        printf("Eigenvalue < 0, make SPD\n");
        Eigen::SparseMatrix<double> B(A.rows(), A.cols());
        B.setIdentity();
        A += B * (1e-6 - eigen_values.minCoeff());
    }
    return;
}

void MassSpring::reset()
{
    std::cout << "reset" << std::endl;
    this->X = this->init_X;
    this->vel.setZero();
}

Eigen::MatrixXd MassSpring::getSphereCollisionForce(Eigen::Vector3d center, double radius)
{
    Eigen::MatrixXd force = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    OMP_PARALLEL_FOR
    for (int i = 0; i < X.rows(); i++) {
        auto delta_x = X.row(i) - center.transpose();
        force.row(i) += collision_penalty_k *
                        std::max(0.0, collision_scale_factor * radius - delta_x.norm()) *
                        delta_x.normalized();
    }
    return force;
}

}  // namespace USTC_CG::node_relativity_mass_spring
