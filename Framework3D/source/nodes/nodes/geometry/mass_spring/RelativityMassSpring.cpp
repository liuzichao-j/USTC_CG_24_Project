#include "RelativityMassSpring.h"

#include <cmath>
#include <iostream>

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

    // (HW_TODO) Fix two vertices, feel free to modify this
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

    //----------------------------------------------------
    // (HW Optional) Bonus part: Sphere collision
    Eigen::MatrixXd acceleration_collision =
        getSphereCollisionForce(sphere_center.cast<double>(), sphere_radius);
    //----------------------------------------------------

    if (time_integrator == IMPLICIT_EULER) {
        // Implicit Euler
        TIC(step)

        // (HW TODO)
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

        // update X and vel
        for (int i = 0; i < n_vertices; i++) {
            if (!dirichlet_bc_mask[i]) {
                X.row(i) -= delta_X.row(i);
                vel.row(i) = -delta_X.row(i) / h;
                if (vel.row(i).norm() > speed_of_light) {
                    vel.row(i) = (1 - 1e-6) * speed_of_light * vel.row(i).normalized();
                }
            }
        }

        TOC(step)
    }
    else if (time_integrator == SEMI_IMPLICIT_EULER) {
        // Semi-implicit Euler
        Eigen::MatrixXd force = -computeGrad(stiffness) / mass_per_vertex;
        for (int i = 0; i < n_vertices; i++) {
            if (!dirichlet_bc_mask[i]) {
                force.row(i) += acceleration_ext.transpose();
            }
        }
        if (enable_sphere_collision) {
            force += acceleration_collision;
        }
        Eigen::MatrixXd acceleration(n_vertices, 3);
        for (int i = 0; i < n_vertices; i++) {
            if (!dirichlet_bc_mask[i]) {
                auto beta = vel.row(i) / speed_of_light;
                float gamma = 1.0 / std::sqrt(1.0 - beta.squaredNorm());
                acceleration.row(i) = (force.row(i) - force.row(i).dot(beta) * beta) / gamma;
            }
        }

        // (HW TODO): Implement semi-implicit Euler time integration
        // Update X and vel
        for (int i = 0; i < n_vertices; i++) {
            vel.row(i) += h * std::pow(damping, h) * acceleration.row(i);
            if (vel.row(i).norm() > speed_of_light) {
                vel.row(i) = (1 - 1e-6) *speed_of_light * vel.row(i).normalized();
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
        // --------------------------------------------------
        // (HW TODO): Implement the gradient computation
        const Eigen::Vector3d x = X.row(e.first) - X.row(e.second);
        g.row(e.first) += stiffness * (x.norm() - E_rest_length[i]) * x.normalized();
        g.row(e.second) += -stiffness * (x.norm() - E_rest_length[i]) * x.normalized();
        // --------------------------------------------------
        i++;
    }
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
    auto k = stiffness;
    const auto I = Eigen::MatrixXd::Identity(3, 3);

    std::vector<Eigen::Triplet<double>> triplets;
    for (const auto& e : E) {
        // --------------------------------------------------
        // (HW TODO): Implement the sparse version Hessian computation
        // Remember to consider fixed points
        // You can also consider positive definiteness here
        const Eigen::Vector3d x = X.row(e.first) - X.row(e.second);
        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3, 3);
        auto beta1 = vel.row(e.first) / speed_of_light;
        auto beta2 = vel.row(e.second) / speed_of_light;
        const Eigen::MatrixXd He =
            k * (x * x.transpose() / x.squaredNorm() +
                 (1 - E_rest_length[i] / x.norm()) * (I - x * x.transpose() / x.squaredNorm()));
        const Eigen::MatrixXd He1 = (I - beta1 * beta1.transpose()) * He;
        const Eigen::MatrixXd He2 = (I - beta2 * beta2.transpose()) * He;
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
        // --------------------------------------------------
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

// ----------------------------------------------------------------------------------
// (HW Optional) Bonus part
Eigen::MatrixXd MassSpring::getSphereCollisionForce(Eigen::Vector3d center, double radius)
{
    Eigen::MatrixXd force = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); i++) {
        // (HW Optional) Implement penalty-based force here
        auto delta_x = X.row(i) - center.transpose();
        force.row(i) += collision_penalty_k *
                        std::max(0.0, collision_scale_factor * radius - delta_x.norm()) *
                        delta_x.normalized();
    }
    return force;
}
// ----------------------------------------------------------------------------------

/*
bool MassSpring::set_dirichlet_bc_mask(const std::vector<bool>& mask)
{
    if (mask.size() == X.rows()) {
        dirichlet_bc_mask = mask;
        return true;
    }
    else
        return false;
}

bool MassSpring::update_dirichlet_bc_vertices(const MatrixXd& control_vertices)
{
    for (int i = 0; i < dirichlet_bc_control_pair.size(); i++) {
        int idx = dirichlet_bc_control_pair[i].first;
        int control_idx = dirichlet_bc_control_pair[i].second;
        X.row(idx) = control_vertices.row(control_idx);
    }

    return true;
}

bool MassSpring::init_dirichlet_bc_vertices_control_pair(
    const MatrixXd& control_vertices,
    const std::vector<bool>& control_mask)
{
    if (control_mask.size() != control_vertices.rows())
        return false;

    // TODO: optimize this part from O(n) to O(1)
    // First, get selected_control_vertices
    std::vector<VectorXd> selected_control_vertices;
    std::vector<int> selected_control_idx;
    for (int i = 0; i < control_mask.size(); i++) {
        if (control_mask[i]) {
            selected_control_vertices.push_back(control_vertices.row(i));
            selected_control_idx.push_back(i);
        }
    }

    // Then update mass spring fixed vertices
    for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
        if (dirichlet_bc_mask[i]) {
            // O(n^2) nearest point search, can be optimized
            // -----------------------------------------
            int nearest_idx = 0;
            double nearst_dist = 1e6;
            VectorXd X_i = X.row(i);
            for (int j = 0; j < selected_control_vertices.size(); j++) {
                double dist = (X_i - selected_control_vertices[j]).norm();
                if (dist < nearst_dist) {
                    nearst_dist = dist;
                    nearest_idx = j;
                }
            }
            //-----------------------------------------

            X.row(i) = selected_control_vertices[nearest_idx];
            dirichlet_bc_control_pair.push_back(
                std::make_pair(i, selected_control_idx[nearest_idx]));
        }
    }

    return true;
}
*/

}  // namespace USTC_CG::node_relativity_mass_spring
