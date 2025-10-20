#pragma once
#ifndef MATHLINALG_H
#define MATHLINALG_H

#include <vector>
#include <stdexcept>
#include <random>
#include <cstddef>

class VECTOR {
private:
    std::vector<double> data;
    size_t size_;

public:

    VECTOR(size_t size, bool require_random = false);

    size_t size() const;

    double& operator[](size_t index);
    const double& operator[](size_t index) const;

    VECTOR operator+(const VECTOR& other);
    VECTOR operator*(double scalar);
    double dot(const VECTOR& other) const;
    VECTOR hadamard(const VECTOR& other) const;
};

class MATRIX {
private:
    std::vector<VECTOR> data;
    size_t rows_, cols_;

public:
    MATRIX(size_t rows, size_t cols, bool required_random = false);

    size_t rows();
    size_t cols();
  
    VECTOR& operator[](size_t row);
    const VECTOR& operator[](size_t row) const;

    MATRIX operator+(const MATRIX& other);
    MATRIX operator*(double scalar);
    MATRIX T() const;
    VECTOR operator*(const VECTOR& vec) const;

    static MATRIX outer_product(const VECTOR& a, const VECTOR& b);
};

#endif // MATHLINALG_H