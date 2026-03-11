#pragma once
#ifndef MATHLINALG_H
#define MATHLINALG_H

#include <vector>
#include <stdexcept>
#include <random>
#include <cstddef>
#include <cmath>
#include <algorithm>

class VECTOR {
private:
    std::vector<double> data;
    size_t size_;

public:
    VECTOR();
    VECTOR(size_t size, bool require_random = false);

    size_t size() const;

    double& operator[](size_t index);
    const double& operator[](size_t index) const;

    VECTOR operator+(const VECTOR& other) const;
    VECTOR operator-(const VECTOR& other) const;
    VECTOR operator*(double scalar) const;
    double dot(const VECTOR& other) const;
    VECTOR hadamard(const VECTOR& other) const;
};

class MATRIX {
private:
    std::vector<VECTOR> data;
    size_t rows_, cols_;

public:
    MATRIX();
    MATRIX(size_t rows, size_t cols, bool required_random = false);

    size_t rows() const;
    size_t cols() const;

    VECTOR& operator[](size_t row);
    const VECTOR& operator[](size_t row) const;

    MATRIX operator+(const MATRIX& other) const;
    MATRIX operator-(const MATRIX& other) const;
    MATRIX operator*(double scalar) const;
    MATRIX T() const;
    VECTOR operator*(const VECTOR& vec) const;

    static MATRIX outer_product(const VECTOR& a, const VECTOR& b);
};

#endif