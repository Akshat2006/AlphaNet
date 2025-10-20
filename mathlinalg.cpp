#include "mathlinalg.h"

VECTOR::VECTOR(size_t size, bool require_random) : size_(size) {
    data.resize(size);
    if (require_random) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 0.1);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
    }
}

size_t VECTOR::size() const {
    return size_;
}

double& VECTOR::operator[](size_t index) {
    return data[index];
}

const double& VECTOR::operator[](size_t index) const {
    return data[index];
}

VECTOR VECTOR::operator+(const VECTOR& other) {
    if (size_ != other.size_) {
        throw std::invalid_argument("size mismatch!");
    }
    VECTOR result(size_);
    for (size_t i = 0; i < size_; i++) {
        result[i] = data[i] + other[i];
    }
    return result;
}

VECTOR VECTOR::operator*(double scalar) {
    VECTOR result(size_);
    for (size_t i = 0; i < size_; i++) {
        result[i] = data[i] * scalar;
    }
    return result;
}

double VECTOR::dot(const VECTOR& other) const {
    if (size_ != other.size_) throw std::invalid_argument("Size mismatch");
    double result = 0.0;
    for (size_t i = 0; i < size_; i++) {
        result += data[i] * other[i];
    }
    return result;
}

VECTOR VECTOR::hadamard(const VECTOR& other) const {
    if (size_ != other.size_) throw std::invalid_argument("Size mismatch");
    VECTOR result(size_);
    for (size_t i = 0; i < size_; i++) {
        result[i] = data[i] * other[i];
    }
    return result;
}

MATRIX::MATRIX(size_t rows, size_t cols, bool required_random) {
    rows_ = rows;
    cols_ = cols;
    for (size_t i = 0; i < rows; ++i) {
        data.emplace_back(cols, required_random);
    }
}

size_t MATRIX::rows() {
    return rows_;
}

size_t MATRIX::cols() {
    return cols_;
}

VECTOR& MATRIX::operator[](size_t row) {
    return data[row];
}

const VECTOR& MATRIX::operator[](size_t row) const {
    return data[row];
}

MATRIX MATRIX::operator+(const MATRIX& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Shape mismatch");
    }
    MATRIX result(rows_, cols_);
    for (size_t i = 0; i < rows(); i++) {
        result[i] = data[i] + other.data[i];
    }
    return result;
}

MATRIX MATRIX::operator*(double scalar) {
    MATRIX result(rows_, cols_);
    for (size_t i = 0; i < rows_; i++) {
        result[i] = data[i] * scalar;
    }
    return result;
}

MATRIX MATRIX::T() const {
    MATRIX result(cols_, rows_);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result[j][i] = data[i][j];
        }
    }
    return result;
}

VECTOR MATRIX::operator*(const VECTOR& vec) const {
    if (cols_ != vec.size()) {
        throw std::invalid_argument("Matrix columns != vector size");
    }
    VECTOR result(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        result[i] = data[i].dot(vec);
    }
    return result;
}

MATRIX MATRIX::outer_product(const VECTOR& a, const VECTOR& b) {
    MATRIX result(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}