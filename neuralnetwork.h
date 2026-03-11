#pragma once
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layers.h"
#include "mathlinalg.h"
#include "activations.h"
#include "loss.h"
#include <vector>

class neural_net
{
private:
    std::vector<LAYER> layers;
    double learning_rate;

public:
    neural_net(const std::vector<size_t>& architecture, double learning_rate_ = 0.01);

    VECTOR forward(const VECTOR& input);
    void backward(const VECTOR& input, const VECTOR& target);
    double train(const VECTOR& input, const VECTOR& target);
    size_t predict(const VECTOR& input);
};

#endif
