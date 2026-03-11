#include "neuralnetwork.h"

neural_net::neural_net(const std::vector<size_t>& architecture, double learning_rate_)
    : learning_rate(learning_rate_)
{
    for (size_t i = 0; i + 1 < architecture.size(); i++)
    {
        bool is_output = (i == architecture.size() - 2);
        layers.emplace_back(architecture[i], architecture[i + 1], is_output);
    }
}

VECTOR neural_net::forward(const VECTOR& input)
{
    VECTOR current = input;
    for (auto& layer : layers)
    {
        current = layer.activate(current);
    }
    return current;
}

void neural_net::backward(const VECTOR& input, const VECTOR& target)
{
    LAYER& output_layer = layers.back();
    output_layer.setDelta(cross_entropy::gradient(output_layer.getActivation(), target));

    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i)
    {
        LAYER& current = layers[i];
        LAYER& next = layers[i + 1];

        VECTOR propagated = next.getWeights().T() * next.getDelta();
        VECTOR relu_deriv = ReLu::derivative(current.getOutput());
        current.setDelta(propagated.hadamard(relu_deriv));
    }

    VECTOR prev_activation = input;
    for (size_t i = 0; i < layers.size(); ++i) {
        LAYER& layer = layers[i];

        for (size_t j = 0; j < layer.getOutputSize(); ++j) {
            for (size_t k = 0; k < layer.getInputSize(); ++k) {
                layer.getWeights()[j][k] -= learning_rate * layer.getDelta()[j] * prev_activation[k];
            }
            layer.getBiases()[j] -= learning_rate * layer.getDelta()[j];
        }

        prev_activation = layer.getActivation();
    }
}

double neural_net::train(const VECTOR& input, const VECTOR& target)
{
    VECTOR prediction = forward(input);
    double loss = cross_entropy::compute_loss(prediction, target);
    backward(input, target);
    return loss;
}

size_t neural_net::predict(const VECTOR& input)
{
    VECTOR output = forward(input);
    size_t max_index = 0;

    for (size_t i = 1; i < output.size(); ++i)
    {
        if (output[i] > output[max_index])
        {
            max_index = i;
        }
    }
    return max_index;
}