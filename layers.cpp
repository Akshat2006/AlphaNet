#include "layers.h"
#include "mathlinalg.h"
#include "activations.h"

LAYER::LAYER(VECTOR& input, VECTOR& output, bool is_output) :
    input_size(input.size()),
    output_size(output.size()),
    weights(output.size(), input.size(), true),
    biases(input.size(), true),
    output(output_size),
    activation(output_size),
    delta(output_size),
    is_output_layer(is_output){}

VECTOR& LAYER::activate(VECTOR& input)
{
    output = weights * input;

    for (size_t i = 0; i < output_size; ++i) {
        output[i] += biases[i];
    }
    if (is_output_layer) {
        activation = softmax::forward(output);
    }
    else {
        activation = ReLu::forward(output);
    }

    return activation;
}