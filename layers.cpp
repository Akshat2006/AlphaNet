#include "layers.h"

LAYER::LAYER(size_t input_sz, size_t output_sz, bool is_output)
    : input_size(input_sz),
      output_size(output_sz),
      weights(output_sz, input_sz, true),
      biases(output_sz),
      output(output_sz),
      activation(output_sz),
      delta(output_sz),
      is_output_layer(is_output) {}

VECTOR LAYER::activate(const VECTOR& input)
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