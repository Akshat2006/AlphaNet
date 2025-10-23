#include "mathlinalg.h"
#include "loss.h"
#include "activations.h"
#include <vector>
#include <memory>


class LAYER
{
private:
	MATRIX weights;
	VECTOR biases;
	VECTOR output;
	VECTOR activation;
	VECTOR delta;

	size_t input_size;
	size_t output_size;

	bool is_output_layer;

public:
	LAYER(VECTOR& input, VECTOR& output, bool is_output = false) :
		input_size(input.size()),
		output_size(output.size()),
		weights(output.size(), input.size(), true),
		biases(input.size(), true),
		output(output_size),
		activation(output_size),
		delta(output_size),
		is_output_layer(is_output){ }	

	VECTOR& activate(VECTOR& input)
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


};