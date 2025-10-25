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
	neural_net(std::vector <LAYER>& architecture, double learning_rate_ = 0.01)
	{
		learning_rate = learning_rate_;
		for (size_t i = 0; i < architecture.size(); i++)
		{
			bool is_output = (i == architecture.size() - 2);
			layers.emplace_back(architecture[i], architecture[i + 1], is_output);
		}
	}

	const VECTOR& forward(const	VECTOR& input)
	{
		VECTOR current = input;
		for (auto& layer : layers)
		{
			current = layer.activate(current);

		}
		return current;
	}

	void backward(const VECTOR& input, const VECTOR& target)
	{
		LAYER& output_layer = layers.back();
		output_layer.setDelta(cross_entropy::gradient(output_layer.getActivation(), target));

		for (size_t i = layers.size() - 2; i >= 0; --i)
		{
			LAYER& current = layers[i];
			LAYER& next = layers[i + 1];

			current.setDelta(
				(next.getweights().T() * next.getDelta())
				.hadamard(ReLu::derivative(current.getOutput()))
			);

		}

		VECTOR prev_activation = input;
		for (size_t i = 0; i < layers.size(); ++i) {
			LAYER& layer = layers[i];

			for (size_t j = 0; j < layer.getOutputSize(); ++j) {
				for (size_t k = 0; k < layer.getInputSize(); ++k) {
					layer.getweights()[j][k] -= learning_rate * layer.getDelta()[j] * prev_activation[k];
				}
				layer.getBiases()[j] -= learning_rate * layer.getDelta()[j];
			}

			prev_activation = layer.getActivation();
		}
	}

	double train(const VECTOR& input, const VECTOR& target)
	{
		VECTOR prediction = forward(input);
		double loss = cross_entropy::compute_loss(input, target);
		backward(input, target);
		return loss;

	}

	size_t predict(const VECTOR& input)
	{
		VECTOR output = forward(input);
		size_t max_index = 0;
		double max_probability = output[0];

		for (size_t i = 1; i < output.size(); ++i)
		{
			if (output[i] > output[max_index])
			{
				max_probability = output[i];
				max_index = i;
			}
		}
		return max_index;
	}

};