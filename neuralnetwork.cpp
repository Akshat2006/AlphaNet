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

	void backward(VECTOR& input, VECTOR& target)
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

	}
};