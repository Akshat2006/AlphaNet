#include "mathlinalg.h"
#include "loss.h"


double cross_entropy::compute_loss(const VECTOR& predictions, const VECTOR& target)
	{
		double loss = 0.0;
		for (size_t i = 0; i < predictions.size(); i++)
		{
			loss -= target[i] * std::log(predictions[i]);
		}
		return loss;
	}
VECTOR cross_entropy::gradient(const VECTOR& predictions, const VECTOR& target)
{
	VECTOR grad(predictions.size());

	for (size_t i = 0; i < predictions.size(); i++)
	{
			grad[i] = predictions[i] - target[i];
	}
	return grad;

}

