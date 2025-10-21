#include "mathlinalg.h"
#include <cmath>

class cross_entropy
{	

	double compute_loss(VECTOR& predictions, VECTOR& target)
	{
		double loss = 0.0;
		for (size_t i = 0; i < predictions.size(); i++)
		{
			loss -= target[i] * std::log(predictions[i]);
		}
		return loss;
	}
	VECTOR gradient(VECTOR& predictions, VECTOR& target)
	{
		VECTOR grad(predictions.size());

		for (size_t i = 0; i < predictions.size(); i++)
		{
			grad[i] = predictions[i] - target[i];
		}
		return grad;

	}
};
