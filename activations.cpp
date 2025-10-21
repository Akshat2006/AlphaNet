#include "mathlinalg.h"
#include <cmath>

class ReLu
{
public:
	VECTOR forward(const VECTOR& input)
	{
		VECTOR output(input.size());
		for(size_t i = 0 ; i < input.size() ; i++)
		{
			output[i] = std::max(input[i], 0.0);
		}
		return output;
	}

	VECTOR derivative(const VECTOR& input)
	{
		VECTOR deriv(input.size());
		for (size_t i = 0; i < input.size(); i++)
		{
			deriv[i] = (input[i] > 0) ? 1.0 : 0.0; 
		}
		return deriv;
	}
};

class softmax
{
public:
	VECTOR forward(const VECTOR& input)
	{
		VECTOR output(input.size());

		double max_val = input[0];
		for (size_t i = 1; i < input.size(); ++i) {
			if (input[i] > max_val) max_val = input[i];
		}

		double sum = 0.0;
		for (size_t i = 0; i < input.size(); ++i) {
			output[i] = std::exp(input[i] - max_val);
			sum += output[i];
		}
		for (size_t i = 0; i < input.size(); ++i) { //normalize ts gng
			output[i] /= sum;
		}

		return output;
	}
};