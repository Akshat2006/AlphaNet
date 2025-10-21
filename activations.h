#ifndef ACTIVATIONS_H
#define ACTIVATIONs_H

#include "mathlinalg.h"

class ReLu
{
public:
	VECTOR forward(const VECTOR& input);
	VECTOR derivative(const VECTOR& input);
};

class softmax
{
public:
	VECTOR forward(const VECTOR& input);
};

#endif