#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "mathlinalg.h"

class ReLu
{
public:
	static VECTOR forward(const VECTOR& input);
	static VECTOR derivative(const VECTOR& input);
};

class softmax
{
public:
	static VECTOR forward(const VECTOR& input);
};

#endif