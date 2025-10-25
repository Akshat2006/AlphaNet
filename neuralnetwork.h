#pragma once


#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include "layers.h"
#include "mathlinalg.h"
#include "activations.h"
#include "loss.h"
#include <vector>


class neural_net
{
private:
	std::vector<LAYER>layers;
	double learning_rate;
public:
	neural_net(std::vector<LAYER>& architecture, double learning_rate_ = 0.01);
	const VECTOR& forward(const VECTOR& input);
	void backward(VECTOR& input, VECTOR& target);	
	
};


#endif // !NEUTRALNETWORK_H

