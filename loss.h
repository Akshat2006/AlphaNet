#pragma once
#ifndef LOSS_H
#define LOSS_H
#include "mathlinalg.h"
#include <cmath>

class cross_entropy
{
	double compute_loss(VECTOR& predictions, VECTOR& target);
	VECTOR gradient(VECTOR& predictions, VECTOR& target);

};
#endif // !LOSS_H
