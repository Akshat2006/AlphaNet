#pragma once
#ifndef LOSS_H
#define LOSS_H
#include "mathlinalg.h"
#include <cmath>

class cross_entropy
{
public:
	static double compute_loss(const VECTOR& predictions,const VECTOR& target);
	static VECTOR gradient(const VECTOR& predictions,const VECTOR& target);

};
#endif // !LOSS_H
