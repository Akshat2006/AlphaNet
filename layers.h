#ifndef LAYERS_H
#define LAYERS_H

#include "mathlinalg.h"
#include "activations.h"
#include <vector>

class LAYER {
private:
    MATRIX weights;
    VECTOR biases;
    VECTOR output;
    VECTOR activation;
    VECTOR delta;

    size_t input_size;
    size_t output_size;
    bool is_output_layer;

public:
    LAYER(size_t input_sz, size_t output_sz, bool is_output = false);

    VECTOR activate(const VECTOR& input);

    size_t getInputSize() const { return input_size; }
    size_t getOutputSize() const { return output_size; }
    bool getIsOutputLayer() const { return is_output_layer; }

    MATRIX& getWeights() { return weights; }
    const MATRIX& getWeights() const { return weights; }
    VECTOR& getBiases() { return biases; }
    const VECTOR& getBiases() const { return biases; }
    const VECTOR& getOutput() const { return output; }
    const VECTOR& getActivation() const { return activation; }
    const VECTOR& getDelta() const { return delta; }

    void setWeights(const MATRIX& new_weights) { weights = new_weights; }
    void setBiases(const VECTOR& new_biases) { biases = new_biases; }
    void setDelta(const VECTOR& new_delta) { delta = new_delta; }
};

#endif