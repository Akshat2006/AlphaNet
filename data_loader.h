#pragma once
#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "mathlinalg.h"
#include <string>
#include <vector>
#include <utility>

struct Sample {
    VECTOR pixels;
    VECTOR label;
    int raw_label;
};

class DataLoader {
public:
    static std::vector<Sample> load_csv(const std::string& filepath, int max_samples = -1);
    static void shuffle(std::vector<Sample>& data);
    static std::pair<std::vector<Sample>, std::vector<Sample>> split(
        const std::vector<Sample>& data, double train_ratio = 0.8);
    static char label_to_char(int label);
};

#endif
