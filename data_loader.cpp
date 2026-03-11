#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <stdexcept>

std::vector<Sample> DataLoader::load_csv(const std::string& filepath, int max_samples) {
    std::vector<Sample> data;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::string line;
    int count = 0;

    std::streampos pos = file.tellg();
    std::getline(file, line);
    bool is_header = false;
    for (char c : line) {
        if (c == ',' || c == ' ') continue;
        if (!std::isdigit(c)) {
            is_header = true;
        }
        break;
    }
    if (!is_header) {
        file.seekg(pos);
    }

    while (std::getline(file, line)) {
        if (max_samples > 0 && count >= max_samples) break;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        int label = std::stoi(value);

        if (label >= 1 && label <= 26) {
            label -= 1;
        }

        VECTOR pixels(784);
        for (int i = 0; i < 784; ++i) {
            if (std::getline(ss, value, ',')) {
                pixels[i] = std::stod(value) / 255.0;
            }
        }

        VECTOR one_hot(26);
        if (label >= 0 && label < 26) {
            one_hot[label] = 1.0;
        }

        Sample sample;
        sample.pixels = pixels;
        sample.label = one_hot;
        sample.raw_label = label;
        data.push_back(sample);

        count++;
        if (count % 10000 == 0) {
            std::cout << "  Loaded " << count << " samples..." << std::endl;
        }
    }

    std::cout << "  Loaded " << count << " samples total." << std::endl;
    file.close();
    return data;
}

void DataLoader::shuffle(std::vector<Sample>& data) {
    static std::mt19937 gen(42);
    std::shuffle(data.begin(), data.end(), gen);
}

std::pair<std::vector<Sample>, std::vector<Sample>> DataLoader::split(
    const std::vector<Sample>& data, double train_ratio) {

    size_t train_size = static_cast<size_t>(data.size() * train_ratio);
    std::vector<Sample> train(data.begin(), data.begin() + train_size);
    std::vector<Sample> test(data.begin() + train_size, data.end());
    return {train, test};
}

char DataLoader::label_to_char(int label) {
    return static_cast<char>('A' + label);
}
