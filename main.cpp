#include "neuralnetwork.h"
#include "data_loader.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

void print_banner() {
    std::cout << "\n";
    std::cout << "  +-----------------------------------------------+\n";
    std::cout << "  |            AlphaNet - Neural Network           |\n";
    std::cout << "  |      Handwritten Letter Recognition (A-Z)      |\n";
    std::cout << "  |            Built from scratch in C++           |\n";
    std::cout << "  +-----------------------------------------------+\n";
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    print_banner();

    std::string train_csv = "data/emnist-letters-train.csv";
    std::string test_csv  = "data/emnist-letters-test.csv";
    int max_train_samples = -1;
    int max_test_samples  = -1;
    int epochs = 5;
    double learning_rate = 0.001;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--train" && i + 1 < argc) train_csv = argv[++i];
        else if (arg == "--test" && i + 1 < argc) test_csv = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) learning_rate = std::stod(argv[++i]);
        else if (arg == "--max-train" && i + 1 < argc) max_train_samples = std::stoi(argv[++i]);
        else if (arg == "--max-test" && i + 1 < argc) max_test_samples = std::stoi(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: AlphaNet [options]\n"
                      << "  --train <path>      Training CSV (default: data/emnist-letters-train.csv)\n"
                      << "  --test  <path>      Test CSV (default: data/emnist-letters-test.csv)\n"
                      << "  --epochs <n>        Number of epochs (default: 5)\n"
                      << "  --lr <rate>         Learning rate (default: 0.001)\n"
                      << "  --max-train <n>     Max training samples (default: all)\n"
                      << "  --max-test <n>      Max test samples (default: all)\n";
            return 0;
        }
    }

    std::cout << "[1/4] Loading training data from: " << train_csv << "\n";
    std::vector<Sample> train_data;
    try {
        train_data = DataLoader::load_csv(train_csv, max_train_samples);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::cerr << "\nPlease download the EMNIST Letters dataset:\n";
        std::cerr << "  1. Run: python generate_data.py\n";
        std::cerr << "  2. Or visit: https://www.kaggle.com/datasets/crawford/emnist\n";
        std::cerr << "  3. Place CSVs in a 'data/' folder next to the executable\n";
        return 1;
    }

    std::cout << "[2/4] Loading test data from: " << test_csv << "\n";
    std::vector<Sample> test_data;
    try {
        test_data = DataLoader::load_csv(test_csv, max_test_samples);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n  Training samples: " << train_data.size() << "\n";
    std::cout << "  Test samples:     " << test_data.size() << "\n";
    std::cout << "  Input size:       784 (28x28 pixels)\n";
    std::cout << "  Output classes:   26 (A-Z)\n";
    std::cout << "  Learning rate:    " << learning_rate << "\n";
    std::cout << "  Epochs:           " << epochs << "\n\n";

    std::cout << "[3/4] Building neural network: 784 -> 128 -> 64 -> 26\n\n";
    std::vector<size_t> architecture = {784, 128, 64, 26};
    neural_net net(architecture, learning_rate);

    DataLoader::shuffle(train_data);

    std::cout << "[4/4] Training...\n";
    std::cout << "---------------------------------------------------\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        double total_loss = 0.0;
        int correct = 0;

        DataLoader::shuffle(train_data);

        for (size_t i = 0; i < train_data.size(); ++i) {
            double loss = net.train(train_data[i].pixels, train_data[i].label);
            total_loss += loss;

            size_t predicted = net.predict(train_data[i].pixels);
            if (static_cast<int>(predicted) == train_data[i].raw_label) {
                correct++;
            }

            if ((i + 1) % 5000 == 0) {
                double running_acc = 100.0 * correct / (i + 1);
                std::cout << "  Epoch " << (epoch + 1) << "/" << epochs
                          << " | Sample " << std::setw(6) << (i + 1) << "/" << train_data.size()
                          << " | Loss: " << std::fixed << std::setprecision(4) << (total_loss / (i + 1))
                          << " | Acc: " << std::fixed << std::setprecision(1) << running_acc << "%"
                          << std::endl;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_seconds = std::chrono::duration<double>(epoch_end - epoch_start).count();

        double avg_loss = total_loss / train_data.size();
        double train_acc = 100.0 * correct / train_data.size();

        std::cout << "  >> Epoch " << (epoch + 1) << " DONE"
                  << " | Avg Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Train Acc: " << std::fixed << std::setprecision(2) << train_acc << "%"
                  << " | Time: " << std::fixed << std::setprecision(1) << epoch_seconds << "s"
                  << std::endl;
        std::cout << "---------------------------------------------------\n";
    }

    std::cout << "\nEvaluating on test set...\n";
    int test_correct = 0;
    double test_loss = 0.0;

    for (size_t i = 0; i < test_data.size(); ++i) {
        size_t predicted = net.predict(test_data[i].pixels);
        if (static_cast<int>(predicted) == test_data[i].raw_label) {
            test_correct++;
        }

        VECTOR pred_vec = net.forward(test_data[i].pixels);
        test_loss += cross_entropy::compute_loss(pred_vec, test_data[i].label);
    }

    double test_acc = 100.0 * test_correct / test_data.size();
    double avg_test_loss = test_loss / test_data.size();

    std::cout << "\n+-----------------------------------------------+\n";
    std::cout << "|                 TEST RESULTS                  |\n";
    std::cout << "+-----------------------------------------------+\n";
    std::cout << "|  Test Accuracy: " << std::fixed << std::setprecision(2) << std::setw(6) << test_acc
              << "%                        |\n";
    std::cout << "|  Test Loss:     " << std::fixed << std::setprecision(4) << std::setw(8) << avg_test_loss
              << "                       |\n";
    std::cout << "|  Correct:       " << std::setw(6) << test_correct << " / " << std::setw(6) << test_data.size()
              << "                |\n";
    std::cout << "+-----------------------------------------------+\n\n";

    std::cout << "Sample predictions:\n";
    std::cout << "  Actual -> Predicted  [Match?]\n";
    std::cout << "  ---------------------------------\n";

    int show_count = std::min(20, static_cast<int>(test_data.size()));
    for (int i = 0; i < show_count; ++i) {
        size_t predicted = net.predict(test_data[i].pixels);
        char actual_char = DataLoader::label_to_char(test_data[i].raw_label);
        char pred_char = DataLoader::label_to_char(static_cast<int>(predicted));
        bool match = (static_cast<int>(predicted) == test_data[i].raw_label);

        std::cout << "     " << actual_char << "   ->    " << pred_char
                  << "        [" << (match ? "OK" : "WRONG") << "]" << std::endl;
    }
    std::cout << "  ---------------------------------\n\n";

    return 0;
}
