#include <iostream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <unistd.h>
#include "../include/neural_network.hpp"
#include "../include/data_processing/power_consumption.hpp"

void testNeuralNet();
void learnPowerConsumption();

int main(int argc, char **argv)
{
    srand(100);

    learnPowerConsumption();

    return 0;
}

void learnPowerConsumption()
{
    dataset processedData(32000, "../../src/data_processing/data/tetuanCityPowerConsumption.csv", "Tetuan City Power Consumption");
    cout << "processing data" << endl;
    processedData.shuffle();

    vector<dataset> train_test_data = processedData.split(26000, "train_data", "test_data");
    dataset train_data = train_test_data[0];
    dataset test_data = train_test_data[1];
    
    NeuralNetwork nn({6, 10, 5, 3}, 0.01f, {"Linear", "ReLU", "ReLU", "Linear"}, "adam");
    vector<double> input(6, 0);
    vector<double> output(3, 0);

    double avg = 0;
    int avg_cnt = 0;
    double total = 0;
    int view_cnt = 1;
    cout << "testing initial performance..." << endl;
    for(uint i = 0; i < 200; i++)
    {
        for(uint j = 0; j < test_data.data.size(); j++)
        {
            if(j < input.size()) input[j] = test_data.data[i][j];
            else output[j-input.size()] = test_data.data[i][j];
        }

        vector<double> nno = nn.forward(input);

        double sum = 0;
        for(uint i = 0; i < output.size(); i++)
        {
            sum += abs(test_data.minMaxUnnormalization(nno[i], i+5) - test_data.minMaxUnnormalization(output[i], i+5));
        }
        avg_cnt++;
        total += sum/3;
        avg = total/avg_cnt;
    }
    cout << "training..." << endl;
    auto start = chrono::_V2::high_resolution_clock::now();
    for(uint i = 0; i < train_data.data.size(); i++)
    {
        for(uint j = 0; j < train_data.data[i].size(); j++)
        {
            if(j < input.size()) input[j] = train_data.data[i][j];
            else output[j-input.size()] = train_data.data[i][j];
        }

        nn.forward(input);
        nn.backward(output);

        if(i % 1000 == 0) cout << "[" << i/1000 << "/"<<ceil(train_data.data.size()/1000)<<"]" << endl;
    }
    cout << "[" << ceil(train_data.data.size()/1000) << "/"<<ceil(train_data.data.size()/1000)<<"]" << endl;
    auto stop = chrono::_V2::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);

    double error_before_training = avg;
    avg = 0;
    avg_cnt = 0;
    total = 0;
    view_cnt = 1;
    cout << "\ntesting final performance..." << endl;
    for(uint i = 0; i < test_data.data.size(); i++)
    {
        for(uint j = 0; j < test_data.data.size(); j++)
        {
            if(j < input.size()) input[j] = test_data.data[i][j];
            else output[j-input.size()] = test_data.data[i][j];
        }

        vector<double> nno = nn.forward(input);

        double sum = 0.f;
        for(uint i = 0; i < output.size(); i++)
        {
            sum += abs(test_data.minMaxUnnormalization(nno[i], i+5) - test_data.minMaxUnnormalization(output[i], i+5));
        }
        avg_cnt++;
        total += sum/3;
        avg = total/avg_cnt;
    }
    cout << "\nAverage error before training: " << error_before_training << endl;
    cout << "Average error after training: " << avg << endl;
    cout << "\nThe network's predictions are " << (1 - (avg/error_before_training)) * 100 << " percent more accurate than randomly choosing. " << endl;
    cout << "\nThe error is the average difference between the network's prediction of\nthe three region's power consumption and the actual power consumption." << endl;
    if(duration.count() * 0.000001 > 60) cout << "\nTraining time: " << (duration.count() * 0.000001)/60 << " minutes" << endl;
    else cout << "\nTraining time: " << duration.count() * 0.000001 << " seconds" << endl;

    cout << "downloading the statistics of this network." << endl;
    nn.downloadWeights();
    nn.uploadWeights("../../src/neural_network/weights/weights.json");
    avg = 0.f;
    avg_cnt = 0;
    total = 0;
    for(uint i = 0; i < test_data.data.size(); i++)
    {
        for(uint j = 0; j < test_data.data.size(); j++)
        {
            if(j < input.size()) input[j] = test_data.data[i][j];
            else output[j-input.size()] = test_data.data[i][j];
        }

        vector<double> nno = nn.forward(input);

        double sum = 0.f;
        for(uint i = 0; i < output.size(); i++)
        {
            sum += abs(test_data.minMaxUnnormalization(nno[i], i+5) - test_data.minMaxUnnormalization(output[i], i+5));
        }
        avg_cnt++;
        total += sum/3.f;
        avg = total/avg_cnt;
    }

    cout << "downloaded avg: " << avg << endl; 
}
