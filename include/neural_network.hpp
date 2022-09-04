#include <vector>
#include <iostream>
#include <cmath>
#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <jsoncpp/json/json.h>

using namespace std;

class Layer
{
    uint in;
    uint out;
    string activationFunction;
    string layerType;
    double learningRate;
    public:

        Layer(string activationFunction, string layerType, uint in, uint out, double learningRate);

        void setLayer(vector<double> input);

        void updateLayerWeights(Layer nextLayer);

        vector<double> firstDeltas(vector<double> expected);

        vector<double> forwardPropagation();
        class Neuron
        {
            string activationFunction;
            double value;
            double delta_value;
            double cache_value;
            uint in;
            uint out;

            public:
                Neuron(string activationFunction, string layerType, uint in, uint out);

                void setValue(double input);

                void setDelta(double input);

                double getValue();

                double get_delta_value();

                double get_derivative();



                class Weight
                {
                    uint in;
                    uint out;
                    double weight;
                    double alpha;

                    string layerType;
                    string activationFunction;

                    public:
                        Weight(string activationFunction, string layerType, uint in, uint out);

                        void setWeight(double inp);

                        void update(double gradient, double learningRate);

                        void initWeight();

                        double getWeight();

                        double m;

                        double v;
                };

                vector<Weight> weights;
                string layerType;
        };

        class Bias
        {
            double biasValue;
            bool exists;

            public:
                Bias();

                void setBias(double inp);

                void update(double gradient);

                double getBias();

                bool getExists();

                void setExists(bool exists);

                double m;

                double v;

                double alpha;
        };

        Bias bias;

        vector<Neuron> neurons;

        vector<double> backwardPropagation(vector<double> deltas_last_layer, Bias bias_infront_layer);
};

class NeuralNetwork
{
    public:
        NeuralNetwork(vector<uint> topology, double learningRate, vector<string> activationFunctions, string optimizer_);

        vector<double> forward(vector<double> input);

        void backward(vector<double> expected);

        vector<Layer> layers;

        void downloadWeights();

        void uploadWeights(string filepath);
        
    private:
        void calc_errors(vector<double> expected);

        void update_weights();

        double learningRate;
        string optimizer;
        vector<uint> topology;
        vector<string> activationFunctions;
};

double ReLU(double inp);

double ReLUDerivative(double inp);

double HeRandom(uint input);