#include <vector>
#include <iostream>
#include <cmath>
#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <jsoncpp/json/json.h>

using namespace std;

class Layer
{
    uint in;
    uint out;
    string activationFunction;
    string layerType;
    float learningRate;
    public:

        Layer(string activationFunction, string layerType, uint in, uint out, float learningRate);

        void setLayer(vector<float> input);

        void updateLayerWeights(Layer nextLayer);

        vector<float> firstDeltas(vector<float> expected);

        vector<float> forwardPropagation();
        class Neuron
        {
            string activationFunction;
            float value;
            float delta_value;
            float cache_value;
            uint in;
            uint out;

            public:
                Neuron(string activationFunction, string layerType, uint in, uint out);

                void setValue(float input);

                void setDelta(float input);

                float getValue();

                float get_delta_value();

                float get_derivative();



                class Weight
                {
                    uint in;
                    uint out;
                    float weight;
                    float alpha;

                    string layerType;
                    string activationFunction;

                    public:
                        Weight(string activationFunction, string layerType, uint in, uint out);

                        void setWeight(float inp);

                        void update(float gradient, float learningRate);

                        void initWeight();

                        float getWeight();

                        float m;

                        float v;
                };

                vector<Weight> weights;
                string layerType;
        };

        class Bias
        {
            float biasValue;
            bool exists;

            public:
                Bias();

                void setBias(float inp);

                void update(float gradient);

                float getBias();

                bool getExists();

                void setExists(bool exists);

                float m;

                float v;

                float alpha;
        };

        Bias bias;

        vector<Neuron> neurons;

        vector<float> backwardPropagation(vector<float> deltas_last_layer, Bias bias_infront_layer);
};

class NeuralNetwork
{
    public:
        NeuralNetwork(vector<uint> topology, float learningRate, vector<string> activationFunctions, string optimizer_);

        vector<float> forward(vector<float> input);

        void backward(vector<float> expected);

        vector<Layer> layers;

        void downloadWeights();

        void uploadWeights(string filepath);
        
    private:
        void calc_errors(vector<float> expected);

        void update_weights();

        float learningRate;
        string optimizer;
        vector<uint> topology;
        vector<string> activationFunctions;
};

float ReLU(float inp);

float ReLUDerivative(float inp);

float HeRandom(uint input);