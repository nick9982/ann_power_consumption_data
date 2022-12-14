#include "../../include/neural_network.hpp"

double beta1 = 0.9;

double beta2 = 0.999;

int epoch = 1;

string optimizer = "none";

void Layer::setLayer(vector<double> input)
{
    if(input.size() > this->in)
    {
        throw std::invalid_argument("Input size is larger than network parameters.");
    }
    for(uint i = 0; i < this->neurons.size(); i++)
    {
        neurons[i].setValue(input[i]);
    }
}

Layer::Bias::Bias()
{
    this->biasValue = 0;
    this->exists = false;
    this->m = 0.f;
    this->v = 0.f;
    this->alpha = 0.001f;
}

vector<double> Layer::forwardPropagation()
{
    vector<double> result(this->out, 0);
    for(uint i = 0; i < this->out; i++)
    {
        for(uint j = 0; j < this->in; j++)
        {
            Neuron neuron = neurons[j];
            float NeuronValue = neuron.getValue();
            result[i] += neuron.weights[i].getWeight() * NeuronValue;
        }
        if(this->bias.getExists()) result[i] += this->bias.getBias();
    }
    return result;
}

void Layer::updateLayerWeights(Layer nextLayer)
{
    for(uint i = 0; i < this->in; i++)
    {
        for(uint j = 0; j < this->out; j++)
        {
            this->neurons[i].weights[j].update(this->neurons[i].getValue() * nextLayer.neurons[j].get_delta_value(), this->learningRate);
        }
    }

    for(uint i = 0; i < this->out; i++)
    {
        this->bias.update(nextLayer.neurons[i].get_delta_value());
    }
}

vector<double> Layer::firstDeltas(vector<double> expected)
{
    vector<double> result(this->in, 0);
    for(uint i = 0; i < this->neurons.size(); i++)
    {
        double delta = (neurons[i].getValue() - expected[i]) * neurons[i].get_derivative();
        result[i] = delta;
        this->neurons[i].setDelta(delta);
    }
    return result;
}

vector<double> Layer::backwardPropagation(vector<double> deltas_last_layer, Bias bias_infront_layer)
{
    vector<double> result(this->in, 0);
    for(uint i = 0; i < this->in; i++)
    {
        double neuron_derivative = this->neurons[i].get_derivative();
        for(uint j = 0; j < this->out; j++)
        {
            result[i] += this->neurons[i].weights[j].getWeight() * deltas_last_layer[j] * neuron_derivative;
        }
        if(bias_infront_layer.getExists()) result[i] += bias_infront_layer.getBias() * neuron_derivative;
        this->neurons[i].setDelta(result[i]);
    }
    return result;
}

Layer::Neuron::Neuron(string activationFunction, string layerType, uint in, uint out)
{
    this->layerType = layerType;
    this->activationFunction = activationFunction;
    this->in = in;
    this->out = out;
    for(uint i = 0; i < out; i++)
    {
        Weight newWeight(activationFunction, layerType, in, out);
        this->weights.push_back(newWeight);
    }
}

void Layer::Neuron::setValue(double input)
{
    this->cache_value = input;
    if (this->activationFunction == "ReLU")
    {
        this->value = ReLU(input);
    }
    else if(this->activationFunction == "Linear")
    {
        this->value = input;
    }
}

void Layer::Neuron::setDelta(double input)
{
    this->delta_value = input;
}

double Layer::Neuron::getValue()
{
    return this->value;
}

double Layer::Neuron::get_delta_value()
{
    return this->delta_value;
}

double Layer::Neuron::get_derivative()
{
    if(this->activationFunction.compare("ReLU") == 0)
    {
        return ReLUDerivative(this->cache_value);
    }
    else
    {
        return 1;
    }
}

Layer::Neuron::Weight::Weight(string activationFunction, string layerType, uint in, uint out)
{
    this->in = in;
    this->out = out;
    this->activationFunction = activationFunction;
    this->layerType = layerType;

    this->alpha = 0.001f;

    this->initWeight();
}

void Layer::Neuron::Weight::setWeight(double inp)
{
    this->weight = inp;
}

void Layer::Neuron::Weight::update(double gradient, double learningRate)
{
    if(optimizer.compare("adam") == 0)
    {
        this->m = beta1 * this->m + (1 - beta1) * gradient;
        this->v = beta2 * this->v + (1 - beta2) * pow(gradient, 2);

        double mhat = this->m / (1 - pow(beta1, 1));
        double vhat = this->v / (1 - pow(beta2, 1));

        this->weight -= (this->alpha / (sqrt(vhat + 1e-8)) * mhat);
    }
    else this->weight -= learningRate * gradient;
}

void Layer::Neuron::Weight::initWeight()
{
    if(this->activationFunction.compare("ReLU") == 0 || this->activationFunction.compare("Linear") == 0)
    {
        this->weight = HeRandom(this->in);
    }
}

double Layer::Neuron::Weight::getWeight()
{
    return this->weight;
}

void Layer::Bias::setBias(double inp)
{
    this->biasValue = inp;
}

void Layer::Bias::setExists(bool exists)
{
    this->exists = exists;
}

void Layer::Bias::update(double gradient)
{
    if(optimizer.compare("adam") == 0)
    {
        this->m = beta1 * this->m + (1 - beta1) * gradient;
        this->v = beta2 * this->v + (1 - beta2) * pow(gradient, 2);

        double mhat = this->m / (1 - pow(beta1, epoch));
        double vhat = this->v / (1 - pow(beta2, epoch));

        this->biasValue -= (this->alpha / (sqrt(vhat + 1e-8)) * mhat) ;
    }
    else this->biasValue -= this->alpha * gradient;
}

double Layer::Bias::getBias()
{
    return this->biasValue;
}

bool Layer::Bias::getExists()
{
    return this->exists;
}

Layer::Layer(string activationFunction, string layerType, uint in, uint out, double learningRate)
{
    this->in = in;
    this->out = out;
    this->layerType = layerType;
    this->activationFunction = activationFunction;
    this->learningRate = learningRate;
    for(uint i = 0; i < this->in; i++)
    {
        Neuron neuron = Neuron(this->activationFunction, this->layerType, this->in, this->out);
        neurons.push_back(neuron);
    }
    if(this->layerType.compare("Hidden") == 0)
    {
        this->bias.setBias(0);
        this->bias.setExists(true);
    }
    
}

double ReLU(double inp)
{
    if(inp > 0) return inp;
    return 0;
}

double ReLUDerivative(double inp)
{
    if(inp <= 0) return 0;
    return 1;
}

NeuralNetwork::NeuralNetwork(vector<uint> topology, double learningRate, vector<string> activationFunctions, string optimizer_)
{
    this->optimizer = optimizer_;
    this->learningRate = learningRate;
    this->activationFunctions = activationFunctions;
    this->topology = topology;
    string layerType = "Hidden";
    for(uint i = 0; i < topology.size(); i++)
    {
        if(i == 0) layerType = "Input";
        else if(i == topology.size()-1) layerType = "Output";
        Layer layer(activationFunctions[i], layerType, topology[i], topology[i+1], learningRate);
        this->layers.push_back(layer);
        layerType = "Hidden";
    }
}

vector<double> NeuralNetwork::forward(vector<double> input)
{
    layers[0].setLayer(input);
    vector<double> result;
    for(uint i = 0; i < layers.size(); i++)
    {
        if(i != layers.size()-1) 
        {
            result = layers[i].forwardPropagation();
            layers[i+1].setLayer(result);
        }
    }
    return result;
}

void NeuralNetwork::backward(vector<double> expected)
{
    this->calc_errors(expected);
    this->update_weights();

    epoch++;
}

void NeuralNetwork::calc_errors(vector<double> expected)
{
    vector<double> deltas = this->layers[this->layers.size()-1].firstDeltas(expected);

    for(uint i = this->layers.size()-2; i > 0; i--)
    {
        deltas = this->layers[i].backwardPropagation(deltas, this->layers[i-1].bias);
    }
}

void NeuralNetwork::update_weights()
{
    for(uint i = 0; i < this->layers.size()-1; i++)
    {
        this->layers[i].updateLayerWeights(this->layers[i+1]);
    }
}

double HeRandom(uint input)
{
    double hi = 2.f/((double)(sqrt(input)));
    double lo = -hi;
    double number = lo + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(hi-lo)));
    return number;
}

void NeuralNetwork::downloadWeights()
{
    ofstream weightsFile("../../src/neural_network/weights/weights.json");
    Json::Value root;
    root["optimizer"] = this->optimizer;
    root["learningRate"] = this->learningRate;
    Json::Value topology(Json::arrayValue);
    for(uint i = 0; i < this->topology.size(); i++)
    {
        topology.append(this->topology[i]);
    }
    root["topology"] = topology;
    Json::Value actfuncs(Json::arrayValue);
    for(uint i = 0; i < this->activationFunctions.size(); i++)
    {
        actfuncs.append(this->activationFunctions[i]);
    }
    root["activation functions"] = actfuncs;
    Json::Value weights;
    for(uint i = 0; i < this->layers.size()-1; i++)
    {
        for(uint j = 0; j < this->layers[i].neurons.size(); j++)
        {
            for(uint z = 0; z < this->layers[i].neurons[j].weights.size(); z++)
            {
                weights.append(Json::Value(this->layers[i].neurons[j].weights[z].getWeight()));
            }
        }
    }
    Json::Value biases;
    for(uint i = 0; i < this->layers.size(); i++)
    {
        if(i != 0 && i != this->layers.size()-1) biases.append(this->layers[i].bias.getBias());
    }
    root["weights"] = weights;
    root["biases"] = biases;
    Json::StyledWriter sw;
    weightsFile << sw.write(root);
    weightsFile.close();
}

void NeuralNetwork::uploadWeights(string filepath)
{
    ifstream weightsFile(filepath);
    Json::Reader reader;
    Json::Value root;
    string errs;
    weightsFile >> root;
    reader.parse(weightsFile, root);

    Json::Value actfuncs = root["activation functions"];
    Json::Value learningRate = root["learningRate"];
    Json::Value optimizer = root["adam"];
    Json::Value topology = root["topology"];
    Json::Value weights = root["weights"];
    Json::Value biases = root["biases"];

    this->activationFunctions = vector<string>();
    this->topology = vector<uint>();

    if(actfuncs.size() > 0)
    {
        for(Json::ValueIterator itr = actfuncs.begin(); itr != actfuncs.end(); itr++)
        {
            this->activationFunctions.push_back(itr->asString());
        }
    }

    this->learningRate = learningRate.asDouble();
    this->optimizer = optimizer.asString();
    if(topology.size() > 0)
    {
        for(Json::ValueIterator itr = topology.begin(); itr != topology.end(); itr++)
        {
            this->topology.push_back(itr->asUInt());
        }
    }

    this->layers = vector<Layer>();
    string layerType = "Hidden";
    for(uint i = 0; i < topology.size(); i++)
    {
        if(i == 0) layerType = "Input";
        else if(i == topology.size()-1) layerType = "Output";
        Layer layer(this->activationFunctions[i], layerType, this->topology[i], this->topology[i+1], this->learningRate);
        this->layers.push_back(layer);
        layerType = "Hidden";
    }

    uint i = 0, j = 0, z = 0;
    if(weights.size() > 0)
    {
        for(Json::ValueIterator itr = weights.begin(); itr != weights.end(); itr++)
        {
            this->layers[i].neurons[j].weights[z].setWeight(itr->asDouble());
            z++;
            if(z > this->layers[i].neurons[j].weights.size()-1)
            {
                j++;
                z=0;
                if(j > this->layers[i].neurons.size()-1)
                {
                    i++;
                    j = 0;
                }
            }
        }
    }
    i = 1;
    if(biases.size() > 0)
    {
        for(Json::ValueIterator itr = biases.begin(); itr != biases.end(); itr++)
        {
            this->layers[i].bias.setBias(itr->asDouble());
            i++;
        }
    }
}