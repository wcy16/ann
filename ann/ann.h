#pragma once
#include <vector>

class Layer
{
public:
    double(*activation_f)(double);
    double(*derivative)(double);
    std::vector<double> data;
    int size() { return data.size(); }
    void init()
    {
        for (double& d: data)
            d = 1.0 * (rand() % 100) / 100;
    }
    void activate()
    {
        for (double& d : data)
            d = activation_f(d);
    }
};

class Ann
{
    int input;
    int output;
    int depth;
    std::vector<double> desired_result;
    typedef std::vector<std::vector<double>> Weight;
    std::vector<Layer> layers;
    std::vector<Weight> weights;
    std::vector<std::vector<double>> bias;
    // update all the layers
    void feed_forward(const std::vector<double>& input_data);

    void back_propagation(const std::vector<double>& input_data);

    void dot(const std::vector<double> input, const Weight weight, std::vector<double>& output);
    void add_bias(std::vector<double>& layer, const std::vector<double> bias);

    double get_loss();
    double step;
public:
    Ann(int input, std::vector<Layer> layers) : layers(layers), input(input) { depth = layers.size(); }
    void initialize();
    void train(std::vector<std::vector<double>> input_data, std::vector<std::vector<double>> output_data, int iter, double step);
    std::vector<double> predict(std::vector<double>& in);
};
