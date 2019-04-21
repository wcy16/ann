#pragma once
#include <vector>
#include <random>

template <typename Layer>
class FullyConnectedLayer
{
    Layer* input_layer;
    double(*activation)(double);
    double(*dactivation)(double);
    std::vector<double> data;
    std::vector<double> bias;
    std::vector<std::vector<double>> weight;
    int input_number, output_number;
    double step;
public:
    FullyConnectedLayer(Layer* layer, 
        int unit_number, double step,
        double(*activation)(double),
        double(*dactivation)(double))
    { 
        input_layer = layer; 
        this->step = step;
        this->activation = activation;
        this->dactivation = dactivation;

        input_number = input_layer->unit();
        output_number = unit_number;

        weight = std::vector<std::vector<double>>(input_number,
            std::vector<double>(output_number, 0));
        bias = std::vector<double>(output_number, 0);
        data = std::vector<double>(output_number, 0);

        // give random weight
        std::default_random_engine engine;
        std::normal_distribution<double> dist(0, 0.05);
        for (auto v : weight)
            for (double& d : v)
                d = dist(generator);
        for (double& d: bias)
            d = dist(generator);;
    }

    //FullyConnectedLayer() {}

    // return how many unit this layer contains
    int unit()
    {
        return data.size();
    }

    std::vector<double> get_data() const
    {
        return data;
    }

    std::vector<double> feed_forward()
    {
        std::vector<double> input = input_layer->feed_forward();

        for (int i = 0; i != data.size(); i++)
        {
            data[i] = 0;
            for (int j = 0; j != input.size(); j++)
                data[i] += input[j] * weight[j][i];
            // activate
            data[i] = activation(data[i] + bias[i]);
        }
        return data;
    }

    void back_propagation(std::vector<double> delta)
    {
        auto weight_ = weight;
        auto delta_ = std::vector<double>(input_number, 0);
        auto input_data = input_layer->get_data();
        for (int k = 0; k != input_number; k++)
        {
            for (int j = 0; j != output_number; j++)
            {
                double gradient = delta[j] * dactivation(data[j]);
                // update weights
                weight_[k][j] -= step * gradient * input_data[k];
                // update bias
                bias[j] -= step * gradient;
                // update delta
                delta_[k] += gradient * weight[k][j];
            }

        }
        weight = weight_;
        // propagate
        input_layer->back_propagation(delta_);
    }
    
};

class InputLayer
{
    std::vector<double> data;
    int unit_number;
public:
    InputLayer() {}

    InputLayer(int unit_number) 
    {
        this->unit_number = unit_number;
    }

    void input(std::vector<double> data)
    {
        this->data = data;
    }

    int unit()
    {
        return unit_number;
    }

    std::vector<double> get_data() const
    {
        return data;
    }

    std::vector<double> feed_forward() const
    {
        return data;
    }

    void back_propagation(std::vector<double> delta)
    {
        return;
    }
};

template<typename Layer>
FullyConnectedLayer<Layer> add_fully_connected_layer(
    Layer* layer,
    int unit_number, double step,
    double(*activation)(double),
    double(*dactivation)(double))
{
    return FullyConnectedLayer<Layer>(layer,
        unit_number, step, activation, dactivation);
}
