#include "ann.h"
#include <iostream>

void Ann::feed_forward(const std::vector<double>& input_data)
{
    auto last_layer = &input_data;
    for (int i = 0; i != layers.size(); i++)
    {
        dot(*last_layer, weights[i], layers[i].data);
        add_bias(layers[i].data, bias[i]);
        layers[i].activate();
        last_layer = &(layers[i].data);
    }
}

void Ann::back_propagation(const std::vector<double>& input_data)
{
    std::vector<double> delta = std::vector<double>(output, 0);
    for (int i = 0; i != output; i++)
        delta[i] = layers[depth - 1].data[i] - desired_result[i];

    auto weights_ = weights;
    for (int i = layers.size() - 1; i > 0; i--)
    {
        //std::vector<double> gradient = std::vector<double>(output, 0);
        
        auto delta_ = std::vector<double>(layers[i - 1].data.size(), 0);
        for (int k = 0; k != layers[i - 1].data.size(); k++)
        {
            for (int j = 0; j != layers[i].data.size(); j++)
            {
                double gradient = delta[j] * layers[i].derivative(layers[i].data[j]);
                // update weights
                weights_[i][k][j] -= step * gradient * layers[i - 1].data[k];
                // update bias
                bias[i][j] -= step * gradient;
                // update delta
                delta_[k] += gradient * weights[i][k][j];
            }

        }
        // update delta
        delta = delta_;
    }

    // update the weights between input layer and the first hidden layer
    for (int k = 0; k != input_data.size(); k++)
    {
        for (int j = 0; j != layers[0].data.size(); j++)
        {
            double gradient = delta[j] * layers[0].derivative(layers[0].data[j]);
            // update weights
            weights_[0][k][j] -= step * gradient * input_data[k];
            // update bias
            bias[0][j] -= step * gradient;
        }
    }

    // update
    weights = weights_;

}

void Ann::dot(const std::vector<double> input, const Weight weight, std::vector<double>& output)
{
    for (int i = 0; i != output.size(); i++)
    {
        output[i] = 0;
        for (int j = 0; j != input.size(); j++)
            output[i] += input[j] * weight[j][i];
    }
}

void Ann::add_bias(std::vector<double>& layer, const std::vector<double> bias)
{
    for (int i = 0; i != layer.size(); i++)
        layer[i] += bias[i];
}

void Ann::initialize()
{
    int x = input;
    for (int i = 0; i != layers.size(); i++)
    {
        int y = layers[i].size();
        weights.push_back(std::vector<std::vector<double>>(x, std::vector<double>(y, 0)));
        layers[i].init();
        bias.push_back(std::vector<double>(y, 0));
        x = y;
    }

    for (std::vector<double>& b: bias)
        for (double& d: b)
            d = 1.0 * (rand() % 100) / 100;

    for (Weight& w: weights)
        for (std::vector<double>& v: w)
            for (double& d: v)
                d = 1.0 * (rand() % 100) / 100;
}

void Ann::train(std::vector<std::vector<double>> input_data, 
    std::vector<std::vector<double>> output_data, 
    int iter, double step)
{
    this->step = step;
    int data_len = input_data.size();
    output = output_data[0].size();
    
    for (int i = 0; i != iter; i++)   // iterations
    {
        double loss = 0;
        for (int j = 0; j != data_len; j++)
        {
            desired_result = output_data[j];
            feed_forward(input_data[j]);
            //std::cout << get_loss();
            loss += get_loss();
            back_propagation(input_data[j]);
            //feed_forward(input_data[j]);
            //std::cout << " " << get_loss() << std::endl;
        }
        if (i % 50 == 0)
            std::cout << "iteration: " << i << "\t\tloss: " << loss << std::endl;
    }
}

std::vector<double> Ann::predict(std::vector<double>& in)
{
    feed_forward(in);
    return layers[depth - 1].data;
}

double Ann::get_loss()
{
    double loss = 0;
    for (int i = 0; i != output; i++)
    {
        double err = desired_result[i] - layers[depth - 1].data[i];
        loss += err * err;
    }
    return loss / 2;
}
