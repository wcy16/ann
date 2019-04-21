#include "layer.h"
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

using namespace std;

normal_distribution<double> dist(0, 0.1);
std::default_random_engine generator;

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivation(double x) { return x * (1.0 - x); }

double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double x) { return x > 0 ? 1 : 0; }

double y(double x) { return x; }
double dy(double x) { return 1; }

double tanh_derivative(double x) { return 1.0 - x * x; }

double func(double x)
{
    /*double ret = 0;
    for (int i = 0; i != 2; i++)
        ret += pow(x, i + 1);
    return ret;*/
    return pow(x, 2); // + dist(generator);
    //return sin(x); // + dist(generator);
}

int main()
{
    InputLayer input_layer(1);
    FullyConnectedLayer<InputLayer> hidden_layer(&input_layer, 5, 0.01, sigmoid, sigmoid_derivation);
    FullyConnectedLayer<FullyConnectedLayer<InputLayer>> 
        output_layer(&hidden_layer, 1, 0.1, y, dy);

    vector<vector<double>> input(30, vector<double>(1, 0));
    vector<vector<double>> output(30, vector<double>(1, 0));

    for (int i = 0; i != 30; i++) {
        double pos = 1.0 * i / 30 * 6.28 - 3.14;
        for (int j = 0; j != 1; j++)
            input[i][j] = pow(pos, (j + 1));
        output[i][0] = func(pos);
    }

    for (int iter = 0; iter != 1000; iter++)
    {
        for (int i = 0; i != input.size(); i++)
        {
            input_layer.input(input[i]);
            auto data = output_layer.feed_forward();
            std::vector<double> delta;
            delta.push_back(data[0] - output[i][0]);
            output_layer.back_propagation(delta);

            if (i == 0)
            {
                double loss = 0.5 * (data[0] - output[i][0]) * (data[0] - output[i][0]);
                cout << iter << ": " << loss << endl;
            }
        }
    }

    for (auto predict : input) {
        input_layer.input(predict);
        vector<double> pred = output_layer.feed_forward();
        double desired = func(predict[0]);
        /*cout << "predicted: " << pred[0] << endl;
        cout << "desired: " << desired << endl;*/
        cout << predict[0] << ", " << pred[0] << ", " << desired << endl;
    }

    system("pause");

}

