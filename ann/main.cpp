#include "ann.h"

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
    //return 2 * x; // + dist(generator);
    return sin(x); // + dist(generator);
}

int main()
{
    vector<Layer> layers;

    for (int i = 0; i != 4; i++) {
        Layer hidden;
        hidden.activation_f = tanh;
        hidden.derivative = tanh_derivative;
        hidden.data = vector<double>(5, 0);
        layers.push_back(hidden);
    }

    Layer out;
    out.activation_f = y;
    out.derivative = dy;
    out.data = vector<double>(1, 0);

    layers.push_back(out);

    auto network = Ann(1, layers);
    network.initialize();

    vector<vector<double>> input(30, vector<double>(1, 0));
    vector<vector<double>> output(30, vector<double>(1, 0));

    for (int i = 0; i != 30; i++) {
        double pos = 1.0 * i / 30 * 6.28 - 3.14;
        for (int j = 0; j != 1; j++)
            input[i][j] = pow(pos, (j + 1));
        output[i][0] = func(pos);
    }

    network.train(input, output, 1000, 0.1);
    for (auto predict : input) {
        vector<double> pred = network.predict(predict);
        double desired = func(predict[0]);
        /*cout << "predicted: " << pred[0] << endl;
        cout << "desired: " << desired << endl;*/
        cout << predict[0] << ", " << pred[0] << ", " << desired << endl;
    }

    system("pause");

}

