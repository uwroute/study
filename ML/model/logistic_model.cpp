#include "logistic_model.h"
#include <cmath>
#include <fstream>

namespace ML {

using std::ifstream;
using std::ofstream;

double LogisticModel::wx(const Feature* sample, const vector<double>& w)
{
    double wx = 0.0;
    while (sample->index != -1)
    {
        wx += sample->value * w[sample->index];
    }
    return wx;
}
double LogisticModel::predict(const Feature* sample, const vector<double>& w)
{
    double value = wx(sample, w);
    if (value > 30.0)
    {
        return 1.0;
    }
    else if (value < -30.0)
    {
        return 0.0;
    }
    else
    {
        return 1.0/(1.0 + exp(-1.0*value));
    }
}
double LogisticModel::log_loss(const Feature* sample, const vector<double>& w, double label)
{
    double value = -1.0*label*wx(sample, w);
    if (value < -30.0)
    {
        return 0.0;
    }
    else if (value > 30.0)
    {
        return value;
    }
    else
    {
        return log(1.0 + exp(value));
    }
}
void LogisticModel::grad(const vector<double>& w, const DataSet& data, vector<double>& grads)
{
    for (size_t i=0; i<data.sample_num; ++i)
    {
        const Feature* sample = &(data.samples[data.sample_idx[i]]);
        double h = predict(sample, w);
        double y = data.labels[data.sample_idx[i]];
        if (y < 0.5)
        {
            y = 0.0;
        }
        double g = h - y;
        while (sample->index != -1)
        {
            grads[sample->index] += g*sample->value;
            sample++;
        }
    }
    if (_l2 > MinDoubleValue)
    {
        for (int i=0; i<grads.size(); ++i)
        {
            grads[i] += _l2*w[i];
        }
    }
}
void LogisticModel::loss(const vector<double>& w, const DataSet& data, double& loss)
{
    loss = 0.0;
    for (size_t i=0; i<data.sample_num; ++i)
    {
        loss += log_loss(&(data.samples[0]) + data.sample_idx[i], _w, data.labels[i]);
    }
    if (_l2 > MinDoubleValue)
    {
        for (int i=0; i<w.size(); ++i)
        {
            loss += 0.5*_l2*w[i]*w[i];
        }
    } 
}
void LogisticModel::grad_and_loss(const vector<double>& w, const DataSet& data, vector<double>& grad, double& loss)
{
    loss = 0.0;
    for (size_t i=0; i<data.sample_num; ++i)
    {
        loss += log_loss(&(data.samples[0]) + data.sample_idx[i], _w, data.labels[i]);
        for (size_t i=0; i<data.sample_num; ++i)
        {
            const Feature* sample = &(data.samples[data.sample_idx[i]]);
            double h = predict(sample, w);
            double y = data.labels[data.sample_idx[i]];
            if (y < 0.5)
            {
                y = 0.0;
            }
            double g = h - y;
            while (sample->index != -1)
            {
                grad[sample->index] += g*sample->value;
                sample++;
            }
        }
    }
    if (_l2 > MinDoubleValue)
    {
        for (int i=0; i<grad.size(); ++i)
        {
            grad[i] += _l2*w[i];
            loss += 0.5*_l2*w[i]*w[i];
        }
    }
}

double LogisticModel::predict(const Feature* sample)
{
    return predict(sample, _w);
}

void LogisticModel::save_model(const char* model_file)
{
    ofstream ofile(model_file);
    for (int i=0; i<_w.size(); ++i)
    {
        ofile << _w[i] << std::endl;
    }
    ofile.close();
}

void LogisticModel::load_model(const char* model_file)
{
    ifstream infile(model_file);
    std::string line;
    getline(infile, line);
    while (!infile.eof())
    {
        _w.push_back(atof(line.c_str()));
        getline(infile, line);
    }
    infile.close();
}

}
