#include "logistic_model.h"


double LogisticModel::wx(const Feature* sample, const vector<double>& w)
{
    double wx = 0.0;
    while (sample->index != -1)
    {
        wx += sample->value * w[sample.index];
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
void LogisticModel::grad(const vector<double>& w, DataSet& data, vector<double>& grad);
{

}
void LogisticModel::loss(const vector<double>& w, DataSet& data, double& loss);
{

}
void LogisticModel::grad_and_loss(const vector<double>& w, DataSet& data, vector<double>& grad, double& loss);
{}
double LogisticModel::predict(const Feature* sample)
{
    return predict(sample, _w);
}
void LogisticModel::save_model(const char* model_file)
{}
void LogisticModel::load_model(const char* model_file)
{}
