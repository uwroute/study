#include <vector>
#include <string>
#include <iostream>
#include "data/data.h"

using std::vector;

// virtual interface Model for compute loss and grad
// child class such as logistic and linear model
class Model
{
public:
    // Common interface
    Model(){}
    virtual ~Model(){}
    virtual double predict(const Feature* sample);
    virtual void save_model(const char* model_file);
    virtual void load_model(const char* model_file);
public:
    // special interface for opt
    virtual double predict(const Feature* sample, const vector<double>& w);
    virtual void grad(const vector<double>& w, const DataSet& data, vector<double>& grad) = 0;
    virtual void loss(const vector<double>& w, const DataSet& data, double& loss) = 0;
    virtual void grad_and_loss(const vector<double>& w, const DataSet& data, vector<double>& grad, double& loss) = 0;
};
