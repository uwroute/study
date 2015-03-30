#include "Model.h"
#include "data/data.h"

class LogisticModel : public Model
{
public:
    // function for predict
    LogisticModel(){}
    ~LogisticModel(){}
    double predict(const Feature* sample);
    void save_model(const char* model_file);
    void load_model(const char* model_file);
public:
    // function for opt
    double predict(const Feature* sample, const vector<double>& w);
    void grad(const vector<double>& w, DataSet& data, vector<double>& grad);
    void loss(const vector<double>& w, Dataset& data, double& loss);
    void grad_and_loss(const vector<double>& w, DataSet& data, vector<double>& grad, double& loss);
    void set_param(const vector<double>& w);
private:
    double wx(const Feature* sample, const vector<double>& w);
    double log_loss(const Feature* sample, double label);
private:
    // model param
    vector<double> _w;
}
