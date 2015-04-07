/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: Tue 07 Apr 2015 05:30:52 PM CST [10.146.36.174]
#
# Filename: logistic_model.h
#
# Description: 
#
=============================================================================*/
#ifndef _LOGISTIC_MODEL_H_
#define _LOGISTIC_MODEL_H_

#include "model.h"
#include "data/data.h"

namespace ML
{

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
    void grad(const vector<double>& w, const DataSet& data, vector<double>& grad);
    void loss(const vector<double>& w, const DataSet& data, double& loss);
    void grad_and_loss(const vector<double>& w, const DataSet& data, vector<double>& grad, double& loss);
    void set_param(vector<double>& w)
    {
        _w.swap(w);
        // _w.resize(w.size());
        // for (int i=0; i<w.size(); ++i)
        // {
        //    _w[i] = w[i];
        // }
    }
    void set_l2(const double l2)
    {
        _l2 = l2;
    }
    void set_dim(const int dim)
    {
        _w.resize(dim);
    }
private:
    double wx(const Feature* sample, const vector<double>& w);
    double log_loss(const Feature* sample, const vector<double>& w, double label);
private:
    // model param
    vector<double> _w;
    double _l2;
};

}
#endif
