/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: Fri 03 Apr 2015 03:25:24 PM CST [10.146.36.174]
#
# Filename: model.h
#
# Description: 
#
=============================================================================*/
#ifndef _MODEL_H_
#define _MODEL_H_

#include <vector>
#include <string>
#include <iostream>
#include "data/data.h"


namespace ML {

using std::vector;

const double MinDoubleValue = 1.0e10;

// virtual interface Model for compute loss and grad
// child class such as logistic and linear model
class Model
{
public:
    // Common interface
    Model(){}
    virtual ~Model(){}
    virtual double predict(const Feature* sample) = 0;
    virtual void save_model(const char* model_file) = 0;
    virtual void load_model(const char* model_file) = 0;
public:
    // special interface for opt
    virtual double predict(const Feature* sample, const vector<double>& w) = 0;
    virtual void grad(const vector<double>& w, const DataSet& data, vector<double>& grad) = 0;
    virtual void loss(const vector<double>& w, const DataSet& data, double& loss) = 0;
    virtual void grad_and_loss(const vector<double>& w, const DataSet& data, vector<double>& grad, double& loss) = 0;
    virtual void set_param(const vector<double>& w) = 0;
};

}

#endif
