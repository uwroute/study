/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: Tue 07 Apr 2015 05:27:33 PM CST [10.146.36.174]
#
# Filename: owlqn.h
#
# Description: 
#
=============================================================================*/
#ifndef _OWLQN_H_
#define _OWLQN_H_

#include <vector>
#include <string>
#include <iostream>
#include "model/model.h"
#include "data/data.h"

namespace ML
{

using std::vector;

class OWLQN
{
public:
    OWLQN():_steepest_dir(_next_grad){}
    ~OWLQN(){}
public:
    void optimize();
    void set_data(DataSet* data) {_data=data;}
    void set_model(Model* model) {_model=model;}
    void set_l1(double l1) {_l1=l1;}
    void set_max_iter(int iter) {_max_iter=iter;}
    void set_m(int m) {_M = m;}
    void set_error(double e) {_error=e;}
    void set_dim(int dim) {_N = dim;}
    void init();
    int caluc_space();
private:
    void makeSteepestDescDir(); // -grad dir
	void mapDirByInverseHessian(); // -Hk+1*grad
	void fixDirSign();  // -Hk+1*grad should be same dir with -grad
	void updateDir();	// compute _dir
	double checkDir();  // _dir should be desc dir
    void linearSearch();
    void shiftState();
	double l1Loss();
    void getNextPoint(double alpha);
    bool checkEnd();
private:
    double dotProduct(vector<double>& x, vector<double>& y);
	void add(vector<double>& out, const vector<double>& in);
	void addScale(vector<double>& out, const vector<double>& x, const double scale);
	void addScaleInto(vector<double>& out, const vector<double>& x, const vector<double>& y, const double scale);
	void scale(vector<double>& out, const double scale);
	void scaleInto(vector<double>& out, const vector<double>& x, const double scale);
private:
    // current param
    size_t _N;
    vector<double> _w;
    vector<double> _next_w;
    // current dir
    vector<double> _dir;
    vector<double> _grad;
    vector<double> _next_grad;
    vector<double>& _steepest_dir;
    double _loss;
    // S,Y in LBFGS
    vector<double> _alpha;
    double _beta;
    vector<double> _sy;
    int _M;
    vector<vector<double> > _Y;
    vector<vector<double> > _S;
    int _start;
    int _end;
    // Model : such as logistics, linear
    Model* _model;
    DataSet* _data;
    // overfit
    double _l1;
    // train param
    int _max_iter;
    int _cur_iter;
    double _error;
};

}

#endif
