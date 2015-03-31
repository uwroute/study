/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-03-21 10:07
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
    OWLQN(){}
    ~OWLQN(){}
public:
    void optimize();
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
