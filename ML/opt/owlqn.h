#include <vector>
#include <string>
#include <iostream>
#include "model.h"

using std::vector;

const double MinDoubleValue = 1.0e10;

class OWLQN
{
public:
    OWLQN(){}
    ~OWLQN(){}
public:
    void optimize();
private:
    void grad();
    void dir();
    void linearSearch();
    void shift();
    double dot(vector<double>& x, vector<double>& y);
private:
    // current param
    int _N;
    vector<double> _w;
    vector<double> _next_w;
    // current dir
    vector<double> _dir;
    vector<double> _grad;
    vector<double> _next_grad;
    double _loss;
    // S,Y in LBFGS
    vector<double> _alpha;
    vector<double> _beta;
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
    double _l2;
    double _l1;
    // train param
    int _max_iter;
    int _cur_iter;
    double _error;
};

