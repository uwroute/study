#include <vector>
#include <string>
#include <iostream>

using std::vector;

const double MinDoubleValue = 1.0e10;

class Model
{
public:
    Model(){}
    ~Model(){}
    virtual void grad(const vector<double>& w, vector<double>& grad);
    virtual void loss(const vector<double>& w, double& loss);
    virtual void grad_and_loss(const vector<double>& w, vector<double>& grad, double& loss);
};

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
    int _M;
    vector<vector<double> > _Y;
    vector<vector<double> > _S;
    int _start;
    int _end;
    // Model : such as logistics, linear
    Model* _model;
    // overfit
    double _l2;
    double _l1;
    // train param
    int _max_iter;
    double _error;
};

