/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 20:32
#
# Filename: matchbox.h
#
# Description: 
#
=============================================================================*/
#ifndef _MATCHBOX_H_
#define _MATCHBOX_H_

#include <map>
// #include <tr1/unordered_map>
#include <string>
#include <stdint.h>
#include <unordered_map>
#include "data/data.h"

namespace ML {
class MatchBox {
public:
	struct Param
	{
	public:
		double m;
		double v;
	public:
		Param() :m(0.0), v(0.0) {}
		Param(double tm, double tv) : m(tm), v(tv) {}
		~Param() {}
		void reset() {m=0.0; v=0.0;}
	};
	typedef std::unordered_map<uint64_t, Param> ParamMap;
public:
	MatchBox() : _k(0), _beta(1.0), _w_prior(0.0, 1.0), _bias(0.0, 1.0) {}
	~MatchBox() {}
	void init(int k, std::string prior_m, std::string prior_v, double _beta, int max_fea = 1000*1000);
	void train(const LongMatrixFeature* sample, double label);
	double predict(const LongMatrixFeature* sample);
	void save_model(const std::string& file);
	void load_model(const std::string& file);
	double cumulative_probability(double  t, double mean=0.0, double variance=1.0);
	double gauss_probability(double t, double mean=0.0, double variance=1.0);
private:
	Param divGauss(const Param& p1, const Param& p2);
	Param multGauss(const Param& p1, const Param& p2);
	Param addGauss(const Param& p1, const Param& p2);
	Param decGauss(const Param& p1, const Param& p2);
	Param truncatedGauss(const Param& p, double label);
	Param truncatedGauss(const Param& p, double a, double b);
	Param get_user_param(uint64_t idx, int k);
	Param get_item_param(uint64_t idx, int k);
	Param get_w_param(uint64_t idx);
	void set_user_param(uint64_t idx, int k, Param& p) {_user[k][idx] = p;}
	void set_item_param(uint64_t idx, int k, Param& p) {_item[k][idx] = p;}
	void set_w_param(uint64_t idx, Param& p) {_w[idx] = p;}
	double square(double x) {return x*x;}
	double second_moment(Param p) {return p.v + square(p.m);}
	void clear_state();
	void saveMap(std::ostream& out_file, ParamMap& map);
	void loadMap(std::istream& infile, ParamMap& map);
private:
	int _k;
	double _beta;
	// train param
	std::vector<Param> _user_prior;
	std::vector<Param> _item_prior;
	Param _w_prior;
	std::vector<Param> _s;
	std::vector<Param> _t;
	std::vector<Param> _z;
	Param _b;
	Param _r;
	// model param
	std::vector<ParamMap> _user;
	std::vector<ParamMap> _item;
	ParamMap _w;
	Param _bias;
};

}

#endif  // _MATCHBOX_H_
