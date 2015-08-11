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
	typedef std::unordered_map<uint64_t, double> DoubleHashMap;
	struct ParamMap {
		DoubleHashMap mMap;
		DoubleHashMap vMap;
	};
	struct Param
	{
		double m;
		double v;
	};
public:
	void train(const LongFeature* sample, double label);
    double predict(const LongFeature* sample);
    void save_model(const std::string& file);
    void load_model(const std::string& file);
private:
	void divGauss(const Param& p1, const Param& p2, Param& r);
	void multGauss(const Param& p1, const Param& p2, Param& r);
	void addGauss(const Param& p1, const Param& p2, Param& r);
	void decGauss(const Param& p1, const Param& p2, Param& r);
	void truncatedGauss(const Param& p, double a, double b, Param& r);
private:
	// model param
	int _k;
	std::vector<ParamMap> _user;
	std::vector<ParamMap> _item;
	ParamMap _w;
	// train param
	std::vector<Param> _prior;
	double _beta;
	std::vector<Param> _s;
	std::vector<Param> _t;
	std::vector<Param> _z;
	double _b;
	double _r;
};

}

#endif  // _MATCHBOX_H_
