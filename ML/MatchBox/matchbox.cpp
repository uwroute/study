/*=============================================================================
#
# Author: route - uwnroute@126.com
#
# Last modified: 2015-08-11 20:32
#
# Filename: matchbox.cpp
#
# Description: 
#
=============================================================================*/
#include "adpredictor.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "Common/log.h"

namespace ML {

void MatchBox::divGauss(const Param& p1, const Param& p2, Param& r) {
	r.v = p1.v/(p2.v-p1.v)*p2.v;
	r.m = p1.m*(r.v/p1.v) - p2.m*(r.v/p2.v);
}
void MatchBox::multGauss(const Param& p1, const Param& p2, Param& r) {
	r.v = p1.v/(p2.v+p1.v)*p2.v;
	r.m = p1.m*(r.v/p1.v) + p2.m*(r.v/p2.v);
}
void MatchBox::addGauss(const Param& p1, const Param& p2, Param& r) {
	r.v = p1.v + p2.v;
	r.m = p1.m + p2.m;
}
void MatchBox::decGauss(const Param& p1, const Param& p2, Param& r) {
	r.v = p1.v + p2.v;
	r.m = p1.m - p2.m;
}
void truncatedGauss(const Param& p, double a, double b, Param& r) {

}

void MatchBox::train(const LongFeature* sample, double label) {

}

double MatchBox::predict(const LongFeature* sample) {
	return 0.0;
}

void MatchBox::save_model(const std::string& file) {

}

void AdPredictor::load_model(const std::string& file) {

}

}