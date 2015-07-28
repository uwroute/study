#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "gflags/gflags.h"
#include "Common/log.h"

DEFINE_string(res_file, "", "predict res");
DEFINE_int32(log_level, 2, "LogLevel :"
    "0 : TRACE "
    "1 : DEBUG "
    "2 : INFO "
    "3 : ERROR");

uint32_t log_level = 0;

namespace ML {
struct Elem {
	double p;
	double y;
};

bool LessThan(const Elem& a, const Elem& b) {
	return a.p < b.p;
}

bool Equal(const Elem& a, const Elem& b) {
	return fabs(a.p - b.p) < 1.e-10;
}

void check(const Elem& e, int& n, int& p) {
	if (e.y < 0.5) {
		n++;
	}
	else {
		p++;
	}
}

double auc(std::vector<Elem>& res) {
	if (res.size() <= 0)
	{
		return 0.5;
	}
	std::sort(res.begin(), res.end(), LessThan);
	int n = 0;
	int p = 0;
	int cur_n = 0;
	int cur_p = 0;
	double correct = 0;
	check(res[0], cur_n, cur_p);
	for (size_t i=1; i<res.size(); ++i)
	{
		if (Equal(res[i], res[i-1]))
		{
			check(res[i], cur_n, cur_p);
		}
		else
		{
			correct += cur_p*n + 0.5*cur_p*cur_n;
			p += cur_p;
			n += cur_n;
			cur_p = 0;
			cur_n = 0;
			check(res[i], cur_n, cur_p);
		}
	}
	correct += cur_p*n + 0.5*cur_p*cur_n;
	p += cur_p;
	n += cur_n;
	LOG_INFO("POSITIVE : %d , NEGATIVE : %d, CORRECT : %lf", p, n, correct);
	if (n == 0) return 1.0;
	if (p == 0) return 0.0;
	return correct*1.0/(p*n);
}

}

using namespace ML;
int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    log_level = FLAGS_log_level;
    if (FLAGS_res_file.empty())
    {
        std::cout<< "res file is empty!" << std::endl;
        return 0;
    }
    std::ifstream infile(FLAGS_res_file.c_str());
    if (!infile)
    {
        LOG_ERROR("Load res file : %s failed!", FLAGS_res_file.c_str());
        return 0;
    }

    std::string line;
    std::vector<Elem> res;
    Elem e;
    e.p = 0.0;
    e.y = 1.0;
    getline(infile, line);
    while (!infile.eof())
    {
    	sscanf(line.c_str(), "%lf %lf", &(e.y), &(e.p));
    	LOG_TRACE("Elem p : %lf, y : %lf", e.p, e.y);
    	res.push_back(e);
    	getline(infile, line);
    }
    LOG_TRACE("Res size : %lu", res.size());
    printf("[%s]AUC = %lf\n", FLAGS_res_file.c_str(), auc(res));
    return 0;
}
