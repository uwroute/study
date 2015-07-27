#include <iostream>
#include <vector>
#include <algothim>

namespace ML {
struct Elem {
	double p;
	double y;
};

bool LessThan(Elem a, Elem b)
{
	return a.p < b.p;
}

double auc(std::vector<Elem>& res)
{
	std::sort(res, LessThan);
	double auc = 0.0;
	int rank = res.size();
	for (int i=0; i<res.size(); ++i)
	{

	}
}

}

