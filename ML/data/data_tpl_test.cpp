#include "data_tpl.hpp"
#include <iostream>

uint32_t log_level = 0;

int main(int argc, char* argv[])
{
    typedef ML::Feature<int, float> Feature;
    ML::DataSet<Feature> data;
    char* infile = argv[1];
    data.load(infile);
    return 0;
}
