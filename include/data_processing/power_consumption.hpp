#include <stdlib.h>
#include <vector>
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <random>

using namespace std;

class minMaxFinder
{
    public:
        minMaxFinder();

        void isMinOrMax(float x);

        float min;
        float max;
};

class dataset
{
    public:
        vector<vector<string>> unnormalized_data;
        vector<vector<float>> data;
        vector<minMaxFinder> unnorminfo;
        uint size;
        string filepath;
        string name;
        vector<string> column_labels;

        dataset(uint size, string filepath, string name);

        void readFile();

        void normalize();

        float normalizeTime(string time);

        float minMaxNormalization(float x, minMaxFinder mmf);

        float minMaxUnnormalization(float x, uint column);

        vector<dataset> split(uint splitAt, string name1, string name2);

        void setData(uint i, uint j, float val);

        void peak(uint limit);

        string column_labels_toString();

        void shuffle();
};