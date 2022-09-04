#include "../../include/data_processing/power_consumption.hpp"

dataset::dataset(uint size, string filepath, string name)
{
    this->filepath = filepath;
    this->size = size;
    this->name = name;

    this->readFile();
    this->normalize();
}

void dataset::readFile()
{
    fstream fin;

    string line, word, temp;

    fin.open(this->filepath, ios::in);
    if(!fin.is_open())
    {
        cout << "File not found\n";
        return;
    }

    vector<string> row;
    uint size = this->size+1;
    while(size > 0)
    {
        row.clear();

        getline(fin, line);

        stringstream s(line);

        while(getline(s, word, ','))
        {
            row.push_back(word);
        }
        if(size == this->size+1) this->column_labels = row;
        else this->unnormalized_data.push_back(row);
        size--;
    }
}

void dataset::normalize()
{
    this->data = vector<vector<double>>(this->unnormalized_data.size(), vector<double>(this->unnormalized_data[0].size()));
    unnorminfo = vector<minMaxFinder>(this->unnormalized_data[0].size()-1);

    for(uint i = 0; i < this->unnormalized_data.size(); i++)
    {
        this->data[i][0] = this->normalizeTime(this->unnormalized_data[i][0]);
    }

    for(uint i = 1; i < this->unnormalized_data[0].size(); i++)
    {
        minMaxFinder mmf;
        for(uint j = 0; j < this->unnormalized_data.size(); j++)
        {
            mmf.isMinOrMax(stof(this->unnormalized_data[j][i]));
        }
        unnorminfo[i-1] = mmf;
        for(uint j = 0; j < this->unnormalized_data.size(); j++)
        {
            this->data[j][i] = this->minMaxNormalization(stof(this->unnormalized_data[j][i]), mmf);
        }
    }
}

double dataset::normalizeTime(string time)
{
    string pt1;
    string pt2;

    uint idx_colon;
    uint idx_space;
    for(uint i = 0; i < time.length(); i++)
    {
        if(time.at(i) == 58) idx_colon = i;
        else if(time.at(i) == 32) idx_space = i;
    }

    pt1 = time.substr(idx_space+1, idx_colon-idx_space-1);
    pt2 = time.substr(idx_colon+1);

    double mins = (stof(pt1) * 60.f) + stof(pt2);
    //cout << mins/1440 << endl;
    return mins/1440;
}

double dataset::minMaxNormalization(double x, minMaxFinder mmf)
{
    return (x - mmf.min) / (mmf.max - mmf.min);
}

double dataset::minMaxUnnormalization(double x, uint column)
{
    return (x * (this->unnorminfo[column].max - this->unnorminfo[column].min) + this->unnorminfo[column].min);
}

minMaxFinder::minMaxFinder()
{
    this->min = numeric_limits<double>::max();
    this->max = numeric_limits<double>::min();
}

void minMaxFinder::isMinOrMax(double x)
{
    if(x < this->min) this->min = x;
    if(x > this->max) this->max = x;
}

void dataset::peak(uint limit)
{
    cout << this->name << " (to row "<<limit<<"):\n{";
    for(uint i = 0; i < limit && i < this->data.size(); i++)
    {
        cout << "{";
        for(uint j = 0; j < this->data[i].size(); j++)
        {
            if(j != this->data[i].size()-1) cout << this->data[i][j] << ", ";
            else cout << this->data[i][j];
        }
        if(i != this->data.size()-1 && i != limit-1) cout << "}," << endl;
        else cout << "}" << endl;
    }
    cout << "}" << endl;
}

vector<dataset> dataset::split(uint splitAt, string name1, string name2)
{
    vector<dataset> datasets;
    dataset one(this->size, this->filepath, name1);
    dataset two(this->size, this->filepath, name2);
    one.data = vector<vector<double>>(splitAt, vector<double>(this->data[0].size()));
    two.data = vector<vector<double>>(this->data.size()-splitAt, vector<double>(this->data[0].size()));
    one.column_labels = this->column_labels;
    two.column_labels = this->column_labels;
    one.unnorminfo = this->unnorminfo;
    two.unnorminfo = this->unnorminfo;
    for(uint i = 0; i < this->data.size(); i++)
    {
        for(uint j = 0; j < this->data[i].size(); j++)
        {
            if(i < splitAt) one.setData(i, j, this->data[i][j]);
            else two.setData(i-splitAt, j, this->data[i][j]);
        }
    }

    datasets.push_back(one);
    datasets.push_back(two);

    return datasets;
}

void dataset::setData(uint i, uint j, double val)
{
    this->data[i][j] = val;
}

string dataset::column_labels_toString()
{
    string s = "";
    for(uint i = 0; i < column_labels.size(); i++)
    {
        s.append(to_string(i));
        s.append(": ");
        s.append(column_labels[i]);
        s.append("\n");
    }
    s.append("\n");
    return s;
}

void dataset::shuffle()
{
    auto rng = default_random_engine {};
    std::shuffle(begin(this->data), end(this->data), rng);
}