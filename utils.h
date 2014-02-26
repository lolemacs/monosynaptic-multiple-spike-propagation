#include<math.h>
#include<fstream>
#include <vector>
#include <sstream>
#include <iterator>

inline double decay(double s){
	if (s <= 0)
		return 0;
	return exp(-s / 4.0) - exp(-s / 2.0);
}

inline double dDecay(double s){
	if (s <= 0)
		return 0;
	return exp(-s / 2.0) / 2.0 - exp(-s / 4.0) / 4.0;
}

inline double refrac(double s){
	if (s <= 0)
		return 0;
	return -exp(-s / 20.0);
}

inline double str2dbl(std::string Text){
	std::stringstream ss(Text);
	double result;
	return ss >> result ? result : 0;
}

inline void printVector(std::vector<double> vector){
	for (auto& i : vector)
		std::cout << i << " ";
	std::cout << std::endl;
}

inline void printMatrix(std::vector<std::vector<double> > matrix){
	for (auto& v : matrix)
		printVector(v);
}

inline std::vector<std::vector<double> > readFile(std::string s){
	std::ifstream myReadFile;
	myReadFile.open(s);

	std::vector<std::vector<double> > sets;
	std::vector<double> set;

	std::string str;

	if (myReadFile.is_open()) {
		while (!myReadFile.eof()) {
			set.clear();
			getline(myReadFile, str);

			std::istringstream buf(str);
			std::istream_iterator<std::string> beg(buf), end;
			std::vector<std::string> tokens(beg, end);

			for (auto& s : tokens)
				set.push_back(str2dbl(s));
			sets.push_back(set);
		}
	}
	myReadFile.close();

	return sets;
}



