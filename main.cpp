#include "network.h"
#include<vector>


using namespace std;

void main(int argc, char *argv[]){
	Network nw = Network();

	nw.addLayer();
	nw.addNeurons(0, 4);

	nw.addLayer();
	nw.addNeurons(1, 1);
	//nw.addNeurons(1, 1, true);

	//nw.addLayer();
	//nw.addNeurons(2, 1);

	nw.finish();


	std::vector<std::vector<double> > injects = readFile(argv[1]);
	std::vector<std::vector<double> > goals = readFile(argv[2]);

	std::vector<std::vector<double> > vInjects = readFile(argv[3]);
	std::vector<std::vector<double> > vGoals = readFile(argv[4]);

	cout << "Non-trained: " << nw.test(injects, goals, 16.5)[1] << endl;

	nw.train(injects, goals, vInjects, vGoals, 17, 80, false, true);

	cout << "Trained: " << nw.test(injects, goals, 16.5)[1] << endl;

	injects = readFile(argv[5]);
	goals = readFile(argv[6]);

	cout << "Test: " << nw.test(injects, goals, 16.5)[1] << endl;

	std::cin.get();
}