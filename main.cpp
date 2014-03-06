#include "network.h"
#include<vector>


using namespace std;

void main(){
	Network nw = Network();

	nw.addLayer();
	nw.addNeurons(0, 4);

	nw.addLayer();
	nw.addNeurons(1, 5);
	//nw.addNeurons(1, 1, true);

	nw.addLayer();
	nw.addNeurons(2, 1);

	nw.finish();

	std::vector<std::vector<double> > injects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\trainX");
	std::vector<std::vector<double> > goals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\trainY");

	cout << "Non-trained: " << nw.test(injects, goals, 16.5) << endl;

	nw.train(injects, goals, 17, 200, false, false);

	cout << "Trained: " << nw.test(injects, goals, 16.5) << endl;

	/*for (auto& l : nw.layers)
	{
		for (auto& n : l){
			cout << "Neuron " << n->label << endl;
			for (auto& s : n->synapses){
				cout << "    Weight: " << s.weight << " - Delay: " << s.delay << endl;

			}
		}
	}*/

	//injects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\testX");
	//goals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\testY");

	//cout << "Test: " << nw.test(injects, goals, 16) << endl;

	std::cin.get();
}