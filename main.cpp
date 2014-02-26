#include "network.h"
#include<vector>


using namespace std;

void main(){
	Network nw = Network();
	nw.addLayer();
	Neuron a = Neuron("a");
	nw.addNeuron(0, &a);

	nw.addLayer();
	Neuron b = Neuron("b");
	nw.addNeuron(1, &b);


	for (int i = 0; i < 16; i++){
		nw.addSynapse(&a, &b, 5, i);
	}

	nw.setOutput(&b);


	std::vector<std::vector<double> > injects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\trainX");
	std::vector<std::vector<double> > goals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\trainY");

	nw.train(injects, goals, 16, 100);

	//cout << nw.test(injects, goals, 16) << endl;

	for (auto& s : nw.layers[0][0]->synapses){
		cout << "Weight: " << s.weight << " - Delay: " << s.delay << endl;

	}

	std::cin.get();
}