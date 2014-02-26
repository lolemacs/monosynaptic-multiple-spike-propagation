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

	nw.addSynapse(&a, &b, 5, 1);
	nw.addSynapse(&a, &b, 5, 7);

	nw.setOutput(&b);


	std::vector<std::vector<double> > injects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\trainX");
	std::vector<std::vector<double> > goals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\trainY");

	cout << nw.test(injects, goals, 10) << endl;

	std::cin.get();
}