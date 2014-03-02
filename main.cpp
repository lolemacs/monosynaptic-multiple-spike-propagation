#include "network.h"
#include<vector>


using namespace std;

void main(){
	Network nw = Network();
	nw.addLayer();

	Neuron a1 = Neuron("a1");
	nw.addNeuron(0, &a1);

	Neuron a2 = Neuron("a2");
	nw.addNeuron(0, &a2);

	Neuron a3 = Neuron("a3");
	nw.addNeuron(0, &a3);

	Neuron a4 = Neuron("a4");
	nw.addNeuron(0, &a4);

	nw.addLayer();
	Neuron b1 = Neuron("b1");
	nw.addNeuron(1, &b1);

	//nw.addLayer();
	//Neuron c1 = Neuron("c1");
	//nw.addNeuron(2, &c1);

	int minDelay = 7;
	int maxDelay = 15;

	for (int i = minDelay; i < maxDelay; i++){
		nw.addSynapse(&a1, &b1, 5, i);
	}

	for (int i = minDelay; i < maxDelay; i++){
		nw.addSynapse(&a2, &b1, 5, i);
	}

	for (int i = minDelay; i < maxDelay; i++){
		nw.addSynapse(&a3, &b1, 5, i);
	}

	for (int i = minDelay; i < maxDelay; i++){
		nw.addSynapse(&a4, &b1, 5, i);
	}

	//for (int i = minDelay; i < maxDelay; i++)
	//	nw.addSynapse(&b1, &c1, 5, i);

	nw.setOutput(&b1);

	std::vector<std::vector<double> > injects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\trainX");
	std::vector<std::vector<double> > goals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\trainY");

	cout << "Non-trained: " << nw.test(injects, goals, 16) << endl;

	nw.train(injects, goals, 16, 999, true, false);

	cout << "Trained: " << nw.test(injects, goals, 16) << endl;

	for (auto& l : nw.layers)
	{
		for (auto& n : l){
			cout << "Neuron " << n->label << endl;
			for (auto& s : n->synapses){
				cout << "    Weight: " << s.weight << " - Delay: " << s.delay << endl;

			}
		}
	}

	//injects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\testX");
	//goals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\testY");

	//cout << "Test: " << nw.test(injects, goals, 16) << endl;

	std::cin.get();
}