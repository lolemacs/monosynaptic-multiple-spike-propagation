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

	Neuron a4 = Neuron("4");
	nw.addNeuron(0, &a4);

	nw.addLayer();
	Neuron b = Neuron("b");
	nw.addNeuron(1, &b);

	int minDelay = 7;
	int maxDelay = 15;

	for (int i = minDelay; i < maxDelay; i++){
		nw.addSynapse(&a1, &b, 5, i);
	}

	for (int i = minDelay; i < maxDelay; i++){
		nw.addSynapse(&a2, &b, 5, i);
	}

	for (int i = minDelay; i < maxDelay; i++){
		nw.addSynapse(&a3, &b, 5, i);
	}

	for (int i = minDelay; i < maxDelay; i++){
		nw.addSynapse(&a4, &b, 5, i);
	}

	nw.setOutput(&b);


	std::vector<std::vector<double> > injects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\trainX");
	std::vector<std::vector<double> > goals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\trainY");

	cout << "Non-trained: " << nw.test(injects, goals, 16) << endl;

	nw.train(injects, goals, 16, 1000);

	cout << "Trained: " << nw.test(injects, goals, 16) << endl;

	for (auto& n : nw.layers[0]){
		cout << "Neuron " << n->label << endl;
		for (auto& s : n->synapses){
			cout << "    Weight: " << s.weight << " - Delay: " << s.delay << endl;

		}
	}

	injects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\testX");
	goals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\testY");

	cout << "Test: " << nw.test(injects, goals, 16) << endl;

	std::cin.get();
}