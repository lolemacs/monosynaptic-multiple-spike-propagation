#include "network.h"
#include<vector>


using namespace std;

void main(){
	Network nw = Network();

	nw.addLayer();
	nw.addNeurons(0, 4);

	nw.addLayer();
	nw.addNeurons(1, 1);
	//nw.addNeurons(1, 1, true);

	//nw.addLayer();
	//nw.addNeurons(2, 1);

	nw.finish();


	std::vector<std::vector<double> > injects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\1\\trainX");
	std::vector<std::vector<double> > goals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\1\\trainY");

	std::vector<std::vector<double> > vInjects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\1\\validX");
	std::vector<std::vector<double> > vGoals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\1\\validY");

	cout << "Non-trained: " << nw.test(injects, goals, 16.5)[1] << endl;

	nw.train(injects, goals, vInjects, vGoals, 17, 80, false, true);

	cout << "Trained: " << nw.test(injects, goals, 16.5)[1] << endl;

	injects = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\1\\testX");
	goals = readFile("C:\\Users\\Pedro\\Documents\\Faculdade\\ProjFin\\iris\\1\\testY");

	cout << "Test: " << nw.test(injects, goals, 16.5)[1] << endl;

	std::cin.get();
}