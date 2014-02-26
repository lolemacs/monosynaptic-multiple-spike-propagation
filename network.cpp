#include "network.h"

Network::Network(){
	time = 0;
	clock = 0.5;
}

void Network::setOutput(Neuron *n){
	output = n;
}

void Network::addLayer(){
	std::vector<Neuron*> layer;
	layers.push_back(layer);
}

void Network::addNeuron(int layer, Neuron *n){
	layers[layer].push_back(n);
}

void Network::addSynapse(Neuron *origin, Neuron *destiny, double weight, double delay){
	origin->addSynapse(destiny, weight, delay);
}

bool Network::tick(){
	time += clock;

	for (std::vector<std::vector<Neuron*> >::iterator i = layers.begin(); i != layers.end(); i++){
		for (std::vector<Neuron*>::iterator j = (*i).begin(); j != (*i).end(); j++){
			if ((*j)->tick(time) && output != NULL){
				if (output->label == (*j)->label)
					return true;
			}
		}
	}
	return false;
}


double Network::run(double t){
	double output = -1;
	for (double i = 0; i < t; i += clock){
		if (tick() && output == -1)
			output = time;
	}
	return output;
}

void Network::inject(std::vector<double> injects){
	std::vector<double>::iterator i = injects.begin();

	for (std::vector<Neuron*>::iterator n = layers[0].begin(); n != layers[0].end(); n++){
		(*n)->inject(*i);
		i++;
	}
}


void Network::reset(){
	time = 0;
	for (std::vector<std::vector<Neuron*> >::iterator i = layers.begin(); i != layers.end(); i++){
		for (std::vector<Neuron*>::iterator j = (*i).begin(); j != (*i).end(); j++){
			(*j)->v = 0;
			(*j)->spikes.clear();
			(*j)->refractions.clear();
		}
	}
}

double Network::test(std::vector<std::vector<double> > testInjects, std::vector<std::vector<double> > testGoals, double time){
	double error = 0;
	int i = 0;

	std::vector<std::vector<double> >::iterator goal = testGoals.begin();
	for (std::vector<std::vector<double> >::iterator injects = testInjects.begin(); injects != testInjects.end(); injects++){
		reset();
		i++;
		inject(*injects);
		error += pow((*goal)[0] - run(time),2);
		goal++;
	}
	return error / i;
}

void Network::train(std::vector<std::vector<double> > trainInjects, std::vector<std::vector<double> > trainGoals, double time, int iterations){
	for (int i = 0; i < iterations; i++){
		std::vector<std::vector<double> >::iterator goal = trainGoals.begin();
		for (std::vector<std::vector<double> >::iterator injects = trainInjects.begin(); injects != trainInjects.end(); injects++){
			reset();
			inject(*injects);

			double desired = (*goal)[0];
			double actual = run(time);
			double f1 = desired - actual;
			//std::cout << f1 << std::endl;

			double f2 = 0;
			for (auto& n : layers[0]){
				for (auto& t : n->refractions){
					for (auto& s : n->synapses)
						f2 += s.weight * dDecay(desired - t - s.delay);
				}
			}
			if (f2 < 0.1)
				f2 = 0.1;

			for (auto& n : layers[0]){
				for (auto& s : n->synapses){
					double f3 = decay(desired - n->refractions[0] - s.delay);
					s.weight += -0.1*(f1 * f3 / f2);
				}
			}

			goal++;
		}
	}
}