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
		error += pow((*goal)[0] - run(time), 2);
		goal++;
	}
	return error / i;
}

void Network::changeWeights(int events){
	for (auto& l : layers){
		for (auto& n : l){
			for (auto& s : n->synapses){
				s.weight += s.weightGrad / events;
				s.weightGrad = 0;
			}
		}
	}
}

void Network::train(std::vector<std::vector<double> > trainInjects, std::vector<std::vector<double> > trainGoals, double time, int iterations, bool batch, bool safeStop){
	int events = trainGoals.size();

	double error = 999;
	double cError;

	for (int i = 0; i < iterations; i++){

		if (safeStop){
			cError = test(trainInjects, trainGoals, 16);
			std::cout << "Error: " << cError << std::endl;
			if (cError <= error)
				error = cError;
			else{
				return;
			}
		}

		std::cout << "Epoch : " << i << std::endl;

		trainIteration(trainInjects, trainGoals, time, batch);

		if (batch)
			changeWeights(events);
	}
}

void Network::trainIteration(std::vector<std::vector<double> > trainInjects, std::vector<std::vector<double> > trainGoals, double time, bool batch){
	double desired, actual;
	double f1, f2, f3, interF1;

	double rate = 0.1;

	std::vector<std::vector<double> >::iterator goal = trainGoals.begin();
	for (std::vector<std::vector<double> >::iterator injects = trainInjects.begin(); injects != trainInjects.end(); injects++){
		reset();
		inject(*injects);

		desired = (*goal)[0];
		actual = run(time);

		for (std::vector<std::vector<Neuron*> >::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); ++layer){
			std::vector<Neuron*> h = *layers.rbegin();
			
			int a = 1;

			if (layer == layers.rbegin()){
				//std::cout << "Output layer :" << (*layer)[0]->label << std::endl;
				for (auto& output : *layer){
					f1 = actual - desired;

					f2 = 0;
					for (auto& n : *(layer + 1)){
						for (auto& t : n->refractions){
							for (auto& s : n->synapses)
								f2 += s.weight * dDecay(output->refractions[0] - t - s.delay);
						}
					}

					if (f2 < 0.1)
						f2 = 0.1;

					output->delta = f1 / f2;

					for (auto& n : *(layer + 1)){
						for (auto& s : n->synapses){
							f3 = decay(output->refractions[0] - n->refractions[0] - s.delay);
							s.weightGrad += rate * output->delta * f3;
						}
					}
				}
			}
			else if (layer != layers.rend() - 1){
				//std::cout << "Hidden layer :" << (*layer)[0]->label << std::endl;
				for (auto& output : *layer){
					f1 = 0;
					for (auto& n : *(layer - 1)){
						interF1 = 0;
						for (auto& t : n->refractions){
							for (auto& s : n->synapses)
								interF1 += s.weight * dDecay(desired - t - s.delay);
						}
						f1 += n->delta * interF1;
					}

					f2 = 0;
					for (auto& n : *(layer + 1)){
						for (auto& t : n->refractions){
							for (auto& s : n->synapses)
								f2 += s.weight * dDecay(desired - t - s.delay);
						}
					}

					if (f2 < 0.1)
						f2 = 0.1;

					output->delta = f1 / f2;

					for (auto& n : *(layer + 1)){
						for (auto& s : n->synapses){
							f3 = decay(desired - n->refractions[0] - s.delay);
							s.weightGrad += -rate * output->delta * f3;
						}
					}
				}
			}
		}

		if (!batch)
			changeWeights(1);
		goal++;
	}
}