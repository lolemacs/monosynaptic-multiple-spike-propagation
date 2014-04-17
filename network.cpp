#include "network.h"

Network::Network(){
	time = 0;
	clock = 0.5;
	minDelay = 1;
	maxDelay = 16;
	terminalOrder = 15;
	numberOutputs = 0;
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

void Network::addNeurons(int layer, int amount, bool inhibitory){
	Neuron* n;
	double w;
	for (int i = 0; i < amount; i++){
		std::string r = std::to_string(layer) + std::to_string(i);
		n = new Neuron(r, inhibitory);
		addNeuron(layer, n);
		if (layer > 0){
			for (auto& m : layers[layer - 1]){
				for (double d = minDelay; d < maxDelay; d += (maxDelay - minDelay) / terminalOrder)
				{
					w = rand() % 600 / 1000.0 - 0.1;
					if (m->inhibitory) w = -w / 2.0;
					addSynapse(m, n, w, d);
				}
			}
		}
	}
}

void Network::finish(){
	numberOutputs = layers.back().size();
	setOutput(layers.back().front());
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


double* Network::run(double t){
	double* output = new double[numberOutputs];
	for (double i = 0; i < t; i += clock)
		tick();

	int i = 0;
	for (auto& neuron : layers.back()){
		if (neuron->refractions.empty()){
			neuron->refractions.push_back(30);
			output[i] = 30;
			//throw 1;
		}
		else
			output[i] = neuron->refractions[0];
		i++;
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

double* Network::test(std::vector<std::vector<double> > testInjects, std::vector<std::vector<double> > testGoals, double time){
	double error = 0;
	double misses = 0;
	double* actual;
	int i = 0;

	std::vector<std::vector<double> >::iterator goal = testGoals.begin();
	for (std::vector<std::vector<double> >::iterator injects = testInjects.begin(); injects != testInjects.end(); injects++){
		reset();
		i++;
		inject(*injects);
		actual = run(time);

		if (goal->size() == 1){
			//std::cout << "Goal: " << (*goal)[0] << "  |   Actual: " << actual[0] << std::endl;
			if ((*goal)[0] == 10 && (pow(actual[0] - 14, 2) <= pow(actual[0] - 10, 2)))
				misses++;
			if ((*goal)[0] == 14 && (pow(actual[0] - 14, 2) >= pow(actual[0] - 10, 2)))
				misses++;
		}

		else{
			if (getArrayIndexMin(actual, numberOutputs) != getVectorIndexMin(*goal))
				misses++;
		}

		for (int j = 0; j < numberOutputs; j++)
			error += pow((*goal)[j] - actual[j], 2);

		goal++;
	}
	misses /= testGoals.size();
	error /= i;
	double* r = new double[2];
	r[0] = error;
	r[1] = misses;
	return r;
}

void Network::changeWeights(int events){
	for (auto& l : layers){
		for (auto& n : l){
			for (auto& s : n->synapses){
				s.weight += s.weightGrad / events;
				/*if (!n->inhibitory && s.weight < 0)
				s.weight = 0;
				else if (n->inhibitory && s.weight > 0)
				s.weight = 0;*/
				s.weightGrad = 0;
			}
		}
	}
}

void Network::changeDelays(int events){
	for (auto& l : layers){
		for (auto& n : l){
			for (auto& s : n->synapses){
				s.delay += s.delayGrad / events;
				s.delayGrad = 0;
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
			cError = test(trainInjects, trainGoals, 16.5)[1];
			std::cout << "Error: " << cError << std::endl;
			if (cError <= error)
				error = cError;
			else{
				//return;
			}
		}

		std::cout << "Epoch : " << i << std::endl;

		trainIteration(trainInjects, trainGoals, time, batch);

		if (batch){
			changeWeights(events);
			changeDelays(events);
		}
	}
}

void Network::trainIteration(std::vector<std::vector<double> > trainInjects, std::vector<std::vector<double> > trainGoals, double time, bool batch){
	double* actual;
	double f1, f2, f3, f3_2;

	double rate = 0.001;
	double rate2 = 0;

	std::vector<std::vector<double> >::iterator goal = trainGoals.begin();
	for (std::vector<std::vector<double> >::iterator injects = trainInjects.begin(); injects != trainInjects.end(); injects++){
		reset();
		inject(*injects);

		actual = run(time);

		for (std::vector<std::vector<Neuron*> >::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); ++layer){
			std::vector<Neuron*> h = *layers.rbegin();

			int a = 1;

			if (layer == layers.rbegin()){

				int j = 0;
				for (auto& output : *layer){

					f1 = (*goal)[j] - actual[j];

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

					j++;

				}

				for (auto& n : *(layer + 1)){
					for (auto& s : n->synapses){
						f3 = decay(s.neuron->refractions[0] - n->refractions[0] - s.delay);
						s.weightGrad += -rate * s.neuron->delta * f3;
						f3_2 = -s.weight * dDecay(s.neuron->refractions[0] - n->refractions[0] - s.delay);
						s.delayGrad += -rate2 * s.neuron->delta * f3_2;
					}
				}
			}

			else if (layer != layers.rend() - 1){
				//std::cout << "Hidden layer :" << (*layer)[0]->label << std::endl;
				for (auto& output : *layer){

					f1 = 0;
					for (auto& s : output->synapses)
						f1 += s.neuron->delta * s.weight * dDecay(s.neuron->refractions[0] - output->refractions[0] - s.delay);

					f2 = 0;
					for (auto& n : *(layer + 1)){
						for (auto& t : n->refractions){
							for (auto& s : n->synapses){
								if (s.neuron == output)
									f2 += s.weight * dDecay(output->refractions[0] - t - s.delay);
							}
						}
					}

					if (f2 < 0.1)
						f2 = 0.1;

					output->delta = f1 / f2;

					for (auto& n : *(layer + 1)){
						for (auto& s : n->synapses){
							if (s.neuron == output){
								f3 = decay(output->refractions[0] - n->refractions[0] - s.delay);
								s.weightGrad += -rate * output->delta * f3;
							}
						}
					}
				}
			}
		}

		if (!batch){
			changeWeights(1);
			changeDelays(1);
		}
		goal++;
	}
}