#include "neuron.h"


Neuron::Neuron(std::string l, bool inh){
	label = l;
	inhibitory = inh;
	v = 0;
	vt = 1;
	delta = 0;
}

void Neuron::addSynapse(Neuron *n, double w, double d){
	synapses.push_back(Synapse(n, w, d));
}

void Neuron::excite(double clock){
	for (std::vector<std::vector<double> >::const_iterator i = spikes.begin(); i != spikes.end(); i++){
		double w = (*i)[0];
		double t = (*i)[1];
		v += w * decay(clock - t);
	}
}

void Neuron::inhibit(double clock){
	for (std::vector<double>::const_iterator i = refractions.begin(); i != refractions.end(); i++)
		v += refrac(clock - (*i));
}

void Neuron::spike(double clock){
	//std::cout << "Spike! at " << clock << " of neuron " << label << std::endl;
	if (refractions.empty())
		refractions.push_back(clock);
	for (std::vector<Synapse>::iterator i = synapses.begin(); i != synapses.end(); i++){
		Neuron *n = i->neuron;
		std::vector<double> v;
		v.push_back(i->weight);
		v.push_back(clock + i->delay);
		n->spikes.push_back(v);
	}
}

bool Neuron::tick(double clock){
	v = 0;

	if (refractions.empty()){
		excite(clock);
		//inhibit(clock);
		if (v >= vt){
			spike(clock);
			return true;
		}
	}
	return false;
}

void Neuron::inject(double time){
	refractions.push_back(time);
	for (std::vector<Synapse>::iterator i = synapses.begin(); i != synapses.end(); i++){
		Neuron *n = i->neuron;
		std::vector<double> v;
		v.push_back(i->weight);
		v.push_back(time + i->delay);
		n->spikes.push_back(v);
	}
}

Synapse::Synapse(Neuron *n, double w, double d){
	neuron = n;
	weight = w;
	delay = d;
	weightGrad = 0;
	delayGrad = 0;
	bestWeight = 0;
	bestDelay = 0;
}