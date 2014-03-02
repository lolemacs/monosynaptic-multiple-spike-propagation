#pragma once

#include<iostream>
#include<string>
#include<vector>
#include"utils.h"

class Synapse;

class Neuron {
public:
	std::string label;
	double v, vt, delta;
	std::vector<double> refractions;
	std::vector<std::vector<double> > spikes;
	std::vector<Synapse> synapses;

	Neuron();
	Neuron(std::string l);
	void addSynapse(Neuron *n, double w, double d);
	bool tick(double clock);
	void excite(double clock);
	void inhibit(double clock);
	void spike(double clock);
	void inject(double time);
};

class Synapse {
public:

	double weight, delay;
	Neuron *neuron;
	double weightGrad;
	Synapse(Neuron *n, double w, double d);
};