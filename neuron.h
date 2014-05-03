#pragma once

#include<iostream>
#include<string>
#include<vector>
#include"utils.h"

class Synapse;

class Neuron {
public:
	std::string label;
	double v, vt, delta, bestVt;
	bool inhibitory;
	std::vector<double> refractions;
	std::vector<std::vector<double> > spikes;
	std::vector<Synapse> synapses;

	Neuron(std::string l = "generic", bool inhibitory = false);
	void addSynapse(Neuron *n, double w, double d);
	bool tick(double clock);
	void excite(double clock);
	void inhibit(double clock);
	void spike(double clock);
	void inject(double time);
};

class Synapse {
public:

	double weight, delay, bestWeight, bestDelay;
	Neuron *neuron;
	double weightGrad, delayGrad;
	Synapse(Neuron *n, double w, double d);
};