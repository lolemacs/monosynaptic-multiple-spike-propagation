#pragma once

#include<iostream>
#include<string>
#include<vector>
#include"neuron.h"

class Network {
public:
	std::vector<std::vector<Neuron*> > layers;
	double time, clock;
	Neuron *output;

	Network();
	void setOutput(Neuron *n);
	void addLayer();
	void addNeuron(int layer, Neuron *n);
	void addSynapse(Neuron *origin, Neuron *destiny, double weight, double delay);
	bool tick();
	void reset();
	double run(double time);
	void inject(std::vector<double>);
	double test(std::vector<std::vector<double> > injects, std::vector<std::vector<double> > goals, double time);
};