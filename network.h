#pragma once

#include<iostream>
#include<string>
#include<vector>
#include"neuron.h"

class Network {
public:
	std::vector<std::vector<Neuron*> > layers;
	double time, clock, minDelay, maxDelay;
	Neuron *output;

	Network();
	void setOutput(Neuron *n);
	void addLayer();
	void addNeuron(int layer, Neuron *n);
	void addNeurons(int layer, int amount, bool inhibitory = false);
	void addSynapse(Neuron *origin, Neuron *destiny, double weight, double delay);
	bool tick();
	void reset();
	double run(double time);
	void inject(std::vector<double>);
	void changeWeights(int events);
	void finish();
	double test(std::vector<std::vector<double> > testInjects, std::vector<std::vector<double> > testGoals, double time);
	void train(std::vector<std::vector<double> > trainInjects, std::vector<std::vector<double> > trainGoals, double time, int iterations, bool batch = true, bool safeStop = false);
	void trainIteration(std::vector<std::vector<double> > trainInjects, std::vector<std::vector<double> > trainGoals, double time, bool batch);
};