#pragma once
#include "header.cuh"

class Fluid
{
	int nx, ny;
	double density, dxy;
	v1d pressure, smoke, newSmoke;
	v1d u, v, newU, newV, barrier;

public:

	v1d& getU();
	v1d& getV();
	v1d& getSmoke();
	v1d& getBrrier();
	v1d& getPressure();

	int getNX();
	int getNY();
	double getdxy();

	Fluid(double density, int NX, int NY, double h);

private:
	class Solver
	{
	public:
		static void PCG_GSRB(int myMaxIter, double dt, Fluid& f);
		static void GaussSedielRB(int myMaxIter, double dt, Fluid& f);
		static void ConjugateGradient(int myMaxIter, double dt, Fluid& f);
	};

	void boundryCondPeriod();
	void boundryCondNoSlip();
	void boundryCondDirichlet();

	void advectVel(double dt);
	void advectSmoke(double dt);
	void integrate(double dt, double gravity);

public:
	void simulate(double dt, double gravity, int numIters);
};

struct Scene
{
	Fluid* fluid = nullptr;
	double dt, gravity;
	double ballX = 0, ballY = 0, ballR;
	bool enableMouse = true;
};

extern Scene scene;