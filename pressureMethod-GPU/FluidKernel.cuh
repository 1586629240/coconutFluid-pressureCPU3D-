#include "Fluid.cuh"

#include <thrust/transform.h>
#include <device_launch_parameters.h>

#pragma once

void integrateCuImpl(double dt, double gravity, int nx, int ny, v1d& v);

__global__ void advectVelCuImpl(
	double dt, int nx, int ny, double dxy, int threadCnt, 
	double* barrier, double* u, double* v, double* newU, double* newV);

__global__ void advectSmokeCuImpl(
	double dt, int nx, int ny, double dxy, int threadCnt,
	double* barrier, double* u, double* v, double* smoke, double* newSmoke);

__global__ void clearToZeroCuImpl(int nx, int ny, int threadCnt, double* arr);

__host__ void GaussSedielRBCu(
	int iter, double dt, int nx, int ny, double pdxydt,
	double* barrier, double* u, double* v, double* pre);

__global__ void setObstacleCuImpl(
	double vx, double vy, int nx, int ny,
	double dxy, int threadCnt, double ballX, double ballY,
	double ballR2, double t, double* barrier, double* u, double* v, double* smoke);

__global__ void initSceneVortexStreetCuImpl(
	int nx, int ny, double dx, double dy, int threadCnt,
	double inVel, double pipeMinH, double pipeMaxH,
	double* barrier, double* smoke, double* u);

void boundryCondPeriodCuImpl(
	int nx, int ny, double* u, double* v);

void ConjugateGradientCu(
	int maxIter, double dt, int nx, int ny, double pdxydt,
	v1d& barrier, v1d& u, v1d& v, v1d& pressure);