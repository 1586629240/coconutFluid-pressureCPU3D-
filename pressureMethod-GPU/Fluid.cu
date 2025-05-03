#include "Fluid.cuh"
#include "FluidKernel.cuh"

Scene scene;

v1d& Fluid::getU() { return u; }

v1d& Fluid::getV() { return v; }

v1d& Fluid::getSmoke() { return smoke; }

v1d& Fluid::getBrrier() { return barrier; }

v1d& Fluid::getPressure() { return pressure; }

int Fluid::getNX() { return nx; }

int Fluid::getNY() { return ny; }

double Fluid::getdxy() { return dxy; }

Fluid::Fluid(double density, int NX, int NY, double h)
{
	this->dxy = h;
	this->density = density;
	nx = NX + 2, ny = NY + 2;

	smoke = newSmoke = v1d(ny * nx);
	u = v = newU = newV = barrier = pressure = v1d(ny * nx);
}

void Fluid::integrate(double dt, double gravity)
{
	integrateCuImpl(dt, gravity, nx, ny, v);
}

void Fluid::advectVel(double dt)
{
	newU = u, newV = v;
	advectVelCuImpl << <256, 64 >> > (
		dt, nx, ny, dxy, 256 * 64,
		thrust::raw_pointer_cast(barrier.data()),
		thrust::raw_pointer_cast(u.data()),
		thrust::raw_pointer_cast(v.data()),
		thrust::raw_pointer_cast(newU.data()),
		thrust::raw_pointer_cast(newV.data()));
	cudaDeviceSynchronize();
	u = newU, v = newV;
}

void Fluid::advectSmoke(double dt)
{
	newSmoke = smoke;
	advectSmokeCuImpl << <256, 64 >> > (
		dt, nx, ny, dxy, 256 * 64,
		thrust::raw_pointer_cast(barrier.data()),
		thrust::raw_pointer_cast(u.data()),
		thrust::raw_pointer_cast(v.data()),
		thrust::raw_pointer_cast(smoke.data()),
		thrust::raw_pointer_cast(newSmoke.data()));
	cudaDeviceSynchronize();
	smoke = newSmoke;
}

void Fluid::simulate(double dt, double gravity, int numIters)
{
	integrate(dt, gravity);

	clearToZeroCuImpl << <32, 128 >> > (
		nx, ny, 32 * 128,
		thrust::raw_pointer_cast(pressure.data()));
	cudaDeviceSynchronize();

	Solver::PCG_GSRB(numIters, dt, *this);

	boundryCondPeriod();
	advectVel(dt);
	advectSmoke(dt);
}

void Fluid::Solver::GaussSedielRB(int myMaxIter, double dt, Fluid& f)
{
	const double pdxydt = f.density * f.dxy / dt * 1.9;
	GaussSedielRBCu(myMaxIter, dt, f.nx, f.ny, pdxydt,
		thrust::raw_pointer_cast(f.barrier.data()),
		thrust::raw_pointer_cast(f.u.data()),
		thrust::raw_pointer_cast(f.v.data()),
		thrust::raw_pointer_cast(f.pressure.data()));
}

void Fluid::Solver::ConjugateGradient(int myMaxIter, double dt, Fluid& f)
{
	ConjugateGradientCu(myMaxIter, dt,
		f.nx, f.ny, f.density * f.dxy / dt,
		f.barrier, f.u, f.v, f.pressure);
}


void Fluid::Solver::PCG_GSRB(int myMaxIter, double dt, Fluid& f)
{
    GaussSedielRB(myMaxIter * 0.2, dt, f);
    ConjugateGradient(myMaxIter * 0.8, dt, f);
}