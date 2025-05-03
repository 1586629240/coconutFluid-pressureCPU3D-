#include "Fluid.cuh"
#include "FluidKernel.cuh"

void Fluid::boundryCondPeriod()
{
	boundryCondPeriodCuImpl(nx, ny,
		thrust::raw_pointer_cast(u.data()),
		thrust::raw_pointer_cast(v.data()));
}

//void Fluid::boundryCondNoSlip()
//{
//	for (int i = 0; i < ny; i++)
//	{
//		u[i][0] = u[i][1];
//		u[i][nx - 1] = u[i][nx - 2];
//	}
//	for (int i = 0; i < nx; i++)
//	{
//		v[0][i] = v[1][i];
//		v[ny - 1][i] = v[ny - 2][i];
//	}
//}
//
//void Fluid::boundryCondDirichlet()
//{
//	for (int i = 0; i < nx; i++)u[0][i] = u[ny - 1][i] = 0.0;
//	for (int j = 0; j < ny; j++)v[j][0] = v[j][nx - 1] = 0.0;
//}