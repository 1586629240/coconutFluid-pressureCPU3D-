#include "FluidKernel.cuh"

#define myMax(a, b) ((a) > (b) ? (a) : (b))
#define myMin(a, b) ((a) < (b) ? (a) : (b))

void integrateCuImpl(double dt, double gravity, int nx, int ny, v1d& v)
{
	if (fabs(gravity) < 1e-6)return;
	thrust::transform(
		thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(nx * ny),
		v.data(),
		[=]__device__(int i) { return v[i] + gravity * dt; }
	);
}

__global__ void setObstacleCuImpl(
	double vx, double vy, int nx, int ny,
	double dxy, int threadCnt, double ballX, double ballY,
	double ballR2, double t, double* barrier, double* u, double* v, double* smoke)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny; idx += threadCnt)
	{
		int x = idx % (nx - 2) + 1, y = idx / (nx - 2) + 1;
		if (y >= ny - 2 || x >= nx - 2)continue;

		barrier[y * nx + x] = 1.0;

		double dx1 = (x + 0.5) * dxy - ballX;
		double dy1 = (y + 0.5) * dxy - ballY;

		if (dx1 * dx1 + dy1 * dy1 < ballR2)
		{
			barrier[y * nx + x] = 0;
			smoke[y * nx + x] = (0.5 + 0.5 * sin(t));
			u[y * nx + (x + 1)] = u[y * nx + x] = vx;
			v[(y + 1) * nx + x] = v[y * nx + x] = vy;
		}
	}
}

__device__ double avgUCuImpl(int x, int y, int nx, double* u)
{
	return (
		u[(y - 1) * nx + x] + u[y * nx + x] +
		u[(y - 1) * nx + (x + 1)] + u[y * nx + (x + 1)]
		) * 0.25;
}

__device__ double avgVCuImpl(int x, int y, int nx, double* v)
{
	return (
		v[y * nx + (x - 1)] + v[y * nx + x] +
		v[(y + 1) * nx + (x - 1)] + v[(y + 1) * nx + x]
		) * 0.25;
}

/*
fieldTy: 1.U 2.V 3.S
*/
__device__ double interpolationCuImpl(
	double x, double y, int fieldTy,
	double dxy, int nx, int ny, double* arr)
{
	double h1 = 1.0 / dxy;
	double h2 = 0.5 * dxy;
	double dx = 0.0, dy = 0.0;

	x = myMax(myMin(x, nx * dxy), dxy);
	y = myMax(myMin(y, ny * dxy), dxy);

	switch (fieldTy)
	{
	case 1: dy = h2; break;
	case 2: dx = h2; break;
	case 3: dx = dy = h2; break;
	default: break;
	}

	int x0 = myMin(floor((x - dx) * h1), nx - 1);
	int y0 = myMin(floor((y - dy) * h1), ny - 1);

	double tx = ((x - dx) - x0 * dxy) * h1;
	double ty = ((y - dy) - y0 * dxy) * h1;

	int x1 = myMin(x0 + 1, nx - 1);
	int y1 = myMin(y0 + 1, ny - 1);

	double sx = 1.0 - tx;
	double sy = 1.0 - ty;

	double val = sx * sy * arr[y0 * nx + x0] +
		sx * ty * arr[y1 * nx + x0] +
		tx * ty * arr[y1 * nx + x1] +
		tx * sy * arr[y0 * nx + x1];
	return val;
}

__global__ void advectVelCuImpl(
	double dt, int nx, int ny, double dxy, int threadCnt,
	double* barrier, double* u, double* v, double* newU, double* newV)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny; idx += threadCnt)
	{
		int x = idx % (nx - 2) + 1, y = idx / (nx - 2) + 1;

		if (y >= ny - 1 || x >= nx - 1)continue;

		double dxy2 = 0.5 * dxy;

		if (barrier[y * nx + x] != 0 && barrier[y * nx + (x - 1)] != 0 && y < ny - 1)
		{
			double y2 = y * dxy + dxy2, x2 = x * dxy;
			double valu = u[y * nx + x], valv = avgVCuImpl(x, y, nx, v);

			x2 = x2 - dt * valu, y2 = y2 - dt * valv;
			newU[y * nx + x] = interpolationCuImpl(x2, y2, 1, dxy, nx, ny, u);
		}

		if (barrier[y * nx + x] != 0 && barrier[(y - 1) * nx + x] != 0 && x < nx - 1)
		{
			double y2 = y * dxy, x2 = x * dxy + dxy2;
			double valu = avgUCuImpl(x, y, nx, u), valv = v[y * nx + x];

			x2 = x2 - dt * valu, y2 = y2 - dt * valv;
			newV[y * nx + x] = interpolationCuImpl(x2, y2, 2, dxy, nx, ny, v);
		}
	}
}

__global__ void advectSmokeCuImpl(
	double dt, int nx, int ny, double dxy, int threadCnt,
	double* barrier, double* u, double* v, double* smoke, double* newSmoke)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny; idx += threadCnt)
	{
		int x = idx % nx, y = idx / nx;
		if (y >= ny - 1 || x >= nx - 1)continue;

		double dxy2 = 0.5 * dxy;
		if (barrier[y * nx + x] != 0)
		{
			double y2 = y * dxy + dxy2, x2 = x * dxy + dxy2;
			double valu = (u[y * nx + x] + u[y * nx + (x + 1)]) * 0.5;
			double valv = (v[y * nx + x] + v[(y + 1) * nx + x]) * 0.5;
			x2 = x2 - dt * valu, y2 = y2 - dt * valv;
			newSmoke[y * nx + x] = interpolationCuImpl(x2, y2, 3, dxy, nx, ny, smoke);
		}
	}
}

__global__ void clearToZeroCuImpl(int nx, int ny, int threadCnt, double* arr)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny; idx += threadCnt)
	{
		arr[idx] = 0.0;
	}
}

__global__ void initSceneVortexStreetCuImpl(
	int nx, int ny, double dx, double dy, int threadCnt,
	double inVel, double pipeMinH, double pipeMaxH,
	double* barrier, double* smoke, double* u)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny; idx += threadCnt)
	{
		int x = idx % nx, y = idx / nx;
		if (y >= ny || x >= nx)continue;

		double s = 1;
		if (x == 0 || y == 0 || y == ny - 1)s = 0.0;
		barrier[y * nx + x] = s;

		if (x == 1)u[y * nx + x] = inVel;
		if (x == 0 && (y > pipeMinH && y < pipeMaxH))
			smoke[y * nx + x] = 1;
	}
}

__global__ void boundryCondPeriodCuImplXaxis(
	int nx, int ny, int threadCnt, double* u)
{
	double swapTmp;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx; idx += threadCnt)
	{
		swapTmp = u[idx];
		u[idx] = u[(ny - 1) * nx + idx];
		u[(ny - 1) * nx + idx] = swapTmp;
	}
}

__global__ void boundryCondPeriodCuImplYaxis(
	int nx, int ny, int threadCnt, double* u)
{
	double swapTmp;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ny; idx += threadCnt)
	{
		swapTmp = u[idx * nx];
		u[idx * nx] = u[idx * nx + nx - 1];
		u[idx * nx + nx - 1] = swapTmp;
	}
}

void boundryCondPeriodCuImpl(
	int nx, int ny, double* u, double* v)
{
	boundryCondPeriodCuImplXaxis << <2, 64 >> > (nx, ny, 64 * 64, u);
	cudaDeviceSynchronize();
	boundryCondPeriodCuImplYaxis << <2, 64 >> > (nx, ny, 64 * 64, v);
	cudaDeviceSynchronize();
}