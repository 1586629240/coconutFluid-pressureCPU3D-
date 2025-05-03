#include "FluidKernel.cuh"

__device__ void GaussSedielKernelCuImpl(
	int x, int y, int nx, int ny, double pdxydt,
	double* barrier, double* u, double* v, double* pre)
{
	if (barrier[y * nx + x] == 0)return;
	const double sx0 = barrier[y * nx + (x - 1)], sx1 = barrier[y * nx + (x + 1)];
	const double sy0 = barrier[(y - 1) * nx + x], sy1 = barrier[(y + 1) * nx + x];
	const double s = sx0 + sx1 + sy0 + sy1;
	if (s == 0)return;

	const double div =
		u[y * nx + (x + 1)] - u[y * nx + x] +
		v[(y + 1) * nx + x] - v[y * nx + x];
	const double p = -div / s;

	pre[y * nx + x] += pdxydt * p;
	u[y * nx + x] -= sx0 * p, u[y * nx + (x + 1)] += sx1 * p;
	v[y * nx + x] -= sy0 * p, v[(y + 1) * nx + x] += sy1 * p;
}

thrust::device_vector<unsigned> redIdx;
thrust::device_vector<unsigned> blackIdx;

void initRedBlackIdx(int nx, int ny)
{
	if (redIdx.size() != 0)return;

	thrust::device_vector<unsigned> redIdxh;
	thrust::device_vector<unsigned> blackIdxh;

	redIdxh.resize((nx - 2) * (ny - 2), 0);
	blackIdxh.resize((nx - 2) * (ny - 2), 0);

	unsigned redPos = 0, blockPos = 0;

	for (int i = 0; i < ny - 2; i++)
	{
		for (int j = (i % 2); j < nx - 2; j += 2)
			redIdxh[redPos++] = i * (nx - 2) + j;

		for (int j = (i + 1) % 2; j < nx - 2; j += 2)
			blackIdxh[blockPos++] = i * (nx - 2) + j;
	}

	redIdxh.resize(redPos);
	blackIdxh.resize(blockPos);

	redIdx = redIdxh;
	blackIdx = blackIdxh;
}

__global__ void GaussSedielCuImpl(
	double dt, int nx, int ny, int threadCnt, double pdxydt, unsigned* gridIdx, unsigned size,
	double* barrier, double* u, double* v, double* pre)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += threadCnt)
	{
		int newIdx = gridIdx[idx];

		int x = newIdx % (nx - 2) + 1, y = newIdx / (nx - 2) + 1;
		GaussSedielKernelCuImpl(x, y, nx, ny, pdxydt, barrier, u, v, pre);
	}
}

__host__ void GaussSedielRBCu(
	int iter, double dt, int nx, int ny, double pdxydt,
	double* barrier, double* u, double* v, double* pre)
{
	initRedBlackIdx(nx, ny);
	for (int i = 0; i < iter; i++)
	{
		GaussSedielCuImpl << <64, 64 >> > (dt, nx, ny, 64 * 64, pdxydt,
			thrust::raw_pointer_cast(redIdx.data()), redIdx.size(),
			barrier, u, v, pre);
		cudaDeviceSynchronize();
		GaussSedielCuImpl << <64, 64 >> > (dt, nx, ny, 64 * 64, pdxydt,
			thrust::raw_pointer_cast(blackIdx.data()), blackIdx.size(),
			barrier, u, v, pre);
		cudaDeviceSynchronize();
	}
}
