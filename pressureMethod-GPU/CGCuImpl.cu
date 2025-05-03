#include "FluidKernel.cuh"
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>

#define rawPtr1(a)         thrust::raw_pointer_cast(##a.data())
#define rawPtr2(a,b)       rawPtr1(a),rawPtr1(b)
#define rawPtr3(a,b,c)     rawPtr1(a),rawPtr1(b),rawPtr1(c)
#define rawPtr4(a,b,c,d)   rawPtr1(a),rawPtr1(b),rawPtr1(c),rawPtr1(d)
#define rawPtr5(a,b,c,d,e) rawPtr1(a),rawPtr1(b),rawPtr1(c),rawPtr1(d),rawPtr1(e)

__global__ void updateVel0CuImpl(
    int nx, int ny, int threadCnt,
    double* barrier, double* u, double* v, double* p)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny; idx += threadCnt)
    {
        int x = idx % (nx - 2) + 1, y = idx / (nx - 2) + 1;

        if (x >= nx - 1 || y >= ny - 1) return;
        if (barrier[y * nx + x] == 0) continue;

        double sx0 = barrier[y * nx + (x - 1)];
        double sy0 = barrier[(y - 1) * nx + x];
        if (sx0 + sy0 == 0)continue;

        double p_val = p[y * nx + x];
        u[y * nx + x] -= sx0 * p_val;
        v[y * nx + x] -= sy0 * p_val;
    }
}

__global__ void updateVel1CuImpl(
    int nx, int ny, int threadCnt,
    double* barrier, double* u, double* v, double* p)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny; idx += threadCnt)
    {
        int x = idx % (nx - 2) + 1, y = idx / (nx - 2) + 1;

        if (x >= nx - 1 || y >= ny - 1) return;
        if (barrier[y * nx + x] == 0) continue;

        double sx1 = barrier[y * nx + (x + 1)];
        double sy1 = barrier[(y + 1) * nx + x];
        if (sx1 + sy1 == 0)continue;

        double p_val = p[y * nx + x];
        u[y * nx + x + 1] += sx1 * p_val;
        v[(y + 1) * nx + x] += sy1 * p_val;
    }
}

__global__ void ComputeBAndSvalKernel(
    int nx, int ny, int threadCnt,
    double* barrier, double* u, double* v, double* b, double* sval)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny; idx += threadCnt)
    {
        int x = idx % (nx - 2) + 1, y = idx / (nx - 2) + 1;
        if (y >= ny - 1 || x >= nx - 1) return;

        double sx0 = barrier[y * (nx)+(x - 1)];
        double sx1 = barrier[y * (nx)+(x + 1)];
        double sy0 = barrier[(y - 1) * (nx)+x];
        double sy1 = barrier[(y + 1) * (nx)+x];
        double s = sx0 + sx1 + sy0 + sy1;
        sval[y * nx + x] = s;

        if (s == 0)
        {
            b[y * nx + x] = 0.0;
            continue;
        }

        double div =
            u[y * nx + (x + 1)] - u[y * nx + x] +
            v[(y + 1) * nx + x] - v[y * nx + x];
        b[y * nx + x] = -div;
    }
}

__global__ void MatrixVectorMultiplyKernel(
    int nx, int ny, int threadCnt,
    double* barrier, double* sval, double* vec, double* result)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nx * ny; idx += threadCnt)
    {
        int x = idx % (nx - 2) + 1, y = idx / (nx - 2) + 1;

        if (y >= ny - 1 || x >= nx - 1) return;
        double s = sval[y * nx + x];
        double sum =
            vec[y * nx + (x - 1)] * barrier[y * nx + (x - 1)] +
            vec[y * nx + (x + 1)] * barrier[y * nx + (x + 1)] +
            vec[(y - 1) * nx + x] * barrier[(y - 1) * nx + x] +
            vec[(y + 1) * nx + x] * barrier[(y + 1) * nx + x];

        result[y * nx + x] = s * vec[y * nx + x] - sum * (s != 0);
    }
}

__global__ void dot3MatKernel(
    unsigned size, int threadCnt,
    double* a, double* b, double* c,double* res)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += threadCnt)
        res[idx] = a[idx] * b[idx] *c[idx];
}

__global__ void rudeceKernel(const double* input, double* result, int N)
{
    __shared__ double smem[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[threadIdx.x] = (tid < N) ? input[tid] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)atomicAdd(result, smem[0]);
}

double myReduce(const thrust::device_vector <double> & data) 
{
    thrust::device_vector <double> result(1, 0);
    const int block_size = 256;
    int grid_size = (data.size() + block_size - 1) / block_size;

    rudeceKernel << <grid_size, block_size >> > (rawPtr2(data, result), data.size());
    cudaDeviceSynchronize();
    return result[0];
}

double dot3MatCu(v1d& a, v1d& b, v1d& c)
{
    static auto tmp = a;

    dot3MatKernel << <128, 128 >> > (a.size(), 128 * 128, rawPtr4(a, b, c, tmp));
    cudaDeviceSynchronize();
    return myReduce(tmp);
}

__global__ void xAddcMulyCuKernel(
    unsigned size, int threadCnt,
    double* a, double* b, double* res, double c)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += threadCnt)
        res[idx] = a[idx] + c * b[idx];
}

void xAddcMulyCu(v1d& a, v1d& b, bool resPos, double c)
{
    if (resPos == 0)
        xAddcMulyCuKernel << <128, 128 >> > (a.size(), 128 * 128, rawPtr3(a, b, a), c);
    else
        xAddcMulyCuKernel << <128, 128 >> > (a.size(), 128 * 128, rawPtr3(a, b, b), c);
	cudaDeviceSynchronize();
}

void ConjugateGradientCu(
    int maxIter, double dt, int nx, int ny, double pdxydt,
    v1d& barrier, v1d& u, v1d& v, v1d& pressure)
{
    thrust::device_vector<double> d_b(ny * nx, 0.0);
    thrust::device_vector<double> d_sval(ny * nx, 0.0);

    ComputeBAndSvalKernel << <128, 128 >> > (
        nx, ny, 128 * 128, rawPtr5(barrier, u, v, d_b, d_sval));
    cudaDeviceSynchronize();

    thrust::device_vector<double> d_r = d_b;
    thrust::device_vector<double> d_d = d_b;
    thrust::device_vector<double> d_p(ny * nx, 0.0);
    thrust::device_vector<double> d_Ad(ny * nx, 0.0);

    double rr_old = dot3MatCu(d_r, d_r, barrier);
    if (rr_old < 1e-6) return;

    for (int iter = 0; iter < maxIter; ++iter)
    {
        MatrixVectorMultiplyKernel << <128, 128 >> > (nx, ny, 128 * 128, rawPtr4(barrier, d_sval, d_d, d_Ad));
        cudaDeviceSynchronize();

        double d_dot_Ad = dot3MatCu(d_d, d_Ad, barrier);
        double alpha = rr_old / (d_dot_Ad + 1e-6);

        xAddcMulyCu(d_p, d_d, 0, alpha), xAddcMulyCu(d_r, d_Ad, 0, -alpha);
        double rr_new = dot3MatCu(d_r, d_r, barrier);
        if (std::sqrt(rr_new) < 1e-6) break;

        double beta = rr_new / rr_old;
        xAddcMulyCu(d_r, d_d, 1, beta);
        rr_old = rr_new;
    }

    xAddcMulyCu(pressure, d_p, 0, pdxydt);

    updateVel0CuImpl << <128, 128 >> > (nx, ny, 128 * 128, rawPtr4(barrier, u, v, d_p));
    cudaDeviceSynchronize();
    updateVel1CuImpl << <128, 128 >> > (nx, ny, 128 * 128, rawPtr4(barrier, u, v, d_p));
    cudaDeviceSynchronize();
}