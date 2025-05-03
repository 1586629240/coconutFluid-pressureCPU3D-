#include "Fluid.cuh"
#include "FluidKernel.cuh"
#include <thrust/host_vector.h>

void setObstacle(double x, double y, bool reset, double t = 0)
{
	auto f = scene.fluid;
	double oldX = scene.ballX, oldY = scene.ballY;
	scene.ballX = x, scene.ballY = y;

	if (x<=scene.ballR || x>=f->getNX() - scene.ballR)return;
	if (y<=scene.ballR || y>=f->getNY() - scene.ballR)return;

	double vx = (!reset) * ((x - oldX) / scene.dt);
	double vy = (!reset) * ((y - oldY) / scene.dt);

	vx = max(min(vx, 2.0), -2.0);
	vy = max(min(vy, 2.0), -2.0);

	setObstacleCuImpl << <16, 128 >> > (
		vx, vy, f->getNX(), f->getNY(), f->getdxy(), 16 * 128,
		x, y, scene.ballR * scene.ballR, t,
		thrust::raw_pointer_cast(f->getBrrier().data()),
		thrust::raw_pointer_cast(f->getU().data()),
		thrust::raw_pointer_cast(f->getV().data()),
		thrust::raw_pointer_cast(f->getSmoke().data())
		);
	cudaDeviceSynchronize();
}

void initScenePaint(size_t nxy)
{
	scene.dt = 0.01;
	scene.ballR = 0.05;

	double dxdy = 1.0 / nxy;
	int nx = nxy, ny = nxy;

	double density = 1000.0;
	scene.fluid = new Fluid(density, nx, ny, dxdy);

	scene.gravity = 0;
	scene.enableMouse = true;
}

void initSceneVortexStreet(size_t nxy)
{
	scene.dt = 0.01;
	scene.ballR = 0.05;

	double dxdy = 1.0 / nxy;
	int nx = nxy, ny = nxy;

	double density = 100000.0;
	auto f = scene.fluid = new Fluid(density, nx, ny, dxdy);

	double inVel = 2;
	double pipeH = 0.05 * f->getNY();
	double pipeMinH = floor(0.5 * f->getNY() - 0.5 * pipeH);
	double pipeMaxH = floor(0.5 * f->getNY() + 0.5 * pipeH);

	initSceneVortexStreetCuImpl << <16, 128 >> > (
		nx + 2, ny + 2, dxdy, dxdy, 16 * 128,
		inVel, pipeMinH, pipeMaxH,
		thrust::raw_pointer_cast(f->getBrrier().data()),
		thrust::raw_pointer_cast(f->getSmoke().data()),
		thrust::raw_pointer_cast(f->getU().data()));
	cudaDeviceSynchronize();

	setObstacle(0.2, 0.5, true);
	scene.gravity = 0.0;
	scene.enableMouse = true;
}

void simulate(int step)
{
	for (int i = 0; i < step; i++)
		scene.fluid->simulate(scene.dt, scene.gravity, 20);
}

void display(Fluid& fluid) {
    int N = fluid.getNY();
    auto buf = GetImageBuffer();
	
	int nx = fluid.getNX();
	thrust::host_vector<double> smoke = fluid.getSmoke();

    for (int y = 0; y < 480; y++) {
        for (int x = 0; x < 640; x++) {
            int dx = static_cast<int>((x / 640.0f) * N);
            int dy = static_cast<int>((y / 480.0f) * N);
			float d = smoke[dy * nx + dx];
			//buf[y * 640 + x] = HSVtoRGB(d*360, 1, 1);
			buf[y * 640 + x] = RGB(d * 255, d * 255, d * 255);
        }
    }
    FlushBatchDraw();
}

void mouseEvent(Fluid& cube)
{
	if (scene.enableMouse == false) return;

	static bool drawing = false;
	static double t = 0;
	ExMessage m;
	if (peekmessage(&m)) 
	{
		switch (m.message) 
		{
		case WM_LBUTTONDOWN: drawing = true; break;
		case WM_LBUTTONUP:   drawing = false; break;
		}
	}

	if (drawing && m.message == WM_MOUSEMOVE) 
	{
		double x = m.x / 640.;
		double y = m.y / 480.;
		setObstacle(x, y, false, t);
		t += 0.001;
	}
}

int run()
{	
	initSceneVortexStreet(200);
	initgraph(640, 480);

	for (;;)
	{	
		simulate(1);
		mouseEvent(*scene.fluid);
		display(*scene.fluid);
	}

	delete scene.fluid;
	return 0;
}

int profile()
{
	initSceneVortexStreet(200);
	simulate(1);
	delete scene.fluid;
	return 0;
}

int main(int argc, char* argv[])
{
	if (argc > 1)
		return profile();
	else
		return run();
}