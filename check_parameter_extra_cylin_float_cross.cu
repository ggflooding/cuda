#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "time.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define ERRMAX 1E-5
#define NX 1375
#define NY 223
#define NTMAX 100000000
#define NTHREAD 64
#define NBLOCK 64
#define Q 9
#define Inputparameter "Inputparameter.dat"

//Macroscopic properties
#define L_P (0.011*sqrt(3)+0.011*2)  //0.01/2/2.5/1.38/10*2.23 ! Channel height (m)
#define BHR (0.19565)  //0.09   ! BLOCK/HEIGHT RATIO~ circle diameter / length
#define BLOK (0.018)  //L_P*BHR !block height (m) circle diameter
//#define BLKRAD ((NY*2) * BLOK * 0.5 / 0.092)  //how many node in radius
#define Pr (0.7)  //0.001767358!0.7
#define NU_P (1.568E-5)  //U_P*L_P/Re !1.983E-5 !Kinematic Viscosiy (m^2/s)
#define U_P (0.6)  //Physical inlet Velocity (m/s)
//#define U_gap U_P*22./4.  //max velocity in pousiile
#define RE ((U_P * 0.08316) / (NU_P))  //U_P*0.019/NU_P
#define DX_P (0.08316 / 452.)  //BLOK/(NY-2)	! (m)grid size
#define multiCLD (0.08316 * 452. / 0.092) //multi channel hight grid number
#define CLD (multiCLD * 0.041 / 0.08316) //half channel hight grid number
#define DT_P ((DX_P) * (UMAX) / (U_P))  //phesical time step size(s)
#define alpha ((NU_P) / (Pr))   //thermal diffusivity
#define Battery_density (2960.)  //(kg/m^3)
#define Battery_capacity (830.)  //(J/K)
#define Battery_conductivity (3.)  //(w/m K)
#define Ah (3.5)  //total capacitance
#define CRate (2.)  //discharge rate
#define Amper (Ah*CRate)  //ampere(A)

//Microscopic properties
#define TH (1.)  //temp
#define TL (0.)  //coolant temp
#define TM ((TH + TL) / 2.)  //mean temp
#define C (1.)  //dimensionless lattice speed
#define Cs (sqrt( C / 3. )) //dimensionless sound speed Cs
#define DX_LB (1.) //LBM dx
#define DT_LB (DX_LB  / C)  //LBM dt
#define UMAX (0.01) //LBM U0
#define NU_LB (UMAX * (452.) / (RE))  //LB viscosity
#define RE_LB (UMAX * (452.) / (NU_LB))
#define TauF (0.5 + (3. * NU_LB))  //fluid relaxation factor
#define Co ((U_P) / (UMAX))  //dimensionless inlet velocity
#define ALFA_LB (alpha / (DX_P * DX_P) * DT_P)  //NU_LB/pr
#define TauG (0.5 + (3 * ALFA_LB)) //thermal relaxation time
#define tau_solid ((Battery_conductivity / (Battery_capacity*Battery_density)) / ((DX_P) * (DX_P)) * (DT_P) * 3. + 0.5)  //solid relaxation time
#define p0 (1.)  //LB density
#define Ma_P ((U_P)/(Cs*Co))  //U_P/Co become dimensionless U_LB

const float BLKRAD ((125 * 4) * BLOK * 0.5 / 0.092);  //how many node in radius
const float BLKCNTX[16]={250 ,250+59.788 ,250+59.788*2 ,250+59.788*3 ,250+59.788*4 ,250+59.788*5 ,250+59.788*6 ,250+59.788*7 \
						,250+59.788*8 ,250+59.788*9 ,250+59.788*10 ,250+59.788*11 ,250+59.788*12 ,250+59.788*13 \
						,250+59.788*14 ,250+59.788*15};
const float BLKCNTY[16]={59.788 ,NY-59.788 ,59.788 ,NY-59.788 ,59.788 ,NY-59.788 ,59.788 ,NY-59.788 ,59.788 ,NY-59.788 , \
						59.788 ,NY-59.788 ,59.788 ,NY-59.788 ,59.788 ,NY-59.788};

struct attribute{
	float pIn[Q];
	float Q_valueXY;
	float Q_valueXX;
	float Q_valueYY;
	float Q_value;
	float tau_vis;
	float tau_LB;
	float fEq[Q];
	float suma[Q];
	float fm[Q];
	float sumb;
	float pOut[Q];
	float u[2];
	float u0[2];
	float u1[2];
	float rho;
	float vor;
	float B;
};

__constant__ float dev_BLKRAD ((125*4) * BLOK * 0.5 / 0.092);  //how many node in radius
__constant__ float dev_BLKCNTX[16]={250 ,250+59.788 ,250+59.788*2 ,250+59.788*3 ,250+59.788*4 ,250+59.788*5 ,250+59.788*6 ,250+59.788*7 \
						,250+59.788*8 ,250+59.788*9 ,250+59.788*10 ,250+59.788*11 ,250+59.788*12 ,250+59.788*13 \
						,250+59.788*14 ,250+59.788*15};
__constant__ float dev_BLKCNTY[16]={59.788 ,NY-59.788 ,59.788 ,NY-59.788 ,59.788 ,NY-59.788 ,59.788 ,NY-59.788 ,59.788 ,NY-59.788 , \
						59.788 ,NY-59.788 ,59.788 ,NY-59.788 ,59.788 ,NY-59.788};

__constant__ float dev_w[Q] = { 4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 36, 1. / 36, 1. / 36, 1. / 36 };  //LBM weights factor
__constant__ int M_p[Q * Q] = {
	1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , \
	-4. , -1. , -1. , -1. , -1. , 2. , 2. , 2. , 2. , \
	4. , -2. , -2. , -2. , -2. , 1. , 1. , 1. , 1. , \
	0 , 1. , 0 , -1. , 0 , 1. , -1. , -1. , 1. , \
	0 , -2. , 0 , 2. , 0 , 1. , -1. , -1. , 1. , \
	0 , 0 , 1. , 0 , -1. , 1. , 1. , -1. , -1. , \
	0 , 0 , -2. , 0 , 2. , 1. , 1. , -1. , -1. , \
	0 , 1. , -1. , 1. , -1. , 0 , 0 , 0 , 0 , \
	0 , 0 , 0 , 0 , 0 , 1. , -1. , 1. , -1. };
__constant__ int M_nega[Q * Q] = {
	4. , -4. , 4. , 0 , 0 , 0 , 0 , 0 , 0 , \
	4. , -1. , -2. , 6. , -6. , 0 , 0 , 9. , 0 , \
	4. , -1. ,  -2. , 0 , 0 , 6. , -6. , -9. , 0 , \
	4. , -1. , -2. , -6. , 6. , 0 , 0 , 9. , 0 , \
	4. , -1. , -2. , 0 , 0 , -6. , 6. , -9. , 0 , \
	4. , 2. , 1. , 6. , 3. , 6. , 3. , 0 , 9. , \
	4. , 2. , 1. , -6. , -3. , 6. , 3. , 0 , -9. , \
	4. , 2. , 1. , -6. , -3. , -6. , -3. , 0 , 9. , \
	4. , 2. , 1. , 6. , 3. , -6. , -3. , 0 , -9. };
__constant__ int dev_e[Q * 2] = { 0, 0, C, 0, 0, C, -C, 0, 0, -C, C, C, -C, C, -C, -C, C, -C };

attribute *domain;

cudaError_t LBM();
int Ord2(int x, int y, int nx);
void Init();
void Load();
void OutWatch(int t, float err, float rhoav);
void Parameter();
void Outp(int t);
void Point_checkX_a(int t, float ux_a);
void Point_checkX_b(int t, float ux_b);
void Point_checkX_c(int t, float ux_c);
void Point_checkX_d(int t, float ux_d);
void Point_checkX_e(int t, float ux_e);
void Point_checkX_f(int t, float ux_f);
void Point_checkX_g(int t, float ux_g);
void Point_checkX_h(int t, float ux_h);
void Point_checkX_i(int t, float ux_i);
void Point_checkY_a(int t, float uy_a);
void Point_checkY_b(int t, float uy_b);
void Point_checkY_c(int t, float uy_c);
void Point_checkY_d(int t, float uy_d);
void Point_checkY_e(int t, float uy_e);
void Point_checkY_f(int t, float uy_f);
void Point_checkY_g(int t, float uy_g);
void Point_checkY_h(int t, float uy_h);
void Point_checkY_i(int t, float uy_i);
void SinglePointCheck();

__device__ int d_Ord2(int x, int y, int nx);
__device__ void d_Ord2r(int id, int *x, int *y, int nx);
__device__ float d_Cfeq(float u[2], float rho, int k);
__global__ void Init_1(attribute *domain);
__global__ void Init_load(attribute *domain);
__global__ void Fluid_LES(attribute *domain);
__global__ void Fluid_MRT1(attribute *domain);
__global__ void Fluid_MRT2(attribute *domain);
__global__ void Fluid_MRT3(attribute *domain);
__global__ void bounceback(attribute *domain);
__global__ void Streaming(attribute *domain);
__global__ void Inlet(attribute *domain);
__global__ void Period(attribute *domain);
__global__ void Outlet(attribute *domain);
__global__ void Summation1(attribute *domain);
__global__ void Summation2(attribute *domain);
__global__ void Summation3(attribute *domain);
__global__ void Error(attribute *domain, float *err);
__global__ void dev_Point_checkX(attribute *domain, float *ux_a, float *ux_b, float *ux_c, float *ux_d, float *ux_e, float *ux_f, float *ux_g, float *ux_h, float *ux_i);
__global__ void dev_Point_checkY(attribute *domain, float *uy_a, float *uy_b, float *uy_c, float *uy_d, float *uy_e, float *uy_f, float *uy_g, float *uy_h, float *uy_i);
__global__ void Watch(attribute *domain, float *rhoav);
__global__ void MicroToMacro(attribute *domain);
__global__ void Cvor1(attribute *domain);
__global__ void Cvor2(attribute *domain);
__global__ void Cvor3(attribute *domain);

int main(){
	Parameter();
	Init();
	//Load();
	cudaError_t cudaStatus = LBM();
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "Cuda LBM Failed!");
		return 1;
	}

	free(domain);

	return 0;
}

cudaError_t LBM(){
	attribute *dev_domain;
	cudaError_t cudaStatus;
	clock_t start, finish;
	float *ux_a = (float*)malloc(sizeof(float));
	float *ux_b = (float*)malloc(sizeof(float));
	float *ux_c = (float*)malloc(sizeof(float));
	float *ux_d = (float*)malloc(sizeof(float));
	float *ux_e = (float*)malloc(sizeof(float));
	float *ux_f = (float*)malloc(sizeof(float));
	float *ux_g = (float*)malloc(sizeof(float));
	float *ux_h = (float*)malloc(sizeof(float));
	float *ux_i = (float*)malloc(sizeof(float));
	float *uy_a = (float*)malloc(sizeof(float));
	float *uy_b = (float*)malloc(sizeof(float));
	float *uy_c = (float*)malloc(sizeof(float));
	float *uy_d = (float*)malloc(sizeof(float));
	float *uy_e = (float*)malloc(sizeof(float));
	float *uy_f = (float*)malloc(sizeof(float));
	float *uy_g = (float*)malloc(sizeof(float));
	float *uy_h = (float*)malloc(sizeof(float));
	float *uy_i = (float*)malloc(sizeof(float));
	
	float *err = (float*)malloc(sizeof(float));
	float *rhoav = (float*)malloc(sizeof(float));
	float *dev_err, *dev_rhoav;
	float *dev_ux_a, *dev_ux_b, *dev_ux_c, *dev_ux_d, *dev_ux_e, *dev_ux_f, *dev_ux_g, *dev_ux_h, *dev_ux_i;
	float *dev_uy_a, *dev_uy_b, *dev_uy_c, *dev_uy_d, *dev_uy_e, *dev_uy_f, *dev_uy_g, *dev_uy_h, *dev_uy_i;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) goto Error;

cudaStatus = cudaMalloc((void**)&dev_ux_a, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_ux_b, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_ux_c, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_ux_d, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_ux_e, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_ux_f, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_ux_g, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_ux_h, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_ux_i, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	
	cudaStatus = cudaMalloc((void**)&dev_uy_a, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_uy_b, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_uy_c, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_uy_d, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_uy_e, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_uy_f, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_uy_g, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_uy_h, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_uy_i, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	
	cudaStatus = cudaMalloc((void**)&dev_err, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_rhoav, sizeof(float));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_domain, NX * NY * sizeof(attribute));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMemcpy(dev_domain, domain, NX * NY * sizeof(attribute), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) goto Error;

	start = clock();
		Init_1 <<<NBLOCK, NTHREAD >>>(dev_domain);
		//Init_load <<<NBLOCK, NTHREAD >>>(dev_domain);
	for (int t = 0;; t++){
		Fluid_LES <<<NBLOCK, NTHREAD >>>(dev_domain);
		Fluid_MRT1 <<<NBLOCK, NTHREAD >>>(dev_domain);
		Fluid_MRT2 <<<NBLOCK, NTHREAD >>>(dev_domain);
		Fluid_MRT3 <<<NBLOCK, NTHREAD >>>(dev_domain);
		bounceback <<<NBLOCK, NTHREAD >>>(dev_domain);
		Streaming <<<NBLOCK, NTHREAD >>> (dev_domain);
		Inlet <<<NBLOCK, NTHREAD >>> (dev_domain);
		Outlet <<<NBLOCK, NTHREAD >>> (dev_domain);
		Period <<<NBLOCK, NTHREAD >>> (dev_domain);
		Summation1 <<<NBLOCK, NTHREAD >>>(dev_domain);
		//Summation2 <<<NBLOCK, NTHREAD >>>(dev_domain);
		Summation3 <<<NBLOCK, NTHREAD >>>(dev_domain);
		Error <<< 1 , NTHREAD >>>(dev_domain, dev_err);
		MicroToMacro <<<NBLOCK, NTHREAD >>>(dev_domain);
		dev_Point_checkX <<<NBLOCK, NTHREAD>>>(dev_domain, dev_ux_a, dev_ux_b, dev_ux_c, dev_ux_d, dev_ux_e, dev_ux_f, dev_ux_g, dev_ux_h, dev_ux_i);
		dev_Point_checkY <<<NBLOCK, NTHREAD>>>(dev_domain, dev_uy_a, dev_uy_b, dev_uy_c, dev_uy_d, dev_uy_e, dev_uy_f, dev_uy_g, dev_uy_h, dev_uy_i);
		Cvor1 <<<NBLOCK, NTHREAD >>>(dev_domain);
		Cvor2 <<<NBLOCK, NTHREAD >>>(dev_domain);
		Cvor3 <<<NBLOCK, NTHREAD >>>(dev_domain);

		cudaStatus = cudaMemcpy(err, dev_err, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		
		if (t % 300 == 0){
		cudaStatus = cudaMemcpy(ux_a, dev_ux_a, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkX_a(t, *ux_a);
		cudaStatus = cudaMemcpy(ux_b, dev_ux_b, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkX_b(t, *ux_b);
		cudaStatus = cudaMemcpy(ux_c, dev_ux_c, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkX_c(t, *ux_c);
		cudaStatus = cudaMemcpy(ux_d, dev_ux_d, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkX_d(t, *ux_d);
		cudaStatus = cudaMemcpy(ux_e, dev_ux_e, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkX_e(t, *ux_e);
		cudaStatus = cudaMemcpy(ux_f, dev_ux_f, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkX_f(t, *ux_f);
		cudaStatus = cudaMemcpy(ux_g, dev_ux_g, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkX_g(t, *ux_g);
		cudaStatus = cudaMemcpy(ux_h, dev_ux_h, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkX_h(t, *ux_h);
		cudaStatus = cudaMemcpy(ux_i, dev_ux_i, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkX_i(t, *ux_i);
		
		cudaStatus = cudaMemcpy(uy_a, dev_uy_a, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkY_a(t, *uy_a);
		cudaStatus = cudaMemcpy(uy_b, dev_uy_b, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkY_b(t, *uy_b);
		cudaStatus = cudaMemcpy(uy_c, dev_uy_c, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkY_c(t, *uy_c);
		cudaStatus = cudaMemcpy(uy_d, dev_uy_d, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkY_d(t, *uy_d);
		cudaStatus = cudaMemcpy(uy_e, dev_uy_e, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkY_e(t, *uy_e);
		cudaStatus = cudaMemcpy(uy_f, dev_uy_f, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkY_f(t, *uy_f);
		cudaStatus = cudaMemcpy(uy_g, dev_uy_g, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkY_g(t, *uy_g);
		cudaStatus = cudaMemcpy(uy_h, dev_uy_h, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkY_h(t, *uy_h);
		cudaStatus = cudaMemcpy(uy_i, dev_uy_i, sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		Point_checkY_i(t, *uy_i);
		}
		
		if (t % 10000 == 0){
			Watch <<<1, NTHREAD >>>(dev_domain, dev_rhoav);
			cudaStatus = cudaMemcpy(rhoav, dev_rhoav, sizeof(float),cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) goto Error;
			OutWatch(t, *err, *rhoav);
		}
		if (t % 100000 == 0){
			cudaStatus = cudaMemcpy(domain, dev_domain, NX * NY * sizeof(attribute),cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) goto Error;
			Outp(t);
			//SinglePointCheck();
		}
		if (t > NTMAX) break;
	}
	finish = clock();
	printf("Time Take : %7.2lfs\n", (float)(finish - start) / CLOCKS_PER_SEC);

	/*cudaStatus = cudaMemcpy(domain, dev_domain, NX * NY * sizeof(attribute),cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) goto Error;*/

Error:
	free(err);
	free(rhoav);
	free(ux_a);
	free(ux_b);
	free(ux_c);
	free(ux_d);
	free(ux_e);
	free(ux_f);
	free(ux_g);
	free(ux_h);
	free(ux_i);
	free(uy_a);
	free(uy_b);
	free(uy_c);
	free(uy_d);
	free(uy_e);
	free(uy_f);
	free(uy_g);
	free(uy_h);
	free(uy_i);
	cudaFree(dev_ux_a);
	cudaFree(dev_ux_b);
	cudaFree(dev_ux_c);
	cudaFree(dev_ux_d);
	cudaFree(dev_ux_e);
	cudaFree(dev_ux_f);
	cudaFree(dev_ux_g);
	cudaFree(dev_ux_h);
	cudaFree(dev_ux_i);
	cudaFree(dev_uy_a);
	cudaFree(dev_uy_b);
	cudaFree(dev_uy_c);
	cudaFree(dev_uy_d);
	cudaFree(dev_uy_e);
	cudaFree(dev_uy_f);
	cudaFree(dev_uy_g);
	cudaFree(dev_uy_h);
	cudaFree(dev_uy_i);
	cudaFree(dev_err);
	cudaFree(dev_rhoav);
	cudaFree(dev_domain);

	return cudaStatus;
}

int Ord2(int x, int y, int nx){
	return y * nx + x;
}

void Init(){
	domain = (attribute*)malloc(NX * NY * sizeof(attribute));
	memset(domain, 0, NX * NY * sizeof(attribute));
	for(int n=0;n<16;n++){											//	Block Parameters  (circle parameter) 
		for(int i=0;i<NX;i++){
			for(int j=0;j<NY;j++){
				float r; 
				r = (i - BLKCNTX[n]) * (i - BLKCNTX[n]) + (j - BLKCNTY[n]) * (j - BLKCNTY[n]);
				if(r<=BLKRAD*BLKRAD){
					domain[Ord2(i, j, NX)].B = 1;
				}
			}
		}
	}
	return;
}

void Load(){
	domain = (attribute*)malloc(NX * NY * sizeof(attribute));
	memset(domain, 0, NX * NY * sizeof(attribute));
	FILE *fp;
	fp = fopen("12200000.dat", "r");
	for (int i = 0; i < NX; i++){
		for (int j = 0; j < NY; j++){
			fscanf(fp, "%lf ", &domain[Ord2(i, j, NX)].B);
			fscanf(fp, "%lf ", &domain[Ord2(i, j, NX)].u1[0]);
			fscanf(fp, "%lf ", &domain[Ord2(i, j, NX)].u1[1]);
			fscanf(fp, "%lf ", &domain[Ord2(i, j, NX)].rho);
			fscanf(fp, "%lf", &domain[Ord2(i, j, NX)].vor);
		}
	}
	fclose(fp);
	for(int n=0;n<16;n++){											//	Block Parameters  (circle parameter) 
		for(int i=0;i<NX;i++){
			for(int j=0;j<NY;j++){
				float r;
				r = (i-BLKCNTX[n])*(i-BLKCNTX[n])+(j-BLKCNTY[n])*(j-BLKCNTY[n]);
				if(r<=BLKRAD*BLKRAD){
					domain[Ord2(i, j, NX)].B = 1;
				}
			}
		}
	}
	for (int i = 0; i < NX; i++){
		for (int j = 0; j < NY; j++){
			domain[Ord2(i, j, NX)].u[0] = domain[Ord2(i, j, NX)].u1[0] * 0.08316 * (NU_LB) / (452.) / (NU_P);
			domain[Ord2(i, j, NX)].u[1] = domain[Ord2(i, j, NX)].u1[1] * 0.08316 * (NU_LB) / (452.) / (NU_P);
		}
	}
	return;
}
	
	
void OutWatch(int t, float err, float rhoav){
	printf("Time Step %d, ", t);
	printf("Error = %.15lf, ", err);
	printf("Average Density = %.15lf\n", rhoav);
}

void Outp(int t){
	FILE *fp;
	char filename[16];
	sprintf(filename,"%d%s", t,".dat");
	fp = fopen(filename, "w");
	fprintf(fp, "Title=\"LBM Lid Driven Cavity\"\n");
	fprintf(fp, "VARIABLES=\"X\",\"Y\",\"cylin\",\"U\",\"V\",\"P\",\"OMG\"\n");
	fprintf(fp, "ZONE T=\"BOX\",I=%d,J=%d,F=POINT\n", NY, NX);
	for (int i = 0; i < NX; i++){
		for (int j = 0; j < NY; j++){
			fprintf(fp, "%.15lf ", (float)i / (NY - 1) / DX_LB);
			fprintf(fp, "%.15lf ", (float)j / (NY - 1) / DX_LB);
			fprintf(fp, "%.15lf ", domain[Ord2(i, j, NX)].B);
			fprintf(fp, "%.15lf ", domain[Ord2(i, j, NX)].u1[0]);
			fprintf(fp, "%.15lf ", domain[Ord2(i, j, NX)].u1[1]);
			fprintf(fp, "%.15lf ", domain[Ord2(i, j, NX)].rho);
			fprintf(fp, "%.15lf\n", domain[Ord2(i, j, NX)].vor);
		}
	}
	fclose(fp);
	return;
}

void Point_checkX_a(int t, float ux_a){
	FILE *fp3 = fopen("PointX_1-2.dat", "a+");
		fprintf(fp3, " %.15lf ", (t)*DT_P);
		fprintf(fp3, " %.15lf\n", ux_a);
	fclose(fp3);
	return;
}

void Point_checkX_b(int t, float ux_b){
	FILE *fp4 = fopen("PointX_2-4.dat", "a+");
		fprintf(fp4, " %.15lf ", (t)*DT_P);
		fprintf(fp4, "%.15lf\n", ux_b);
	fclose(fp4);
	return;
}

void Point_checkX_c(int t, float ux_c){
	FILE *fp5 = fopen("PointX_4-5.dat", "a+");
		fprintf(fp5, " %.15lf ", (t)*DT_P);
		fprintf(fp5, "%.15lf\n", ux_c);
	fclose(fp5);
	return;
}

void Point_checkX_d(int t, float ux_d){
	FILE *fp6 = fopen("PointX_5-7.dat", "a+");
		fprintf(fp6, " %.15lf ", (t)*DT_P);
		fprintf(fp6, "%.15lf\n", ux_d);
	fclose(fp6);
	return;
}

void Point_checkX_e(int t, float ux_e){
	FILE *fp7 = fopen("PointX_8-9-10.dat", "a+");
		fprintf(fp7, " %.15lf ", (t)*DT_P);
		fprintf(fp7, "%.15lf\n", ux_e);
	fclose(fp7);
	return;
}

void Point_checkX_f(int t, float ux_f){
	FILE *fp8 = fopen("PointX_10-12.dat", "a+");
		fprintf(fp8, " %.15lf ", (t)*DT_P);
		fprintf(fp8, "%.15lf\n", ux_f);
	fclose(fp8);
	return;
}

void Point_checkX_g(int t, float ux_g){
	FILE *fp9 = fopen("PointX_12-13.dat", "a+");
		fprintf(fp9, " %.15lf ", (t)*DT_P);
		fprintf(fp9, "%.15lf\n", ux_g);
	fclose(fp9);
	return;
}

void Point_checkX_h(int t, float ux_h){
	FILE *fp10 = fopen("PointX_13-15.dat", "a+");
		fprintf(fp10, " %.15lf ", (t)*DT_P);
		fprintf(fp10, "%.15lf\n", ux_h);
	fclose(fp10);
	return;
}

void Point_checkX_i(int t, float ux_i){
	FILE *fp20 = fopen("PointX_15-16.dat", "a+");
		fprintf(fp20, " %.15lf ", (t)*DT_P);
		fprintf(fp20, "%.15lf\n", ux_i);
	fclose(fp20);
	return;
}

void Point_checkY_a(int t, float uy_a){
	FILE *fp11 = fopen("PointY_1-2.dat", "a+");
		fprintf(fp11, " %.15lf ", (t)*DT_P);
		fprintf(fp11, "%.15lf\n", uy_a);
	fclose(fp11);
	return;
}

void Point_checkY_b(int t, float uy_b){
	FILE *fp12 = fopen("PointY_2-4.dat", "a+");
		fprintf(fp12, " %.15lf ", (t)*DT_P);
		fprintf(fp12, "%.15lf\n", uy_b);
	fclose(fp12);
	return;
}

void Point_checkY_c(int t, float uy_c){
	FILE *fp13 = fopen("PointY_4-5.dat", "a+");
		fprintf(fp13, " %.15lf ", (t)*DT_P);
		fprintf(fp13, "%.15lf\n", uy_c);
	fclose(fp13);
	return;
}

void Point_checkY_d(int t, float uy_d){
	FILE *fp14 = fopen("PointY_5-7.dat", "a+");
		fprintf(fp14, " %.15lf ", (t)*DT_P);
		fprintf(fp14, "%.15lf\n", uy_d);
	fclose(fp14);
	return;
}

void Point_checkY_e(int t, float uy_e){
	FILE *fp15 = fopen("PointY_8-9-10.dat", "a+");
		fprintf(fp15, " %.15lf ", (t)*DT_P);
		fprintf(fp15, "%.15lf\n", uy_e);
	fclose(fp15);
	return;
}

void Point_checkY_f(int t, float uy_f){
	FILE *fp16 = fopen("PointY_10-12.dat", "a+");
		fprintf(fp16, " %.15lf ", (t)*DT_P);
		fprintf(fp16, "%.15lf\n", uy_f);
	fclose(fp16);
	return;
}

void Point_checkY_g(int t, float uy_g){
	FILE *fp17 = fopen("PointY_12-13.dat", "a+");
		fprintf(fp17, " %.15lf ", (t)*DT_P);
		fprintf(fp17, "%.15lf\n", uy_g);
	fclose(fp17);
	return;
}

void Point_checkY_h(int t, float uy_h){
	FILE *fp18 = fopen("PointY_13-15.dat", "a+");
		fprintf(fp18, " %.15lf ", (t)*DT_P);
		fprintf(fp18, "%.15lf\n", uy_h);
	fclose(fp18);
	return;
}

void Point_checkY_i(int t, float uy_i){
	FILE *fp19 = fopen("PointY_15-16.dat", "a+");
		fprintf(fp19, " %.15lf ", (t)*DT_P);
		fprintf(fp19, "%.15lf\n", uy_i);
	fclose(fp19);
	return;
}

/*void Outp(int t){
	FILE *fp;
	char filename[0];
	sprintf(filename,"%d%s", t,".dat");
	fp = fopen(filename, "w");
	for (int j = 0; j < NY; j++){
			fprintf(fp, "%.25lf\n", domain[Ord2(NX-1 ,j , NX)].pIn[0]);
		}
	fclose(fp);
	return;
}*/

/*void SinglePointCheck(){
	FILE *infile,*outfile;
	infile = fopen("outputMRT0.dat", "w+r");
	outfile = fopen("outputMRT.dat", "a");
	while (int t = 0){
		fprintf(infile, "%.25lf ", domain[Ord2(680 , 60 , NX)].tau_LB);
		fprintf(infile, "%.25lf ", domain[Ord2(680 , 60 , NX)].fEq[4]);
		fprintf(infile, "%.25lf ", domain[Ord2(680 , 60 , NX)].fm[4]);
		fprintf(infile, "%.25lf", domain[Ord2(680 , 60 , NX)].sumb);
	}
	while (feof(infile) == 0){
		fscanf(infile, "%.25lf ", &domain[Ord2(680 , 60 , NX)].tau_LB);
		fscanf(infile, "%.25lf ", &domain[Ord2(680 , 60 , NX)].fEq[4]);
		fscanf(infile, "%.25lf ", &domain[Ord2(680 , 60 , NX)].fm[4]);
		fscanf(infile, "%.25lf\n", &domain[Ord2(680 , 60 , NX)].sumb);
	}
	fprintf(outfile, "%.25lf ", domain[Ord2(680 , 60 , NX)].tau_LB);
	fprintf(outfile, "%.25lf ", domain[Ord2(680 , 60 , NX)].fEq[4]);
	fprintf(outfile, "%.25lf ", domain[Ord2(680 , 60 , NX)].fm[4]);
	fprintf(outfile, "%.25lf\n", domain[Ord2(680 , 60 , NX)].sumb);
	fclose(infile);
	fclose(outfile);
	return;
}*/

void Parameter(){
	FILE *fp0 = fopen(Inputparameter, "w");
			fprintf(fp0, "RE = %.15lf\n", RE);
			fprintf(fp0, "RE_LB = %.15lf\n", RE_LB);
			fprintf(fp0, "Dimensional Channel height (m) = %.15lf\n", L_P);
			fprintf(fp0, "Dimensional Inlet Flow Speed (m/s) = %.15lf\n", U_P);
			fprintf(fp0, "Physical Time Step (s) = %.15lf\n", DT_P);
			fprintf(fp0, "DX_P = %.15lf\n", DX_P);
			fprintf(fp0, "Dimensionless Inlet Flow Speed = %.15lf\n",UMAX);
			fprintf(fp0, "Dimensionless Time Step(LB unit) = %.15lf\n",DT_LB);
			fprintf(fp0, "Dimensionless spacing(LB unit) = %.15lf\n",DX_LB);
			fprintf(fp0, "dimensionless sound speed Cs = %.15lf\n",Cs);
			fprintf(fp0, "dimensionless lattice speed C = %.15lf\n",C);
			fprintf(fp0, "BLOK_node = %.15lf\n",BLKRAD*2.0);
			fprintf(fp0, "NU_LB = %.15lf\n",NU_LB);
			fprintf(fp0, "NU_P = %.15lf\n", NU_P);
			fprintf(fp0, "RE_LB = %.15lf\n", RE_LB);
			fprintf(fp0, "TauF = %.15lf\n", TauF);
			fprintf(fp0, "TauG = %.15lf\n", TauG);
			fprintf(fp0, "Ma = %.15lf\n", Ma_P);
	fclose(fp0);
	return;
}

__device__ int d_Ord2(int x, int y, int nx){
	return y * nx + x;
}

__device__ void d_Ord2r(int id, int *x, int *y, int nx){
	*y = id / nx;
	*x = id % nx;
	return;
}

__device__ float d_Cfeq(float u[2], float rho, int k){
	float v1 = 3. * (dev_e[k * 2] * u[0] + dev_e[k * 2 + 1] * u[1]);
	float v2 = u[0] * u[0] + u[1] * u[1];
	return (rho * dev_w[k] * (1. + v1 / (C) + v1 * v1 / (2. * (C) * (C)) - v2 * 3. / (2. * (C) * (C))));
}


__global__ void Init_1(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		domain[tid].u[0] = UMAX;
		domain[tid].u[1] = 0.;
		if(domain[tid].B==1){
			(domain[tid].u[0]) = 0.;
			(domain[tid].u[1]) = 0.;
		}
		(domain[tid].rho) = p0;
		for (int k = 0; k < Q; k++){
			(domain[tid].pIn[k]) = d_Cfeq(domain[tid].u, domain[tid].rho, k);
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void Init_load(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		domain[tid].u[0] = domain[tid].u1[0] * 0.08316 * (NU_LB) / (452.) / (NU_P);
		domain[tid].u[1] = domain[tid].u1[1] * 0.08316 * (NU_LB) / (452.) / (NU_P);
		for (int k = 0; k < Q; k++){
			(domain[tid].pIn[k]) = d_Cfeq(domain[tid].u, domain[tid].rho, k);
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}

/*__global__ void Collision(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		for (int k = 0; k < Q; k++){
			float feq = d_Cfeq(domain[tid].u, domain[tid].rho, k);
			(domain[tid].pOut[k]) = (domain[tid].pIn[k] - (domain[tid].pIn[k] - feq) / (TauF));
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}*/

__global__ void Fluid_LES(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){                                                                         	//LES
		(domain[tid].Q_valueXY) = 0.;
		(domain[tid].Q_valueXX) = 0.;
		(domain[tid].Q_valueYY) = 0.;
		for (int b = 0; b < Q; b++){
			(domain[tid].Q_valueXY) += (dev_e[b * 2] * dev_e[b * 2 + 1] * (domain[tid].pIn[b] - \
			d_Cfeq(domain[tid].u, domain[tid].rho, b)));
			(domain[tid].Q_valueXX) += (dev_e[b * 2] * dev_e[b * 2] * (domain[tid].pIn[b] - \
			d_Cfeq(domain[tid].u, domain[tid].rho, b)));
			(domain[tid].Q_valueYY) += (dev_e[b * 2 + 1] * dev_e[b * 2 + 1] * (domain[tid].pIn[b] - \
			d_Cfeq(domain[tid].u, domain[tid].rho, b)));
		}
		(domain[tid].Q_value) = (sqrt( 2.*(domain[tid].Q_valueXY)*(domain[tid].Q_valueXY)  \
		+ (domain[tid].Q_valueXX)*(domain[tid].Q_valueXX) + (domain[tid].Q_valueYY)*(domain[tid].Q_valueYY)));
		(domain[tid].tau_vis) = ((sqrt((domain[tid].Q_value) * 18. * 1.414213562 / (domain[tid].rho) * 0.3 * 0.3  \
		+ (TauF) * (TauF)) - (TauF)) / 2.);
		(domain[tid].tau_LB) =  ((TauF) + (domain[tid].tau_vis));
		tid += blockDim.x * gridDim.x;
	}
	return;                                                                                           //LES end
}

__global__ void Fluid_MRT1(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){                                                                       	 	//MRT
		(domain[tid].fEq[0]) = 0.;
		(domain[tid].fEq[1]) = ((domain[tid].rho)*( -2. + 3. * ((domain[tid].u[0]) * (domain[tid].u[0])  \
		 + (domain[tid].u[1]) * (domain[tid].u[1]))));
	    (domain[tid].fEq[2]) = ((domain[tid].rho) * ( 1. - 3. * ((domain[tid].u[0]) * (domain[tid].u[0])  \
	     + (domain[tid].u[1])*(domain[tid].u[1]))));
	    (domain[tid].fEq[3]) = 0.;
	    (domain[tid].fEq[4]) = (-(domain[tid].rho) * (domain[tid].u[0]));
	    (domain[tid].fEq[5]) = 0.;
	    (domain[tid].fEq[6]) = (-(domain[tid].rho) * (domain[tid].u[1]));
	    (domain[tid].fEq[7]) = ((domain[tid].rho) * ((domain[tid].u[0]) * (domain[tid].u[0]) \
	     - (domain[tid].u[1]) * (domain[tid].u[1])));
	    (domain[tid].fEq[8]) = ((domain[tid].rho) * (domain[tid].u[0]) * (domain[tid].u[1]));
	tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void Fluid_MRT2(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
	for (int e = 0; e < 9; e++){
		(domain[tid].suma[e]) = 0.;
	}
    for (int b=0; b<9; b++){
	    for (int qq=0; qq<9; qq++){
			(domain[tid].suma[b]) += ((M_p[9 * b + qq]) * (domain[tid].pIn[qq]));
	    	}
        (domain[tid].fm[b]) = (domain[tid].suma[b]);
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void Fluid_MRT3(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float sm_nega[Q*Q];
	while (tid < NX * NY){
       	float (sm[Q]) = { 0., 1.4, 1.4, 0., 1.2, 0., 1.2, (1./(domain[tid].tau_LB)), (1./(domain[tid].tau_LB))};
		for (int b=0; b<9; b++){
	    	for (int qq=0; qq<9; qq++){
       			(sm_nega[9 * b + qq]) = ((M_nega[9 * b + qq]) * (sm[qq]) / 36.);
   			}
   		}
   	 	for (int b=0; b<9; b++){
         	(domain[tid].sumb) = 0.;
        	for (int q=0; q<9; q++){
            	(domain[tid].sumb) += ((sm_nega[9 * b + q])*((domain[tid].fm[q])-(domain[tid].fEq[q])));   //MRT collision
     		}
         	(domain[tid].pOut[b]) = ((domain[tid].pIn[b])-(domain[tid].sumb));
     	}
     	tid += blockDim.x * gridDim.x;
 	}		                                                                  							  //MRT end
	return;
}

__global__ void bounceback(attribute *domain){
	int tid= threadIdx.x + blockIdx.x * blockDim.x;
	while( tid < NX*NY ){
		if(domain[tid].B==1){
			(domain[tid].pOut[0]) = (domain[tid].pIn[0]);
			(domain[tid].pOut[1]) = (domain[tid].pIn[3]);
			(domain[tid].pOut[3]) = (domain[tid].pIn[1]);
			(domain[tid].pOut[2]) = (domain[tid].pIn[4]);
			(domain[tid].pOut[4]) = (domain[tid].pIn[2]);
			(domain[tid].pOut[5]) = (domain[tid].pIn[7]);
			(domain[tid].pOut[7]) = (domain[tid].pIn[5]);
			(domain[tid].pOut[6]) = (domain[tid].pIn[8]);
			(domain[tid].pOut[8]) = (domain[tid].pIn[6]);
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void Streaming(attribute *domain){                                                         //streaming
	int x,y;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		for (int k = 0; k < Q; k++){
			int xn = (x + dev_e[k * 2]);
			int yn = (y + dev_e[k * 2 + 1]);
			if ((xn >= 0) && (xn < NX) && (yn >= 0) && (yn < NY)){
			(domain[d_Ord2(xn, yn, NX)].pIn[k]) = (domain[tid].pOut[k]);
			}
		}
		tid += blockDim.x * gridDim.x;
	}
	return;                                                                                            //streaming end
}

__global__ void Inlet(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;																
	while (tid < NY){																	//inlet
		(domain[d_Ord2(0 , tid , NX)].u[0]) = (UMAX);                                                         
		
		(domain[d_Ord2(0 , tid , NX)].rho) = ((domain[d_Ord2(0 , tid , NX)].pIn[0]  \
		+ (domain[d_Ord2(0 , tid , NX)].pIn[2]) + (domain[d_Ord2(0 , tid , NX)].pIn[4])  \
		+ 2. * ((domain[d_Ord2(0 , tid , NX)].pIn[3]) + (domain[d_Ord2(0 , tid , NX)].pIn[6])  \
		+ (domain[d_Ord2(0 , tid , NX)].pIn[7]))) / ((1. - (domain[d_Ord2(0 , tid , NX)].u[0])) / (C)));
		
		(domain[d_Ord2(0 , tid , NX)].pIn[1]) = ((domain[d_Ord2(0 , tid , NX)].pIn[3]) + (2. / 3.) \
		* (domain[d_Ord2(0 , tid , NX)].rho) / (C) * (domain[d_Ord2(0 , tid , NX)].u[0]));
		
		(domain[d_Ord2(0 , tid , NX)].pIn[5]) = ((domain[d_Ord2(0 , tid , NX)].pIn[7])   \
		+ (domain[d_Ord2(0 , tid , NX)].rho) * (domain[d_Ord2(0 , tid , NX)].u[0]) / 6. / (C));
		
		(domain[d_Ord2(0 , tid , NX)].pIn[8]) = ((domain[d_Ord2(0 , tid , NX)].pIn[6])  \
		+ (domain[d_Ord2(0 , tid , NX)].rho) * (domain[d_Ord2(0 , tid , NX)].u[0]) / 6. / (C));           //inlet end
		tid += blockDim.x * gridDim.x;
		}
	 	return;
}
		
__global__ void Period(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;										//period
	while (tid < NX){
		(domain[d_Ord2(tid, NY-1, NX)].pIn[4]) = (domain[d_Ord2(tid, 0, NX)].pIn[4]);
		(domain[d_Ord2(tid, NY-1, NX)].pIn[8]) = (domain[d_Ord2(tid, 0, NX)].pIn[8]);

		(domain[d_Ord2(tid, 0, NX)].pIn[2]) = (domain[d_Ord2(tid, NY-1, NX)].pIn[2]);
		(domain[d_Ord2(tid, 0, NX)].pIn[5]) = (domain[d_Ord2(tid, NY-1, NX)].pIn[5]);
		if (tid < NX-1){
			(domain[d_Ord2(tid, NY-1, NX)].pIn[7]) = (domain[d_Ord2(tid, 0, NX)].pIn[7]);
			(domain[d_Ord2(tid, 0, NX)].pIn[6]) = (domain[d_Ord2(tid, NY-1, NX)].pIn[6]);
		}
			
		tid += blockDim.x * gridDim.x;
	}
	return;
}																							//period end

__global__ void Outlet(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (tid = 0; tid < NY; tid++){                                                             			//outlet    
		(domain[d_Ord2(NX-1, tid, NX)].pIn[3]) = (2.0 * domain[d_Ord2(NX-2, tid, NX)].pIn[3] - domain[d_Ord2(NX-3, tid, NX)].pIn[3]);
		(domain[d_Ord2(NX-1, tid, NX)].pIn[6]) = (2.0 * domain[d_Ord2(NX-2, tid, NX)].pIn[6] - domain[d_Ord2(NX-3, tid, NX)].pIn[6]);
		(domain[d_Ord2(NX-1, tid, NX)].pIn[7]) = (2.0 * domain[d_Ord2(NX-2, tid, NX)].pIn[7] - domain[d_Ord2(NX-3, tid, NX)].pIn[7]);
		tid += blockDim.x * gridDim.x;
		}												                                                   //outlet end
	 return;
}

__global__ void Summation1(attribute *domain){
	int x, y;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		if ((x >= 0) && (y >= 0) && (x < NX) && (y < NY)){
			(domain[tid].rho) = 0.;
			(domain[tid].u[0]) = 0.;
			(domain[tid].u[1]) = 0.;
				for (int k = 0; k < Q; k++){
					domain[tid].rho += domain[tid].pIn[k];
					domain[tid].u[0] += dev_e[k * 2] * domain[tid].pIn[k];
					domain[tid].u[1] += dev_e[k * 2 + 1] * domain[tid].pIn[k];
				}
			domain[tid].u[0] /= domain[tid].rho;
			domain[tid].u[1] /= domain[tid].rho;
		}
	tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void Summation2(attribute *domain){
	int x, y;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NY){
		d_Ord2r(tid, &x, &y, NX);
		if ((x >= 0) && (y >= 0) && (x < NX) && (y < NY)){
      		(domain[d_Ord2(NX-1 , tid , NX)].u[1]) = 0.;
  		}
	tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void Summation3(attribute *domain){
	int tid= threadIdx.x + blockIdx.x * blockDim.x;
	while( tid < NX*NY ){
		if(domain[tid].B==1){
			domain[tid].u[0]=0.;
			domain[tid].u[1]=0.;
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}


__global__ void Error(attribute *domain, float *err){
	__shared__ float cache[NTHREAD][2];
	int x, y;
	int tid = threadIdx.x;
	cache[threadIdx.x][0] = 0.;
	cache[threadIdx.x][1] = 0.;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		if ((x > 0) && (y > 0) && (x < NX - 1) && (y < NY - 1)){
			cache[threadIdx.x][0] += (domain[tid].u[0] - domain[tid].u0[0])\
				* (domain[tid].u[0] - domain[tid].u0[0]) + (domain[tid].u[1]\
				- domain[tid].u0[1]) * (domain[tid].u[1] - domain[tid].u0[1]);
			cache[threadIdx.x][1] += domain[tid].u[0] * domain[tid].u[0]\
				+ domain[tid].u[1] * domain[tid].u[1];
		}
		tid += blockDim.x;
	}
	__syncthreads();
	int i = blockDim.x / 2;
	while (i > 0){
		if (threadIdx.x < i){
			cache[threadIdx.x][0] += cache[threadIdx.x + i][0];
			cache[threadIdx.x][1] += cache[threadIdx.x + i][1];
		}
		__syncthreads();
		i /= 2;
	}
	if (threadIdx.x == 0)
		*err = sqrt(cache[threadIdx.x][0] / (cache[threadIdx.x][1] + 1.E-10));
	return;
}

__global__ void Watch(attribute *domain, float *rhoav){
	__shared__ int cachen[NTHREAD];
	__shared__ float cache[NTHREAD];
	int x, y;
	int tid = threadIdx.x;
	cachen[threadIdx.x] = 0;
	cache[threadIdx.x] = 0.;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		if ((x > 0) && (y > 0) && (x < NX - 1) && (y < NY - 1)){
			cachen[threadIdx.x]++;
			cache[threadIdx.x] += domain[tid].rho;
		}
		tid += blockDim.x;
	}
	__syncthreads();
	int i = blockDim.x / 2;
	while (i > 0){
		if (threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
			cachen[threadIdx.x] += cachen[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (threadIdx.x == 0)
		*rhoav = cache[threadIdx.x] / ((float)cachen[threadIdx.x] + 1.E-10);
	return;
}

__global__ void MicroToMacro(attribute *domain){
	int x, y;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		if ((x >= 0) && (y >= 0) && (x < NX) && (y < NY)){
			domain[tid].u1[0] = domain[tid].u[0]*(452.)/(NU_LB)/(0.08316)*(NU_P);
			domain[tid].u1[1] = domain[tid].u[1]*(452.)/(NU_LB)/(0.08316)*(NU_P);
		}
	tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void dev_Point_checkX(attribute *domain, float *ux_a, float *ux_b, float *ux_c, float *ux_d, float *ux_e, float *ux_f, float *ux_g, float *ux_h, float *ux_i){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		*ux_a = domain[d_Ord2((dev_BLKCNTX[0] + dev_BLKCNTX[1])/2. , (dev_BLKCNTY[0] + dev_BLKCNTY[1]) / 2. , NX)].u1[0];	//1-2
		*ux_b = domain[d_Ord2(dev_BLKCNTX[2] , dev_BLKCNTY[1] , NX)].u1[0];	//2-4
		*ux_c = domain[d_Ord2((dev_BLKCNTX[3] + dev_BLKCNTX[4]) / 2., (dev_BLKCNTY[0]+dev_BLKCNTY[1]) / 2. , NX)].u1[0];	//4-5
		*ux_d = domain[d_Ord2(dev_BLKCNTX[5] , dev_BLKCNTY[0] , NX)].u1[0];	//5-7
		*ux_e = domain[d_Ord2(dev_BLKCNTX[8] , (dev_BLKCNTY[0]+dev_BLKCNTY[1]) / 2. , NX)].u1[0];	//8-9-10
		*ux_f = domain[d_Ord2(dev_BLKCNTX[10] , dev_BLKCNTY[1] , NX)].u1[0];	//10-12
		*ux_g = domain[d_Ord2((dev_BLKCNTX[11] + dev_BLKCNTX[12])/2. , (dev_BLKCNTY[0]+dev_BLKCNTY[1]) / 2. , NX)].u1[0]; //12-13
		*ux_h = domain[d_Ord2(dev_BLKCNTX[13] , dev_BLKCNTY[0] , NX)].u1[0];	//13-15
		*ux_i = domain[d_Ord2((dev_BLKCNTX[14] + dev_BLKCNTX[15])/2. , (dev_BLKCNTY[0]+dev_BLKCNTY[1]) / 2. , NX)].u1[0];	//15-16
		/* *ux_a = sqrt(domain[d_Ord2(250 , NY/2 , NX)].u1[0] * domain[d_Ord2(250 , NY/2 , NX)].u1[0] \
					+ domain[d_Ord2(250 , NY/2 , NX)].u1[1] * domain[d_Ord2(250 , NY/2 , NX)].u1[1]);	//1-1
		*ux_b = sqrt(domain[d_Ord2(375 , NY/2 , NX)].u1[0] * domain[d_Ord2(375 , NY/2 , NX)].u1[0] \
					+ domain[d_Ord2(375 , NY/2 , NX)].u1[1] * domain[d_Ord2(375 , NY/2 , NX)].u1[1]);	//2-1
		*ux_c = sqrt(domain[d_Ord2(500 , NY/2 , NX)].u1[0] * domain[d_Ord2(500 , NY/2 , NX)].u1[0] \
					+ domain[d_Ord2(500 , NY/2 , NX)].u1[1] * domain[d_Ord2(500 , NY/2 , NX)].u1[1]);	//3-1
		*ux_d = sqrt(domain[d_Ord2(625 , NY/2 , NX)].u1[0] * domain[d_Ord2(625 , NY/2 , NX)].u1[0] \
					+ domain[d_Ord2(625 , NY/2 , NX)].u1[1] * domain[d_Ord2(625 , NY/2 , NX)].u1[1]);	//4-1
		*ux_e = sqrt(domain[d_Ord2(750 , NY/2 , NX)].u1[0] * domain[d_Ord2(750 , NY/2 , NX)].u1[0] \
					+ domain[d_Ord2(750 , NY/2 , NX)].u1[1] * domain[d_Ord2(750 , NY/2 , NX)].u1[1]);	//5-1
		*ux_f = sqrt(domain[d_Ord2(875 , NY/2 , NX)].u1[0] * domain[d_Ord2(875 , NY/2 , NX)].u1[0] \
					+ domain[d_Ord2(875 , NY/2 , NX)].u1[1] * domain[d_Ord2(875 , NY/2 , NX)].u1[1]);	//6-1
		*ux_g = sqrt(domain[d_Ord2(1000 , NY/2 , NX)].u1[0] * domain[d_Ord2(1000 , NY/2 , NX)].u1[0] \
					+ domain[d_Ord2(1000 , NY/2 , NX)].u1[1] * domain[d_Ord2(1000 , NY/2 , NX)].u1[1]);	//7-1
		*ux_h = sqrt(domain[d_Ord2(1125 , NY/2 , NX)].u1[0] * domain[d_Ord2(1125 , NY/2 , NX)].u1[0] \
					+ domain[d_Ord2(1125 , NY/2 , NX)].u1[1] * domain[d_Ord2(1125 , NY/2 , NX)].u1[1]);	//8-1 */
	tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void dev_Point_checkY(attribute *domain, float *uy_a, float *uy_b, float *uy_c, float *uy_d, float *uy_e, float *uy_f, float *uy_g, float *uy_h, float *uy_i){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		*uy_a = domain[d_Ord2((dev_BLKCNTX[0] + dev_BLKCNTX[1])/2. , (dev_BLKCNTY[0] + dev_BLKCNTY[1]) / 2. , NX)].u1[1];	//1-2
		*uy_b = domain[d_Ord2(dev_BLKCNTX[2] , dev_BLKCNTY[1] , NX)].u1[1];	//2-4
		*uy_c = domain[d_Ord2((dev_BLKCNTX[3] + dev_BLKCNTX[4]) / 2., (dev_BLKCNTY[0]+dev_BLKCNTY[1]) / 2. , NX)].u1[1];	//4-5
		*uy_d = domain[d_Ord2(dev_BLKCNTX[5] , dev_BLKCNTY[0] , NX)].u1[1];	//5-7
		*uy_e = domain[d_Ord2(dev_BLKCNTX[8] , (dev_BLKCNTY[0]+dev_BLKCNTY[1]) / 2. , NX)].u1[1];	//8-9-10
		*uy_f = domain[d_Ord2(dev_BLKCNTX[10] , dev_BLKCNTY[1] , NX)].u1[1];	//10-12
		*uy_g = domain[d_Ord2((dev_BLKCNTX[11] + dev_BLKCNTX[12])/2. , (dev_BLKCNTY[0]+dev_BLKCNTY[1]) / 2. , NX)].u1[1];	//12-13
		*uy_h = domain[d_Ord2(dev_BLKCNTX[13] , dev_BLKCNTY[0] , NX)].u1[1];	//13-15
		*uy_i = domain[d_Ord2((dev_BLKCNTX[14] + dev_BLKCNTX[15])/2. , (dev_BLKCNTY[0]+dev_BLKCNTY[1]) / 2. , NX)].u1[1];  //15-16
	tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void Cvor1(attribute *domain){
	int x, y;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		if ((x > 0) && (y > 0) && (x < NX - 1) && (y < NY - 1)){
			domain[tid].vor = ((domain[d_Ord2(x + 1, y, NX)].u1[1] \
				- domain[d_Ord2(x - 1, y, NX)].u1[1]) / (DX_P + DX_P) \
				- (domain[d_Ord2(x, y + 1, NX)].u1[0] \
				- domain[d_Ord2(x, y - 1, NX)].u1[0]) / (DX_P + DX_P));
			}
	tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void Cvor2(attribute *domain){														//BC LHS & RHS
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (tid = 1; tid<NY-1; tid++ ){
		domain[d_Ord2(0, tid, NX)].vor = ((domain[d_Ord2(1, tid, NX)].u1[1] \
				- domain[d_Ord2(0, tid, NX)].u1[1]) / (DX_P) \
				- (domain[d_Ord2(0, tid + 1, NX)].u1[0] \
				- domain[d_Ord2(0, tid - 1, NX)].u1[0]) / (DX_P + DX_P));
				
		domain[d_Ord2(NX-1, tid, NX)].vor = ((domain[d_Ord2(NX-1, tid, NX)].u1[1] \
				- domain[d_Ord2(NX-2, tid, NX)].u1[1]) / (DX_P) \
				- (domain[d_Ord2(NX-1, tid+1, NX)].u1[0] \
				- domain[d_Ord2(NX-1, tid - 1, NX)].u1[0]) / (DX_P + DX_P));
	tid += blockDim.x * gridDim.x;
	}
	return;
}																								//BC LHS & RHS end

__global__ void Cvor3(attribute *domain){														//BC TOP & BOT
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (tid = 1; tid<NX-1; tid++ ){
		domain[d_Ord2(tid, 0, NX)].vor = ((domain[d_Ord2(tid+1, 0, NX)].u1[1] \
				- domain[d_Ord2(tid-1, 0, NX)].u1[1]) / (DX_P+DX_P) \
				- (domain[d_Ord2(tid, 1, NX)].u1[0] \
				- domain[d_Ord2(tid, 0, NX)].u1[0]) / (DX_P));
				
		domain[d_Ord2(tid, NY-1, NX)].vor = ((domain[d_Ord2(tid+1, NY-1, NX)].u1[1] \
				- domain[d_Ord2(tid-1, NY-1, NX)].u1[1]) / (DX_P+DX_P) \
				- (domain[d_Ord2(tid, NY-1, NX)].u1[0] \
				- domain[d_Ord2(tid, NY-2, NX)].u1[0]) / (DX_P));
	tid += blockDim.x * gridDim.x;
	}
	return;
}																								//BC TOP & BOT end
