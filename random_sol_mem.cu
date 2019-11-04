
// nvcc -std=c++11 random_sol_mem.cu -o random_sol_mem
#include <iostream>
#include "curand.h"
#include "curand_kernel.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdio.h>
#include <iomanip>

#define N_SOL 10000
#define CHARGE 10



struct point{
    double x;
    double y;
};

__device__ double dist(point p1, point p2){
    return sqrt(pow(p1.x - p2.x, 2) +
                pow(p1.y - p2.y, 2));
}

__device__ bool isIntersecting(point p1, point p2,
                               point q1, point q2){

    return  (((q1.x-p1.x)*(p2.y-p1.y) - (q1.y-p1.y) * (p2.x-p1.x)) * 
            ((q2.x-p1.x)*(p2.y-p1.y) - (q2.y-p1.y) * (p2.x-p1.x)) < 0)
            &&
            (((p1.x-q1.x)*(q2.y-q1.y) - (p1.y-q1.y) * (q2.x-q1.x)) * 
            ((p2.x-q1.x)*(q2.y-q1.y) - (p2.y-q1.y) * (q2.x-q1.x)) < 0);
}


__global__ void path_dist(int *path, double*path_dist, double *vec_dist, point *points, 
                          int N_Sol, const int N, const int charge, const int size){
    
    int i = N * (blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= N_Sol*N) return;
    
    int dist = 0;
    
    __shared__ float local_path[size];
    __shared__ float local_dist[N];

    curandState st;
    curand_init(0, i/N, 0, &st);
    
    for(int k=0; k<size; ++k)
    local_path[k] = k%N;  
    

    for (int p=0; p < charge; ++p){
        int tmp_dist = 0; 
        for(int k=1; k<N; ++k){
            int r = (int) ((N-k) * curand_uniform(&st) + k);
            
            int idx_cr = p*N+k;
            int idx_rd = p*N+r;
            int idx_lt = idx_at-1;

            auto tmp  = path[idx_cr];
            path[idx_cr] = path[idx_rd];
            path[idx_rd] = tmp;
            
            local_dist[p] += vec_dist[local_path[idx_lt] * N + local_path[idx_cr]];   
        }   
        path_dist[p] += vec_dist[local_path[p*N] * N + local_path[p+N-1]];
    }

    //B
    for(int p=0; p<charge; ++p){
        bool flag = true;
        
        while(flag){ 
            flag = false;
            
            for(int k=0; k<N-1; ++k){
                
                for(int j=k+1; j<N; ++j){
                    int last = (j+1 == N-1)? 0 : j+1;
                    
                    int w = p*n; 


                    if(isIntersecting(points[local_path[w+k]], points[local_path[w+k+1]],
                                    points[local_path[w+j]], points[local_path[w+last]])){
                        
                        path_dist[p] -= vec_dist[local_path[w+k] * N + local_path[w+k+1]] + 
                                        vec_dist[local_path[w+k+1] * N + local_path[w+k+2]]+
                                        vec_dist[local_path[w+j-1] * N + local_path[w+j]] + 
                                        vec_dist[local_path[w+j] * N + local_path[w+j+1]];
                        
                        auto tmp = local_path[w+k+1];
                        local_path[w+k+1]  = local_path[w+j];
                        local_path[w+j]    = tmp;
                        

                        path_dist[p] += vec_dist[local_path[w+k] * N + local_path[w+k+1]] + 
                                        vec_dist[local_path[w+k+1] * N + local_path[w+k+2]]+
                                        vec_dist[local_path[w+j-1] * N + local_path[w+j]] + 
                                        vec_dist[local_path[w+j] * N + local_path[w+j+1]];
                        
                        flag = true; 
                    }
                }   
            }
            
            if(path_dist[p] < dist)
                dist = path_dist[p]; 
        }
    }   
    
    for (k=0; k<N; k++)
        path[i+k] = local_path[dist*N+k];
    
    path_dist[i/N] = dist;
}

__global__ void points_distance(double *vec_dist, point *points, int width, int heigth){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= heigth || j >= width) return;
    
    vec_dist[i * width + j] = dist(points[i], points[j]);
}

int main(){
    std::cout << std::fixed <<std::setprecision(5);
    const int N; std::cin >> N;
    thrust::host_vector<point> points_CPU;
    
    for(int i=0; i<N; ++i){
        point pt;
        std::cin >> pt.x; std::cin >> pt.y;
        points_CPU.push_back(pt);
    }

    thrust::device_vector<point> points_GPU(points_CPU);
    thrust::device_vector<double> dist_GPU(N*N, 0);
    

    dim3 dimGrid(ceil(N/32.0*CHARGE), ceil(N/32.0*CHARGE), 1);
    dim3 dimBlock(32, 32, 1);
    

    points_distance<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(dist_GPU.data()),
                                           thrust::raw_pointer_cast(points_GPU.data()),
                                           N, N);

    thrust::device_vector<int> paths((N_SOL*N)/CHARGE, 0);
    thrust::device_vector<double> paths_dists(N_SOL/CHARGE, 0);
    const int size = X*CHARGE;

    path_dist<<<ceil(N_SOL/1024.0), 1024>>>(thrust::raw_pointer_cast(paths.data()),
                                          thrust::raw_pointer_cast(paths_dists.data()),
                                          thrust::raw_pointer_cast(dist_GPU.data()),
                                          thrust::raw_pointer_cast(points_GPU.data()), N_SOL, N, CHARGE, size);
    

    thrust::host_vector<int> paths_CPU(paths);
    thrust::host_vector<double> paths_dists_CPU(paths_dists);
                                      

    auto iter = thrust::min_element(paths_dists_CPU.begin(), paths_dists_CPU.end());
    unsigned int position = iter - paths_dists_CPU.begin();
    double max_val = *iter;

    std::cout << max_val << " 0" << std::endl;

    for (auto it = paths.begin()+(position*N); it != paths.begin()+(position*N)+N; ++it)
        std::cout << *it << " ";
 
    std::cout << std::endl;
    
    //std::cerr << "Time: " << time_end << std::endl; 
}