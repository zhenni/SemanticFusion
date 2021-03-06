/*
 * This file is part of SemanticFusion.
 *
 * Copyright (C) 2017 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is SemanticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/semantic-fusion/semantic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include <stdio.h>
#include <assert.h> 

#include <cuda_runtime.h>

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool
        abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    } 
}

__global__ 
void semanticTableUpdate(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probabilities, const int prob_width, const int prob_height, 
                          const int prob_channels,float* map_table,float* map_max,
                          const int map_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // New uniqueness code
    const int check_patch = 16;
    const int x_min = (x - check_patch) < 0 ? 0 : (x - check_patch);
    const int x_max = (x + check_patch) > 640 ? 640 : (x + check_patch);
    const int y_min = (y - check_patch) < 0 ? 0 : (y - check_patch);
    const int y_max = (y + check_patch) < 480 ? 480 : (y + check_patch);

    int surfel_id = tex2D<int>(ids,x,y);
    int first_h, first_w;
    // for (int h = y_min; h < 480; ++h) {
    //     for (int w = x_min; w < x_max; ++w) {
    //         int other_surfel_id = tex2D<int>(ids,w,h);
    //         if (other_surfel_id == surfel_id) {
    //             first_h = h;
    //             first_w = w;
    //             break;
    //         }
    //     }
    // }

    //patch rule 1
    // for (int h = y_min; h < 480; ++h) {
    //     int other_surfel_id;
    //     for (int w = x_min; w < x_max; ++w) {
    //         other_surfel_id = tex2D<int>(ids,w,h);
    //         if (other_surfel_id == surfel_id) {
    //             first_h = h;
    //             first_w = w;
    //             break;
    //         }
    //     }
    //     if (other_surfel_id == surfel_id) {
    //             break;
    //         }
    // }
    //patch rule 2
    for (int h = y_min; h < y_max; ++h) {
        int other_surfel_id;
        for (int w = x_min; w < x_max; ++w) {
            other_surfel_id = tex2D<int>(ids,w,h);
            if (other_surfel_id == surfel_id) {
                first_h = h;
                first_w = w;
                break;
            }
        }
        if (other_surfel_id == surfel_id) {
                break;
            }
    }

    if (first_h != y || first_w != x) {
        surfel_id = 0;
    }
    if (surfel_id > 0) {
        // x,y coordinates in probability image
        const int prob_x = static_cast<int>((float(x) / ids_width) * prob_width);
        const int prob_y = static_cast<int>((float(y) / ids_height) * prob_height);
        
        // memory offset of the probability of the neighborhood class at the same pixel of probability image
        const int channel_offset = prob_width * prob_height; 
        
        // pointer at (prob_x,prob_y)
        const float* probability = probabilities + (prob_y * prob_width + prob_x);

        // pointer at the surfel in prob_table
        float* prior_probability = map_table + surfel_id;

        // go though all class channels to update prob of the correspond surfel
        float total = 0.0;
        for (int class_id = 0; class_id < prob_channels; ++class_id) {
            prior_probability[0] *= probability[0]; // use prob of a class of a pixel to update its correponsded surfel
            total += prior_probability[0];  // sum prob of all classes
            probability += channel_offset;  // go to the next class prob on prob image
            prior_probability += map_size;  // go to the next class prob on surfel map
        }

        // Reset the pointers to the beginning again
        probability = probabilities + (prob_y * prob_width + prob_x);
        prior_probability = map_table + surfel_id;
        float max_probability = 0.0;
        int max_class = -1;
        float new_total = 0.0;
        // normalize probs and search the class with max prob
        for (int class_id = 0; class_id < prob_channels; ++class_id) {
            // Something has gone unexpectedly wrong - reinitialse
            if (total <= 1e-5) {
                prior_probability[0] = 1.0f / prob_channels;
            } else {
                prior_probability[0] /= total; // normalize prob 
                if (class_id > 0 && prior_probability[0] > max_probability) {
                    max_probability = prior_probability[0];
                    max_class = class_id;
                }
            }
            new_total += prior_probability[0];
            probability += channel_offset;
            prior_probability += map_size;
        }
        map_max[surfel_id] = static_cast<float>(max_class);
        map_max[surfel_id + map_size] = max_probability;
        map_max[surfel_id + map_size + map_size] += 1.0;
    }
}

__host__ 
void fuseSemanticProbabilities(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probabilities, const int prob_width, const int prob_height, 
                          const int prob_channels,float* map_table, float* map_max,
                          const int map_size)
{
    // NOTE Res must be pow 2 and > 32
    const int blocks = 32;
    dim3 dimGrid(blocks,blocks);
    dim3 dimBlock(640/blocks,480/blocks);
    semanticTableUpdate<<<dimGrid,dimBlock>>>(ids,ids_width,ids_height,probabilities,prob_width,prob_height,prob_channels,map_table,map_max,map_size);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ 
void updateTable(int n, const int* deleted_ids, const int num_deleted, const int current_table_size,
                 float const* probability_table, const int prob_width, const int prob_height, 
                 const int new_prob_width, float* new_probability_table, float const * map_table, float* new_map_table)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;  // kernal index
    if (index < n) {
        const int class_id = index / new_prob_width;  // get class id of current kernal in new table
        const int component_id = index - (class_id * new_prob_width);  // get surfel id of current kernal in new table
        const int new_id = (class_id * prob_width) + component_id; // get table index with max_componets as width
        if (component_id >= num_deleted) {
            // Initialise to prior (prob height is the number of classes)
            new_probability_table[new_id] = 1.0f / prob_height;
            // Reset the max class surfel colouring lookup
            new_map_table[component_id] = -1.0;
            new_map_table[component_id + prob_width] = -1.0;
            new_map_table[component_id + prob_width + prob_width] = 0.0;
        } else {
            int offset = deleted_ids[component_id]; // get corresponded surf_id in previous table
            new_probability_table[new_id] = probability_table[(class_id * prob_width) + offset];
            // Also must update our max class mapping
            new_map_table[component_id] = map_table[offset];
            new_map_table[component_id + prob_width] = map_table[prob_width + offset];
            new_map_table[component_id + prob_width + prob_width] = map_table[prob_width + prob_width + offset];
        }
    }
}

__host__ 
void updateProbabilityTable(int* filtered_ids, const int num_filtered, const int current_table_size,
                            float const* probability_table, const int prob_width, const int prob_height, 
                            const int new_prob_width, float* new_probability_table, 
                            float const* map_table, float* new_map_table)
/*
filtered_ids: map->GetDeletedSurfelIdsGpu(),
num_filtered: num_deleted,
current_table_size: current_table_size_,
probability_table: class_probabilities_gpu_->gpu_data(),
prob_width: table_width, prob_height: table_height,
new_prob_width: new_table_width, 
new_probability_table: class_probabilities_gpu_buffer_->mutable_gpu_data(),
map_table: class_max_gpu_->gpu_data(),
new_map_table: class_max_gpu_buffer_->mutable_gpu_data()
*/
{   

    const int threads = 512;
    const int num_to_update = new_prob_width * prob_height; // new_table_width*num_classes_
    const int blocks = (num_to_update + threads - 1) / threads;  
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    updateTable<<<dimGrid,dimBlock>>>(num_to_update,filtered_ids,num_filtered,current_table_size,probability_table,prob_width,prob_height,new_prob_width,new_probability_table, map_table, new_map_table);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}


__global__ 
void renderProbabilityMapKernel(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probability_table, const int prob_width, const int prob_height, 
                          float* rendered_probabilities) 
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int surfel_id = tex2D<int>(ids,x,y);
    int projected_probability_offset = y * ids_width + x;
    int probability_table_offset = surfel_id;
    for (int class_id = 0; class_id < prob_height; ++class_id) {
        if (surfel_id > 0) {
            rendered_probabilities[projected_probability_offset] = probability_table[probability_table_offset];
        } else {
            rendered_probabilities[projected_probability_offset] = ((class_id == 0) ? 1.0 : 0.0);
        }
        projected_probability_offset += (ids_width * ids_height);
        probability_table_offset += prob_width;
    }
}


__host__
void renderProbabilityMap(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probability_table, const int prob_width, const int prob_height, 
                          float* rendered_probabilities) 
{
    // NOTE Res must be pow 2 and > 32
    const int blocks = 32;
    dim3 dimGrid(blocks,blocks);
    dim3 dimBlock(ids_width/blocks,ids_height/blocks);
    renderProbabilityMapKernel<<<dimGrid,dimBlock>>>(ids,ids_width,ids_height,probability_table,prob_width,prob_height,rendered_probabilities);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ 
void updateMaxClassKernel(const int n, const float* probabilities, const int classes,
                          float* map_max, const int map_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        // Reset the pointers to the beginning again
        const float* probability = probabilities + index;
        probability += map_size;
        float max_probability = 0.0;
        int max_class = -1;
        for (int class_id = 1; class_id < classes; ++class_id) {
            if (probability[0] > max_probability) {
                max_probability = probability[0];
                max_class = class_id;
            }
            probability += map_size;
        }
        map_max[index] = static_cast<float>(max_class);
        map_max[index + map_size] = max_probability;
    }
}

__host__ 
void updateMaxClass(const int n, const float* probabilities, const int classes,
                    float* map_max, const int map_size)
{
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    updateMaxClassKernel<<<dimGrid,dimBlock>>>(n,probabilities,classes,map_max,map_size);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}
