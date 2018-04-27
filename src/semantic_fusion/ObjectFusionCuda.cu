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
void updateTable(int num_to_update, const int* deleted_ids, const int num_deleted, const int current_table_size,
                 float const* object_id_table, const int prob_width, const int prob_height, 
                 const int new_prob_width, float* new_object_id_table)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;  // kernal index
    if (index < num_to_update) {
        const int channel_id = index / new_prob_width;  // get class id of current kernal in new table
        const int component_id = index - (channel_id * new_prob_width);  // get surfel id of current kernal in new table
        // const int new_id = (class_id * prob_width) + component_id; // get table index with max_componets as width
        if (component_id >= num_deleted) {
            // Initialise to prior (prob height is the number of classes)
            // new_object_id_table[new_id] = 1.0f / prob_height;
            // Reset the max class surfel colouring lookup
            new_object_id_table[component_id] = 0.0;  // obj id
            new_object_id_table[component_id + prob_width] = 1.0; // obj conf
            new_object_id_table[component_id + prob_width + prob_width] = 0.0;
        } else {
            int offset = deleted_ids[component_id]; // get corresponded surf_id in previous table
            // new_object_id_table[new_id] = object_id_table[(class_id * prob_width) + offset];
            // Also must update our max class mapping
            new_object_id_table[component_id] = object_id_table[offset];
            new_object_id_table[component_id + prob_width] = object_id_table[prob_width + offset];
            new_object_id_table[component_id + prob_width + prob_width] = object_id_table[prob_width + prob_width + offset];
        }
    }
}



__host__
void updateObjectTable(int* deleted_ids, const int num_deleted, const int current_table_size,
                            float const* object_id_table, const int prob_width, const int prob_height, 
                          const int new_table_width, float* new_object_id_table){
    const int threads = 512;
    const int num_to_update = new_table_width;// * prob_height; // new_table_width*num_classes_
    const int blocks = (num_to_update + threads - 1) / threads;  
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    updateTable<<<dimGrid,dimBlock>>>(num_to_update,deleted_ids,num_deleted,current_table_size,
    									object_id_table,prob_width,prob_height,
    									new_table_width,new_object_id_table);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());



}


__global__ 
void objectTableUpdate(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* mask_probabilities,
                    const int x1, const int y1, const int box_width, const int box_height, 
                    const int obj_id, const int class_id, const float class_prob,
                    float* object_id_table, const int map_size)
{
	// masks coordinate indices 
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // New uniqueness code
    if (x<box_width||y<box_height){
    // if (1){
		const int check_patch = 16;
		const int x_min = (x - check_patch) < 0 ? 0 : (x - check_patch);
		const int x_max = (x + check_patch) > box_width ? box_width : (x + check_patch);
		const int y_min = (y - check_patch) < 0 ? 0 : (y - check_patch);
		const int y_max = (y + check_patch) < box_height ? box_height : (y + check_patch);
    	int surfel_id = tex2D<int>(ids,x+x1,y+y1);
    	// int surfel_id = 0;
    	int first_h, first_w;

	    for (int h = y_min; h < y_max; ++h) {
        int other_surfel_id;
        for (int w = x_min; w < x_max; ++w) {
            other_surfel_id = tex2D<int>(ids,w+x1,h+y1);
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

    	if (surfel_id>0) {
    		// pointer at (x,y) on mask
        	const float* probability = mask_probabilities + (y * box_width + x);
        	// pointer at surfel_id on table
	        float* prior_id = object_id_table + surfel_id;

	        // TO DO: fusion
	        //	    
	        if(mask_probabilities[y*box_width+x] > 0.4){
		        object_id_table[surfel_id] = static_cast<float>(obj_id);
	   			object_id_table[surfel_id + map_size] = 1.0;	
	        	object_id_table[surfel_id + map_size + map_size] += 1.0;
	        }    
    	} 
    		

    }
    
}



__host__
void fuseObjects(cudaTextureObject_t ids, const int ids_width, const int ids_height, const float* mask_probabilities,
                    const int x1, const int y1, const int box_width, const int box_height, const int obj_id, const int class_id, const float class_prob,
                    float* object_id_table, const int map_size){
	// NOTE Res must be pow 2 and > 32
    const int blocks = 32; // TODO : global function need check
    dim3 dimGrid(blocks,blocks);
    dim3 dimBlock((box_width+blocks-1)/blocks,(box_height+blocks-1)/blocks);
    objectTableUpdate<<<dimGrid,dimBlock>>>(ids,ids_width,ids_height,mask_probabilities,
    	x1,y1,box_width,box_height, obj_id, class_id, class_prob, object_id_table, map_size);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}


__global__ 
void renderObjectMapKernel(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* object_id_table, const int prob_width, const int prob_height, 
                          float* rendered_objects) 
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int surfel_id = tex2D<int>(ids,x,y);
    int projected_object_offset = y * ids_width + x;
    int object_table_offset = surfel_id;
    if (surfel_id > 0) {
        rendered_objects[projected_object_offset] = object_id_table[object_table_offset]+1;
    } else {
        rendered_objects[projected_object_offset] = 0.0; // ((class_id == 0) ? 1.0 : 0.0);
    }
    // object_table_offset += prob_width;
}
__host__
void renderObjectMap(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* object_id_table, const int prob_width, const int prob_height, 
                          float* rendered_objects)
{
    // NOTE Res must be pow 2 and > 32
    const int blocks = 32; // TODO : global function need check
    dim3 dimGrid(blocks,blocks);
    // dim3 dimBlock(ids_width/blocks,ids_height/blocks);
    dim3 dimBlock((ids_width+blocks-1)/blocks,(ids_height+blocks-1)/blocks);
    renderObjectMapKernel<<<dimGrid,dimBlock>>>(ids,ids_width,ids_height,object_id_table,prob_width,prob_height,rendered_objects);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}