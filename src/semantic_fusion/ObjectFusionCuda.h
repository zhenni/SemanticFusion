#include <cuda_runtime.h>
#include <utilities/MaskLogReader.h>


void updateObjectTable(int* deleted_ids, const int num_deleted, const int current_table_size,
                            float const* object_id_table, const int prob_width, const int prob_height, 
                          const int new_prob_width, float* new_object_id_table);

void fuseObjects(cudaTextureObject_t ids, const int ids_width, const int ids_height, const float* mask_probabilities,
                    const int x1, const int y1, const int box_width, const int box_height, const int obj_id, 
                    const int class_id, const float class_prob, float* object_id_table, const int map_size);
void renderObjectMap(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* object_id_table, const int prob_width, const int prob_height, 
                          float* rendered_objects);
