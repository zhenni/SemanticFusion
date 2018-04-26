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

#include "ObjectFusionInterface.h"
#include "SemanticFusionCuda.h"
#include "ObjectFusionCuda.h"
#include <utilities/Stopwatch.h>
#include <set>
#include <cmath>
#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <unordered_set>
#include <utility>

namespace std 
{
  template <class T>
  inline void my_hash_combine(std::size_t & seed, const T & v)
  {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2); 
  }

  template<typename S, typename T> struct hash<pair<S, T>> 
  {
    inline size_t operator()(const pair<S, T> & v) const
    {   
      size_t seed = 0;
      my_hash_combine(seed, v.first);
      my_hash_combine(seed, v.second);
      return seed;
    }   
  };  
}

// Basic algorithm for removing old items from probability table
template<typename T>
void remove_index(std::vector<T>& vector, const std::vector<int>& to_remove) {
  auto vector_base = vector.begin();
  typename std::vector<T>::size_type down_by = 0;
  for (auto iter = to_remove.cbegin(); 
       iter < to_remove.cend(); 
       iter++, down_by++)
  {
    typename std::vector<T>::size_type next = (iter + 1 == to_remove.cend() 
                                      ? vector.size() 
                                      : *(iter + 1));
    std::move(vector_base + *iter + 1, 
              vector_base + next, 
              vector_base + *iter - down_by);
  }
  vector.resize(vector.size() - to_remove.size());
}

void ObjectFusionInterface::CalculateProjectedProbabilityMap(const std::unique_ptr<ElasticFusionInterface>& map) {
  const int id_width = map->width(); 
  const int id_height = map->height();
  const int table_height = class_probabilities_gpu_->height(); // num_classes_
  const int table_width = class_probabilities_gpu_->width();  // max_components_
  renderProbabilityMap(map->GetSurfelIdsGpu(),id_width,id_height,
                       class_probabilities_gpu_->mutable_gpu_data(),
                       table_width,table_height,
                       rendered_class_probabilities_gpu_->mutable_gpu_data());
}
void ObjectFusionInterface::CalculateProjectedObjectMap(const std::unique_ptr<ElasticFusionInterface>& map){
  const int id_width = map->width(); 
  const int id_height = map->height();
  const int table_height = obj_ID_table_->height(); // num_classes_
  const int table_width = obj_ID_table_->width();  // max_components_
  renderObjectMap(map->GetSurfelIdsGpu(),id_width,id_height,
                       obj_ID_table_->mutable_gpu_data(),
                       table_width,table_height,
                       rendered_objects_gpu_->mutable_gpu_data());
}

std::shared_ptr<caffe::Blob<float> > ObjectFusionInterface::get_rendered_objects() {
  return rendered_objects_gpu_;
}

std::shared_ptr<caffe::Blob<float> > ObjectFusionInterface::get_rendered_probability() {
  return rendered_class_probabilities_gpu_;
}

std::shared_ptr<caffe::Blob<float> > ObjectFusionInterface::get_class_max_gpu() {
  return class_max_gpu_;
}

int ObjectFusionInterface::max_num_components() const {
  return max_components_;
}

// update the size of class_probabilities_gpu_, class_max_gpu_, according to the new global map from elastic_fusion
// the added surfel is initialized with uniqform probability and -1 class label 
void ObjectFusionInterface::UpdateProbabilityTable(const std::unique_ptr<ElasticFusionInterface>& map)
{
  const int new_table_width = map->GetMapSurfelCount();
  const int num_deleted = map->GetMapSurfelDeletedCount();
  printf("%i\n", num_deleted);
  const int table_width = class_probabilities_gpu_->width();  // max_components_
  const int table_height = class_probabilities_gpu_->height();  // num_classes_
  updateProbabilityTable(map->GetDeletedSurfelIdsGpu(),num_deleted,current_table_size_,
                    class_probabilities_gpu_->gpu_data(), table_width, table_height,
                    new_table_width, class_probabilities_gpu_buffer_->mutable_gpu_data(),
                    class_max_gpu_->gpu_data(),class_max_gpu_buffer_->mutable_gpu_data());
  // We then swap the pointers from the buffer to the other one
  class_probabilities_gpu_.swap(class_probabilities_gpu_buffer_);
  class_max_gpu_.swap(class_max_gpu_buffer_);
  current_table_size_ = new_table_width;
}


void ObjectFusionInterface::UpdateObjectTable(const std::unique_ptr<ElasticFusionInterface>& map)
{
  const int new_table_width = map->GetMapSurfelCount();
  const int num_deleted = map->GetMapSurfelDeletedCount();
  printf("num_deleted %i\n", num_deleted);
  printf("new_table_width %i\n", new_table_width);

  const int table_width = class_probabilities_gpu_->width();  // max_components_
  const int table_height = class_probabilities_gpu_->height();  // num_classes_
  updateObjectTable(map->GetDeletedSurfelIdsGpu(),num_deleted,current_table_size_,
                    obj_ID_table_->gpu_data(), table_width, table_height,
                    new_table_width, obj_ID_table_buffer_->mutable_gpu_data());
  // We then swap the pointers from the buffer to the other one
  obj_ID_table_.swap(obj_ID_table_buffer_);
  current_table_size_ = new_table_width;
}


// unused function?
int ObjectFusionInterface::UpdateSurfelProbabilities(const int surfel_id, 
                                                        const std::vector<float>& class_probs) 
{
  assert(static_cast<int>(class_probabilities_.size()) > surfel_id);
  std::vector<float>& surfel_probs = class_probabilities_[surfel_id];
  assert(static_cast<int>(class_probs.size()) == num_classes_);
  assert(static_cast<int>(surfel_probs.size()) == num_classes_);
  float normalisation_denominator = 0.0;
  for (int class_id = 0; class_id < num_classes_; class_id++) {
    surfel_probs[class_id] *= class_probs[class_id];
    normalisation_denominator += surfel_probs[class_id];
  }
  float max_prob = 0.0;
  int max_class = -1;
  for (int class_id = 0; class_id < num_classes_; class_id++) {
    surfel_probs[class_id] /= normalisation_denominator;
    if (surfel_probs[class_id] >= max_prob) {
      max_prob = surfel_probs[class_id];
      max_class = class_id;
    }
  }
  if (max_prob >= colour_threshold_) {
    return max_class;
  }
  return -1;
}

// update probability values in class_probabilities_gpu_, class_max_gpu_, according to new incoming segmentations
void ObjectFusionInterface::UpdateProbabilities(std::shared_ptr<caffe::Blob<float> > probs,
                                      const std::unique_ptr<ElasticFusionInterface>& map)
{
  CHECK_EQ(num_classes_,probs->channels());
  const int id_width = map->width();  //640
  // printf("id_width: %i\n", id_width);
  const int id_height = map->height();  //480
  // printf("id_height: %i\n", id_height);
  const int prob_width = probs->width();  //224
  // printf("prob_width: %i\n", prob_width);  
  const int prob_height = probs->height();  //224
  // printf("prob_height: %i\n", prob_height);  
  const int prob_channels = probs->channels();  //14
  // printf("prob_channels: %i\n", prob_channels);
  const int map_size = class_probabilities_gpu_->width();  //3000000
  // printf("map_size: %i\n", map_size);
  
  fuseSemanticProbabilities(map->GetSurfelIdsGpu(),id_width,id_height,probs->gpu_data(),
                    prob_width,prob_height,prob_channels,
                    class_probabilities_gpu_->mutable_gpu_data(),
                    class_max_gpu_->mutable_gpu_data(),map_size);
  map->UpdateSurfelClassGpu(map_size,class_max_gpu_->gpu_data(),class_max_gpu_->gpu_data() + map_size,colour_threshold_);
  
  // For Debug: get the max probability and class label
  // const float* max_prob = class_max_gpu_->cpu_data() + max_components_;
  // const float* max_class = class_max_gpu_->cpu_data();
  // float this_max_prob=-1;
  // float this_max_class=-1;
  // for (int id = 1 ;id < current_table_size_; id++){
  //             this_max_prob = std::max(this_max_prob, max_prob[id]);
  //             this_max_class = std::max(this_max_class, max_class[id]);
      
  // }
  // std::cout<<"max_prob"<<this_max_prob<<std::endl;
  // std::cout<<"max_class"<<this_max_class<<std::endl;
}


// update object ind in class_probabilities_gpu_, class_max_gpu_, according to new incoming instance masks
void ObjectFusionInterface::UpdateObjectIds(std::vector<MaskInfo>* masks, int num_masks,
                                      const std::unique_ptr<ElasticFusionInterface>& map)
{
  std::cout<< masks->size() << std::endl;
  std::cout<< num_masks << std::endl;

  CHECK_EQ(num_masks,masks->size());
  const int id_width = map->width();  //640
  printf("id_width: %i\n", id_width);
  const int id_height = map->height();  //480
  printf("id_height: %i\n", id_height);
  // const int prob_width = probs->width();  //224
  // // printf("prob_width: %i\n", prob_width);  
  // const int prob_height = probs->height();  //224
  // // printf("prob_height: %i\n", prob_height);  
  // const int prob_channels = probs->channels();  //14
  // // printf("prob_channels: %i\n", prob_channels);
  const int map_size = obj_ID_table_->width();  //3000000
  printf("map_size: %i\n", map_size);

  float class_prob;
  for(int m=0; m < num_masks; m++){
    std::cout<< "Fuse mask " << m << std::endl;
    const MaskInfo curMask = masks->at(m);
    const int x1=curMask.x1;
    const int x2=curMask.x2;
    const int y1=curMask.y1;
    const int y2=curMask.y2;
    const int box_width = curMask.x2 - curMask.x1;
    const int box_height = curMask.y2 - curMask.y1;
    printf("box_width: %i\n", box_width);
    printf("box_height: %i\n", box_height);
    const int obj_id = curMask.mask_id;
    const int class_id = curMask.class_id;
    const float class_prob = curMask.probability;
    cv::Mat mask_mat = curMask.cv_mat;
    
    // copy data from opencv Mat to Caffe blob
    caffe::Blob<float> mask_blob(1, box_height, box_width, 1);
    float* blob_data = mask_blob.mutable_cpu_data();

    for (int h = 0; h < box_height; ++h) {
      uchar* row_ptr = mask_mat.ptr<uchar>(h);
      for (int w = 0; w < box_width; ++w) {
        float prob = static_cast<float>(row_ptr[w])/255.0;
        blob_data[h*box_width+w] = prob;
        // surfel_id = tex2D<int>(ids_cpu,w+x1,h+y1);
        // std::cout<< surfel_id[] << ",";
      }
      // std::cout << std::endl;
    }

    mask_blob.Update();

    //==== check cpu surfelIds ====
    // const std::vector<int>& ids_cpu = map->GetSurfelIdsCpu();
    // std::cout << "!!!! print cpu surfel_id !!!!" << std::endl;
    // int surfel_id;
    // for (int i = 0; i < ids_cpu.size(); ++i){
    //   if (ids_cpu[i] != 0)
    //     std::cout << "ids_cpu: " << i << ": " <<  ids_cpu[i] << std::endl;
    // }

    // For debug
    // const float*  mask_blob_data = mask_blob.cpu_data();
    // for(int j = 0;j < box_height;j++){
    //   for(int i = 0;i < box_width;i++){
    //       float prob = mask_blob_data[box_width * j + i ] ;
    //       std::cout<<prob<<",";
    //   }
    //   std::cout<<std::endl;
    // }




    fuseObjects(map->GetSurfelIdsGpu(), id_width,id_height,mask_blob.gpu_data(),
                    x1, y1, box_width,box_height, obj_id, class_id, class_prob,
                    obj_ID_table_->mutable_gpu_data(),map_size);

    obj_ID_table_->Update();
    // For Debug: get the max probability and class label
    const float* max_prob = obj_ID_table_->cpu_data() + max_components_;
    const float* max_obj = obj_ID_table_->cpu_data();
    float this_max_prob=-1;
    float this_max_obj=-1;
    for (int id = 1 ;id < current_table_size_; id++){
        // std::cout<<"max_prob"<<this_max_prob<<std::endl;
        this_max_prob = std::max(this_max_prob, max_prob[id]);
        this_max_obj = std::max(this_max_obj, max_obj[id]);
    }
    std::cout<<"max_prob"<<this_max_prob<<std::endl;
    std::cout<<"max_obj"<<this_max_obj<<std::endl;


  }
  // map->UpdateSurfelClassGpu(map_size,class_max_gpu_->gpu_data(),class_max_gpu_->gpu_data() + map_size,colour_threshold_);


  // void copyMat2Blob(cv::Mat &mat, caffe::Blob<float>&){

  // }

  // const int prob_width = probs->width();  //224
  // // printf("prob_width: %i\n", prob_width);  
  // const int prob_height = probs->height();  //224
  // // printf("prob_height: %i\n", prob_height);  
  // const int prob_channels = probs->channels();  //14
  // // printf("prob_channels: %i\n", prob_channels);
  // const int map_size = class_probabilities_gpu_->width();  //3000000
  // // printf("map_size: %i\n", map_size);
  
  // fuseSemanticProbabilities(map->GetSurfelIdsGpu(),id_width,id_height,probs->gpu_data(),
  //                   prob_width,prob_height,prob_channels,
  //                   class_probabilities_gpu_->mutable_gpu_data(),
  //                   class_max_gpu_->mutable_gpu_data(),map_size);
  // map->UpdateSurfelClassGpu(map_size,class_max_gpu_->gpu_data(),class_max_gpu_->gpu_data() + map_size,colour_threshold_);
  
  // For Debug: get the max probability and class label
  // const float* max_prob = class_max_gpu_->cpu_data() + max_components_;
  // const float* max_class = class_max_gpu_->cpu_data();
  // float this_max_prob=-1;
  // float this_max_class=-1;
  // for (int id = 1 ;id < current_table_size_; id++){
  //             this_max_prob = std::max(this_max_prob, max_prob[id]);
  //             this_max_class = std::max(this_max_class, max_class[id]);
      
  // }
  // std::cout<<"max_prob"<<this_max_prob<<std::endl;
  // std::cout<<"max_class"<<this_max_class<<std::endl;
}




void ObjectFusionInterface::SaveArgMaxPredictions(std::string& filename,const std::unique_ptr<ElasticFusionInterface>& map) {
  const float* max_prob = class_max_gpu_->cpu_data() + max_components_;
  const float* max_class = class_max_gpu_->cpu_data();
  const std::vector<int>& surfel_ids = map->GetSurfelIdsCpu();
  cv::Mat argmax_image(240,320,CV_8UC3);
  for (int h = 0; h < 240; ++h) {
    for (int w = 0; w < 320; ++w) {
      float this_max_prob = 0.0;
      int this_max_class = 0;
      const int start = 0;
      const int end = 2;
      
      // As segmentation mask is 320x240 while the orginal image is 640x480, used the highest probability of the 2x2 patch
      // and corresponding class label for the pixel in the segmentation mask
      for (int x = start; x < end; ++x) {
        for (int y = start; y < end; ++y) {
          int id = surfel_ids[((h * 2) + y) * 640 + (w * 2 + x)];
          if (id > 0 && id < current_table_size_) {
            if (max_prob[id] > this_max_prob) {
              this_max_prob = max_prob[id];
              this_max_class = max_class[id];
            }
          }
        }
      }
      argmax_image.at<cv::Vec3b>(h,w)[0] = static_cast<int>(this_max_class);
      argmax_image.at<cv::Vec3b>(h,w)[1] =  static_cast<int>(this_max_class);
      argmax_image.at<cv::Vec3b>(h,w)[2] =  static_cast<int>(this_max_class);
    }
  }
  cv::imwrite(filename,argmax_image);
}
