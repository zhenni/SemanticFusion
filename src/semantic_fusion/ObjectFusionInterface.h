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

#ifndef OBJECT_FUSION_INTERFACE_H_
#define OBJECT_FUSION_INTERFACE_H_
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

// This is just to get Blobs for now
#include <cnn_interface/CaffeInterface.h>
#include <map_interface/ElasticFusionInterface.h>
#include <cassert>

#include "CRF/densecrf.h"
#include <utilities/MaskLogReader.h>
#include <cuda_runtime.h>

struct sceneObject{
  int class_id;
  float class_prob;
  std::vector<int> surfel_ids;
};

class ObjectFusionInterface {
public:
  ObjectFusionInterface(const int num_classes, const int prior_sample_size, 
                          const int max_components = 3000000, const float colour_threshold = 0.0)
    : current_table_size_(0)
    , num_classes_(num_classes) 
    , prior_sample_size_(prior_sample_size)
    , max_components_(max_components)
    , colour_threshold_(colour_threshold)
  { 
    // This table contains for each component (surfel) the probability of
    // it being associated with each class
    class_probabilities_gpu_.reset(new caffe::Blob<float>(1,1,num_classes_,max_components_));
    class_probabilities_gpu_buffer_.reset(new caffe::Blob<float>(1,1,num_classes_,max_components_));
    // This contains two rows - one is the max class (if none then negative) the
    // other is the probability
    class_max_gpu_.reset(new caffe::Blob<float>(1,1,3,max_components_));
    class_max_gpu_buffer_.reset(new caffe::Blob<float>(1,1,3,max_components_));
    rendered_class_probabilities_gpu_.reset(new caffe::Blob<float>(1,num_classes_,640,480));
    rendered_objects_gpu_.reset(new caffe::Blob<float>(1,1,640,480));

    obj_ID_table_.reset(new caffe::Blob<float>(1,1,3,max_components_));
    obj_ID_table_buffer_.reset(new caffe::Blob<float>(1,1,3,max_components_));

  }
  virtual ~ObjectFusionInterface() {}

  int UpdateSurfelProbabilities(const int surfel_id, const std::vector<float>& class_probs);
  void UpdateProbabilities(std::shared_ptr<caffe::Blob<float> > probs,const std::unique_ptr<ElasticFusionInterface>& map);
  void UpdateProbabilityTable(const std::unique_ptr<ElasticFusionInterface>& map);
  void CalculateProjectedProbabilityMap(const std::unique_ptr<ElasticFusionInterface>& map);

  void SaveArgMaxPredictions(std::string& filename,const std::unique_ptr<ElasticFusionInterface>& map);
  std::shared_ptr<caffe::Blob<float> > get_rendered_probability();
  std::shared_ptr<caffe::Blob<float> > get_class_max_gpu();
  int max_num_components() const;

  void UpdateObjectTable(const std::unique_ptr<ElasticFusionInterface>& map);
  void UpdateObjectIds(std::vector<MaskInfo>* masks, int num_masks, const std::unique_ptr<ElasticFusionInterface>& map);
  void CalculateProjectedObjectMap(const std::unique_ptr<ElasticFusionInterface>& map);

  std::shared_ptr<caffe::Blob<float> > get_rendered_objects();

private:

  // Returns negative if the class is below the threshold - otherwise returns the class
  std::vector<std::vector<float> > class_probabilities_;
  std::shared_ptr<caffe::Blob<float> > class_probabilities_gpu_;
  int current_table_size_;
  // This is used to store the table swap after updating
  std::shared_ptr<caffe::Blob<float> > class_probabilities_gpu_buffer_;
  std::shared_ptr<caffe::Blob<float> > class_max_gpu_;
  std::shared_ptr<caffe::Blob<float> > class_max_gpu_buffer_;
  // This stores the rendered probabilities of surfels from the map
  std::shared_ptr<caffe::Blob<float> > rendered_class_probabilities_gpu_;


  // Table stores object ID of surfels
  std::vector<sceneObject> objects;
  std::shared_ptr<caffe::Blob<float> > obj_ID_table_;
  std::shared_ptr<caffe::Blob<float> > obj_ID_table_buffer_;
  // This stores the rendered objects of surfels from the map
  std::shared_ptr<caffe::Blob<float> > rendered_objects_gpu_;

  const int num_classes_;
  const int prior_sample_size_;
  const int max_components_;
  const float colour_threshold_;
};

#endif /* OBJECT_FUSION_INTERFACE_H_ */
