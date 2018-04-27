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

#ifndef TYPES_H_
#define TYPES_H_

#include <stddef.h>
#include <stdio.h>
#include <memory>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

typedef unsigned char* ImagePtr;
typedef unsigned short* DepthPtr;

struct ClassColour {
  ClassColour() 
  : name(""), r(0), g(0), b(0) {}
  ClassColour(std::string name_, int r_, int g_, int b_) 
  : name(name_), r(r_), g(g_), b(b_) {}
  std::string name;
  int r, g, b;
};

struct ObjectColour {
  ObjectColour() 
  : id(-1), r(0), g(0), b(0) {}
  ObjectColour(int id_, int r_, int g_, int b_) 
  : id(id_), r(r_), g(g_), b(b_) {}
  // std::string name;
  int id;
  int r, g, b;
};

struct MaskInfo{
  int mask_id;
  int class_id;
  float probability;
  int x1, y1, x2, y2;
  std::string mask_image_path;
  cv::Mat cv_mat;
};

struct FrameInfoMask {
  int64_t timestamp;
  std::string depth_path;
  std::string rgb_path;
  std::string depth_id;
  std::string rgb_id;
  bool labeled_frame;
  std::string frame_id;
  std::vector<MaskInfo> masks_;
  int num_masks;
};

#endif /* TYPES_H_ */
