/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "MaskLogReader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>


MaskLogReader::MaskLogReader(std::string file, std::string labels_file)
 : LogReader(file, true)
 , lastFrameTime(-1)
 , lastGot(-1)
 , has_depth_filled(false)
 , num_labelled(0)
{
	decompressionBufferDepth = new Bytef[Resolution::getInstance().numPixels() * 2];
	decompressionBufferDepthFilled = new Bytef[Resolution::getInstance().numPixels() * 2];
	decompressionBufferImage = new Bytef[Resolution::getInstance().numPixels() * 3];
  std::ifstream infile(file.c_str());
  std::string timestamp, depth_path, rgb_path, depth_id, rgb_id;
  std::map<std::string,int> depth_id_lookup;
  std::string scene_id = file.substr(file.rfind("/") + 1); // get file name
  std::string base_path = file;
  base_path.erase(base_path.rfind('/')); // get base path of file
  std::cout<<"Looking for RGB/Depth images in folder:"<<base_path<<std::endl;
  scene_id.erase(scene_id.length()-4);  // remove .txt of file name
  std::cout << scene_id << std::endl;
  std::string mask_path = base_path + "/" + scene_id + "/" + "masks";
  std::cout<<"Looking for masks in folder:"<<mask_path<<std::endl;

  int id = 0;
  std::string temp_mask_txt_file; //txt file contain masks for current frame
  while (infile >> timestamp >> depth_path >> rgb_path >> depth_id >> rgb_id) {
    FrameInfoMask frame_info;
    std::stringstream ss(timestamp.c_str());
    ss >> frame_info.timestamp;
    frame_info.depth_path = base_path + "/" + depth_path;
    if (id == 0) {
        std::cout<<"E.g.:"<<base_path+"/"+depth_path<<std::endl;
    }
    frame_info.depth_id = scene_id+"/"+depth_id;
    frame_info.rgb_path = base_path + "/" + rgb_path;
    frame_info.rgb_id = scene_id+"/"+rgb_id;
    frame_info.labeled_frame = false;
    depth_id_lookup[scene_id+"/"+depth_id] = id;
    
    temp_mask_txt_file = mask_path  + rgb_path.substr(rgb_path.rfind("/"), rgb_path.rfind(".")-rgb_path.rfind("/")) + ".txt"; 
    // std::cout << temp_mask_txt_file << std::endl;
    std::ifstream mask_file(temp_mask_txt_file.c_str());  // read files contain masks for current frame

    // read mask file for current frame which contains masks of detected object boxes
    float class_prob;
    int mask_id, class_id, x_min, y_min, x_max, y_max;
    std::string mask_image; 
    while(mask_file >> mask_id >> class_id >> class_prob >> x_min >> y_min >> x_max >> y_max >> mask_image){
      MaskInfo temp_mask_info;
      temp_mask_info.mask_id = mask_id;
      temp_mask_info.class_id = class_id;
      temp_mask_info.probability = class_prob;
      temp_mask_info.mask_image_file = mask_path + "/" + mask_image;
      frame_info.masks_.push_back(temp_mask_info);
      // std::cout << frame_info.masks_[mask_id].class_id <<frame_info.masks_[mask_id].probability<<frame_info.masks_[mask_id].mask_image_file << std::endl;
    }
    mask_file.close();

    frame_info.num_masks = mask_id+1;
    frames_.push_back(frame_info);
    id++;
  }
  infile.close();
  //Check if any frames are labelled frames according to the input text file
  std::ifstream inlabelfile(labels_file.c_str());
  std::string frame_id;
  while (inlabelfile >> depth_id >> rgb_id >> frame_id) {
    if (depth_id_lookup.find(depth_id) != depth_id_lookup.end()) {
      int found_id = depth_id_lookup[depth_id];
      frames_[found_id].labeled_frame = true;
      frames_[found_id].frame_id = frame_id;
      std::cout<<"Found:"<<frames_[found_id].depth_path<<std::endl;
      if (frames_[found_id].rgb_id != rgb_id) {
        std::cout<<"Warning, unaligned RGB and Depth frames - depth wins"<<std::endl;
      }
      num_labelled++;
    }
  }
  inlabelfile.close();
}

MaskLogReader::~MaskLogReader()
{
  delete [] decompressionBufferDepth;
  delete [] decompressionBufferDepthFilled;
  delete [] decompressionBufferImage;
}

// void MaskLogReader::getSegmentations(){

// }


void MaskLogReader::getNext()
{
  if ((lastGot + 1) < static_cast<int>(frames_.size())) {
    lastGot++;
    FrameInfoMask info = frames_[lastGot];
    timestamp = info.timestamp;

    // read rgb image
    cv::Mat rgb_image = cv::imread(info.rgb_path,CV_LOAD_IMAGE_COLOR);
    if (flipColors) {
      cv::cvtColor(rgb_image, rgb_image, CV_BGR2RGB); 
    }
    rgb = (unsigned char *)&decompressionBufferImage[0];
    int index = 0;
    for (int i = 0; i < rgb_image.rows; ++i) {
      for (int j = 0; j < rgb_image.cols; ++j) {
        rgb[index++] = rgb_image.at<cv::Vec3b>(i,j)[0];
        rgb[index++] = rgb_image.at<cv::Vec3b>(i,j)[1];
        rgb[index++] = rgb_image.at<cv::Vec3b>(i,j)[2];
      }
    }

    // read depth image
    depth = (unsigned short *)&decompressionBufferDepth[0];
    cv::Mat depth_image = cv::imread(info.depth_path,CV_LOAD_IMAGE_ANYDEPTH);
    index = 0;
    for (int i = 0; i < depth_image.rows; ++i) {
      for (int j = 0; j < depth_image.cols; ++j) {
        depth[index++] = depth_image.at<uint16_t>(i,j);
      }
    }
    
    // read filled depth image
    depthfilled = (unsigned short *)&decompressionBufferDepthFilled[0];
    std::string depth_filled_str = info.depth_path;
    depth_filled_str.erase(depth_filled_str.end()-9,depth_filled_str.end());
    depth_filled_str += "depthfilled.png";
    cv::Mat depthfill_image = cv::imread(depth_filled_str,CV_LOAD_IMAGE_ANYDEPTH);
    if (depthfill_image.data) {
      index = 0;
      for (int i = 0; i < depthfill_image.rows; ++i) {
        for (int j = 0; j < depthfill_image.cols; ++j) {
          depthfilled[index++] = depthfill_image.at<uint16_t>(i,j);
        }
      }
      has_depth_filled = true;
    } else {
      has_depth_filled = false;
    }

    //read mask images
    cvMasks.clear();
    for(int mask_id =0; mask_id < info.num_masks; mask_id++){
      std::string mask_image_path = info.masks_[mask_id].mask_image_file;
      cv::Mat mask_image = cv::imread(mask_image_path,CV_LOAD_IMAGE_ANYDEPTH);
      std::cout<< mask_image.type() << std::endl;
      info.masks_[].push_back(&mask_image);
    }

    imageSize = Resolution::getInstance().numPixels() * 3;
    depthSize = Resolution::getInstance().numPixels() * 2;
  }
}

bool MaskLogReader::isLabeledFrame()
{
    return frames_[lastGot].labeled_frame;
}

std::string MaskLogReader::getLabelFrameId() {
  if (isLabeledFrame()) {
    return frames_[lastGot].frame_id;
  }
  return "";
}

int MaskLogReader::getNumMasks() {
    return frames_[lastGot].num_masks;
}

int MaskLogReader::getNumFrames()
{
    return static_cast<int>(frames_.size());
}

bool MaskLogReader::hasMore()
{
    return (lastGot + 1) < static_cast<int>(frames_.size());
}

std::vector<cv::Mat *> MaskLogReader::getMasks(){
  return cvMasks;
}