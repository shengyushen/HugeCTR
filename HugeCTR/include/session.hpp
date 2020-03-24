/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <thread>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_reader_multi_threads.hpp"
#include "HugeCTR/include/device_map.hpp"
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/network.hpp"
#include "HugeCTR/include/parser.hpp"
#include "ctpl/ctpl_stl.h"

namespace HugeCTR {
/**
 * @brief A simple facade of HugeCTR.
 *
 * This is a class supporting basic usages of hugectr, which includes
 * train; evaluation; get loss; load and download trained parameters.
 * To learn how to use those method, please refer to main.cpp.
 */
class Session {
 private:
  typedef long long TypeKey;               /**< type of input key in dataset. */
  std::vector<Network*> networks_;         /**< networks (dense) used in training. */
    // SSY HugeCTR/include/embeddings/sparse_embedding_hash.cuh
    // HugeCTR/include/embeddings/sparse_embedding_hash.hpp
  Embedding<TypeKey>* embedding_{nullptr}; /**< embedding */
  DataReader<TypeKey>* data_reader_; /**< data reader to reading data from data set to embedding. */
  DataReader<TypeKey>* data_reader_eval_; /**< data reader for evaluation. */
  Parser* parser_;                        /***< model parser */
  GPUResourceGroup gpu_resource_group_;   /**< GPU resources include handles and streams etc.*/
 public:
  /**
   * Ctor of Session.
   * @param batch_size will be used in the following training.
   * @param json_name the json file of configuration.
   * @param device_map a index list of the devices which be used in this training.
   */
  Session(int batch_size, const std::string& json_name, const DeviceMap& device_map);

  /**
   * A method loading trained parameters of both dense and sparse model.
   * @param model_file dense model generated by training
   * @param embedding_file sparse model generated by training
   */
  Error_t load_params(const std::string& model_file, const std::string& embedding_file);

  /**
   * Ctor of Session, which construct and load parameters.
   * @param batch_size will be used in the following training.
   * @param model_file dense model generated by training
   * @param embedding_file sparse model generated by training
   * @param json_name the json file of configuration.
   * @param device_map a index list of the devices which be used in this training.
   */
  Session(int batch_size, const std::string& model_file, const std::string& embedding_file,
          const std::string& json_name, const DeviceMap& device_map)
      : Session(batch_size, json_name, device_map) {
    load_params(model_file, embedding_file);
  }
  /**
   * Dtor of Session.
   */
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /**
   * The all in one training method.
   * This method processes one iteration of a training, including one forward, one backward and
   * parameter update
   */
  Error_t train();
  /**
   * The all in one evaluation method.
   * This method processes one forward of evaluation.
   */
  Error_t eval();
  /**
   * Get current loss from the loss tensor.
   * @return loss in float
   */
  Error_t get_current_loss(float* loss);
  /**
   * Download trained parameters to file.
   * @param weights_file file name of output dense model
   * @param embedding_file file name of output sparse model
   */
  Error_t download_params_to_file(std::string weights_file, std::string embedding_file);
  /**
   * Set learning rate while training
   * @param lr learning rate.
   */
  Error_t set_learning_rate(float lr) {
    for (auto network : networks_) {
      network->set_learning_rate(lr);
    }
    return Error_t::Success;
  }
  /**
   * generate a dense model and initilize with small random values.
   * @param model_file dense model initilized
   */
  Error_t init_params(std::string model_file);
  /**
   * get the number of parameters (reserved for debug)
   */
  long long get_params_num() {
    return static_cast<long long>(networks_[0]->get_params_num()) + embedding_->get_params_num();
  }
};

}  // namespace HugeCTR
