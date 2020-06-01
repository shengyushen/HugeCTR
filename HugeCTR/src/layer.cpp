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


#include "HugeCTR/include/layer.hpp"

#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

void Layer::init_params(std::ofstream& out_stream) {
	//SSY HugeCTR/include/layer.hpp  return empty vector
  std::vector<float> initializer = std::move(get_initializer());
  if (initializer.empty()) return;

  size_t size_in_byte = initializer.size() * sizeof(float);
  out_stream.write(reinterpret_cast<char*>(&initializer.front()), size_in_byte);
}

Layer::~Layer() {
  try {
    int o_device = -1;
    CK_CUDA_THROW_(get_set_device(get_device_id(), &o_device));
    for (auto weight : weights_) {
      delete weight;
    }
    for (auto wgrad : wgrad_) {
      delete wgrad;
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

}  // namespace HugeCTR
