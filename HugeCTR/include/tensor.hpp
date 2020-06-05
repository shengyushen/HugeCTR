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

#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

/**
 * @brief A simple class to implement Tensor in network, for example: gradients, parameters, blobs.
 *
 * Note: before GeneralBuffer::init() no memory will be allocated and the content
 * cannot be accessed. dims = {.. third dimension, second dimension, leading dimension}
 * same order as TensorFormat_t.
 */
template <typename T>
class Tensor {
 private:
  std::vector<int>
      dims_; /**< Dimensions of tensor, and the last element is the leading dimension */
	//SSY GenenralBuffers have built in device
  GeneralBuffer<T>& buff_; /**< GeneralBuffer used in this tensor (the real memory allocator) */
  const TensorFormat_t format_; /**< Format of the tensor */
  const size_t mem_offset_;     /**< An internal used offset to generate pointer of GPU memory */
 public:
  /**
   * Ctor.
   * @param dims dimensions of tensor, and the last element is the leading dimension.
   * @param buffer GeneralBuffer used in this tensor (the real memory allocator).
   * @param format Format of the tensor.
   */
  Tensor(const std::vector<int>& dims, GeneralBuffer<T>& buffer,
         TensorFormat_t format = TensorFormat_t::WH)
      : dims_(dims),
        buff_(buffer), //SSY the GeneralBuffer from outside?
        format_(format),
        mem_offset_(buffer.reserve(get_size_from_dims(dims))) { // SSY add new buffer space according to this tensor shape!!!
    static_assert(std::is_same<T, float>::value || std::is_same<T, long long>::value ||
                      std::is_same<T, unsigned int>::value,
                  "type not support");
    try {
      // verify dims == 2
      if (format_ != TensorFormat_t::WH && format_ != TensorFormat_t::HW && dims_.size() == 2) {
        CK_THROW_(Error_t::WrongInput, "input dims doesn't match format");
      }
      // verify dims == 3
      if (format_ != TensorFormat_t::HSW && dims_.size() == 3) {
        CK_THROW_(Error_t::WrongInput, "input dims doesn't match format");
      }
      if (dims_.size() != 2 && dims_.size() != 3) {
        CK_THROW_(Error_t::WrongInput, "doesn't support dims != 2 and != 3");
      }
      for (auto iter = dims_.begin(); iter < dims_.end(); iter++) {
        if (iter[0] <= 0)
          CK_THROW_(Error_t::WrongInput, "dims vector cannot have 0 or smaller elements");
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  /**
   * Ctor.
   * To construct a new Tensor but share the same memory (content) of input tensor,
   * and just reshape. Note that any modification to the content of this tensor will
   * modify the input tensor too.
   * @param new_dims new dimensions (must has the same product as input tensor).
   * @param C the input tensor or prototype.
   * @param new_format the new format.
   */
  Tensor(const std::vector<int>& new_dims, const Tensor& C, TensorFormat_t new_format)
      : dims_(new_dims), buff_(C.buff_), format_(new_format), mem_offset_(C.mem_offset_) {
    try {
      if (format_ != TensorFormat_t::WH && format_ != TensorFormat_t::HW && dims_.size() == 2) {
        CK_THROW_(Error_t::WrongInput, "input dims doesn't match format");
      }
      // verify dims == 3
      if (format_ != TensorFormat_t::HSW && dims_.size() == 3) {
        CK_THROW_(Error_t::WrongInput, "input dims doesn't match format");
      }
      if (dims_.size() != 2 && dims_.size() != 3) {
        CK_THROW_(Error_t::WrongInput, "doesn't support dims != 2 and != 3");
      }
      int d = 1, _d = 1;
      for (auto dim : dims_) {
        d *= dim;
        if (dim <= 0)
          CK_THROW_(Error_t::WrongInput, "dims vector cannot have 0 or smaller elements");
      }
      for (auto dim : C.dims_) {
        _d *= dim;
      }
      if (d != _d) {
        CK_THROW_(Error_t::WrongInput, "new_dims should match the input Tensor");
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }
  typedef T TYPE;
  int get_device_id() const { return buff_.get_device_id(); }
  T* get_ptr() const { return buff_.get_ptr_with_offset(mem_offset_); }
  std::vector<int> get_dims() const { return dims_; }
  size_t get_num_elements() const {
    size_t tensor_size = 1;
    for (auto dim : dims_) {
      tensor_size *= dim;
    }
    return tensor_size;
  }
  size_t get_size() const { return get_num_elements() * sizeof(T); }
  TensorFormat_t get_format() const { return format_; }
};

/**
 * To print the tensor from begin to end.
 * When both begin and end are positive numbers: print the begin->end elements.
 * When both of them are negtive numbers: print the last elements:
 * @verbatim
 *  begin_ = get_size_from_dims(tensor.get_dims()) + begin;
 *  end_ = get_size_from_dims(tensor.get_dims()) + end;
 * @endverbatim
 */
template <typename T>
bool print_tensor(const Tensor<T>& tensor, int begin, int end) {
  int begin_;
  int end_;
  if (begin >= 0 && end <= (int)get_size_from_dims(tensor.get_dims()) && end > begin) {
    begin_ = begin;
    end_ = end;
  } else if (end < 0 && -begin <= (int)get_size_from_dims(tensor.get_dims()) && end > begin) {
    begin_ = get_size_from_dims(tensor.get_dims()) + begin;
    end_ = get_size_from_dims(tensor.get_dims()) + end;
  } else {
    return false;
  }
  int odevice = -1;
  get_set_device(tensor.get_device_id(), &odevice);
  cudaDeviceSynchronize();
  assert(end_ > begin_ && begin_ >= 0 && end_ < get_size_from_dims(tensor.get_dims()));
  T host_buff[end_ - begin_];
  cudaMemcpy(host_buff, tensor.get_ptr() + begin_, (end_ - begin_) * sizeof(T),
             cudaMemcpyDeviceToHost);
  std::cout << "Tensor: <";
  for (auto d : tensor.get_dims()) {
    std::cout << d << ",";
  }
#ifdef ENABLE_MPI
  int pid(-1), num_procs(-1);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  std::cout << "> in pid: " << pid << " of " << num_procs << " processes." << std::endl;
#else
  std::cout << ">" << std::endl;
#endif
  std::cout << "begin: " << begin_ << " end: " << end_ << std::endl;
  for (int i = 0; i < end_ - begin_; i++) {
    std::cout << host_buff[i] << ",";
  }
  std::cout << std::endl;
  get_set_device(odevice);
  return true;
}

}  // namespace HugeCTR
