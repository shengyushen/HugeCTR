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


#include <fstream>
#include "HugeCTR/include/parser.hpp"
#include "HugeCTR/include/session.hpp"

#ifdef ENABLE_MPI
#include <mpi.h>

#define CK_MPI_THROW__(cmd)                                                                        \
  do {                                                                                             \
    auto retval = (cmd);                                                                           \
    if (retval != MPI_SUCCESS) {                                                                   \
      throw std::runtime_error(std::string("MPI Runtime error: ") + std::to_string(retval) + " " + \
                               __FILE__ + ":" + std::to_string(__LINE__) + " \n");                 \
    }                                                                                              \
  } while (0)

#endif

static const std::string simple_help =
    "usage: huge_ctr.exe [--train] [--model-init] [--help] [--version] config_file.json\n";

enum class CmdOptions_t { Train, ModelInit, Version, Help };

int main(int argc, char* argv[]) {
  try {
    int numprocs = 1, pid = 0;
		//SSY can only use 1 GPU
    cudaSetDevice(0);
#ifdef ENABLE_MPI
    int provided;
    // CK_MPI_THROW__(MPI_Init(&argc, &argv));
    CK_MPI_THROW__(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
    CK_MPI_THROW__(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    CK_MPI_THROW__(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif
    const std::map<std::string, CmdOptions_t> CMD_OPTIONS_TYPE_MAP = {
        {"--train", CmdOptions_t::Train},
        {"--model-init", CmdOptions_t::ModelInit},
        {"--help", CmdOptions_t::Help},
        {"--version", CmdOptions_t::Version}};

    if (argc != 3 && argc != 2 && pid == 0) {
      std::cout << simple_help;
      return -1;
    }

    CmdOptions_t opt = CmdOptions_t::Help;
    if (!HugeCTR::find_item_in_map(&opt, std::string(argv[1]), CMD_OPTIONS_TYPE_MAP) && pid == 0) {
      std::cerr << "wrong option: " << argv[1] << std::endl;
      std::cerr << simple_help;
      return -1;
    }

    switch (opt) {
      case CmdOptions_t::Help: {
        if (pid == 0) {
          std::cout << simple_help;
        }
        break;
      }
      case CmdOptions_t::Version: {
        if (pid == 0) {
          std::cout << "HugeCTR Version: " << HUGECTR_VERSION_MAJOR << "." << HUGECTR_VERSION_MINOR
                    << std::endl;
        }
        break;
      }
      case CmdOptions_t::ModelInit: {
				// SSY where model init happen
        if (argc != 3 && pid == 0) {
          std::cerr << "expect config file." << std::endl;
          std::cerr << simple_help;
          return -1;
        }
        std::string config_file(argv[2]);
        if (pid == 0) {
          std::cout << "Config file: " << config_file << std::endl;
        }
        HugeCTR::SolverParser solver_config(config_file);
        HugeCTR::Session session_instance(solver_config.batchsize, config_file,
                                          *solver_config.device_map);
        // SSY : generate the model from json
        // HugeCTR/src/session.cpp
        session_instance.init_params(solver_config.model_file);
        break;
      }
      case CmdOptions_t::Train: {
        if (argc != 3 && pid == 0) {
          std::cerr << "expect config file." << std::endl;
          std::cerr << simple_help;
          return -1;
        }

        std::string config_file(argv[2]);
        if (pid == 0) {
          std::cout << "Config file: " << config_file << std::endl;
        }
        HugeCTR::SolverParser solver_config(config_file);
	// SSY HugeCTR/include/session.hpp
	// SSY creating the whole pipeline with data_reader, embedding and mlp
        HugeCTR::Session session_instance(solver_config.batchsize, solver_config.model_file,
                                          solver_config.embedding_file, config_file,
                                          *solver_config.device_map);

        HugeCTR::Timer timer;
        timer.start();
        // train
        if (pid == 0) {
          std::cout << "HugeCTR training start:" << std::endl;
        }
        for (int i = 0; i < solver_config.max_iter; i++) {
          //if (i%100==0) 
	  //        std::cout << "main iteration " << i << std::endl;
          // SSY HugeCTR/src/session.cpp
          session_instance.train();
          if (i % solver_config.display == 0 && i != 0) {
            timer.stop();
            // display
            float loss = 0;
            session_instance.get_current_loss(&loss);
            if (pid == 0) {
              MESSAGE_("Iter: " + std::to_string(i) + " Time(" +
                       std::to_string(solver_config.display) + " iters): " +
                       std::to_string(timer.elapsedSeconds()) + "s Loss: " + std::to_string(loss));
            }
            timer.start();
          }
          if (i % solver_config.snapshot == 0 && i != 0) {
            // snapshot
            std::string snapshot_dense_name =
                solver_config.snapshot_prefix + "_dense_" + std::to_string(i) + ".model";
            std::string snapshot_sparse_name =
                solver_config.snapshot_prefix + "_sparse_" + std::to_string(i) + ".model";
            session_instance.download_params_to_file(snapshot_dense_name, snapshot_sparse_name);
          }
          if (solver_config.eval_interval > 0 && i % solver_config.eval_interval == 0 && i != 0) {
            float avg_loss = 0.f;
            for (int j = 0; j < solver_config.eval_batches; ++j) {
              session_instance.eval();
              float tmp_loss = 0.f;
              session_instance.get_current_loss(&tmp_loss);
              avg_loss += tmp_loss;
            }
            if (solver_config.eval_batches > 0) {
              avg_loss /= solver_config.eval_batches;
            }
            if (pid == 0) {
              MESSAGE_("Evaluation, average loss: " + std::to_string(avg_loss));
            }
          }
        }
        break;
      }
      default: { assert(!"Error: no such option && should never get here!"); }
    }
#ifdef ENABLE_MPI
    CK_MPI_THROW__(MPI_Finalize());
#endif
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    std::cerr << "Terminated with error\n";
  }

  return 0;
}
