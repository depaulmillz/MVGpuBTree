/*
 *   Copyright 2022 The Regents of the University of California, Davis
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#include <cuda_profiler_api.h>
#include <gpu_btree.h>
#include <stdlib.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <cmd.hpp>
#include <cstdint>
#include <gpu_timer.hpp>
#include <numeric>
#include <random>
#include <rkg.hpp>
#include <string>
#include <unordered_set>
#include <validation.hpp>
#include <vector>
#include <chrono>

#include <device_bump_allocator.hpp>
#include <slab_alloc.hpp>



template <typename BTree,
          bool supportsVersioning,
          typename KeyT,
          typename ValueT>
void populate(BTree tree, KeyT key_range, size_t size, bool in_place) {

  std::default_random_engine gen;
  std::uniform_int_distribution<KeyT> dist(0, key_range - 1);
  if(dist.max() - dist.min() + 1 != key_range) {
    std::cerr << "Incorrect distribution of keys" << std::endl;
    exit(1);
  }
 
  const KeyT invalid_key     = std::numeric_limits<KeyT>::max();
  const ValueT invalid_value = std::numeric_limits<ValueT>::max();
  thrust::host_vector<KeyT> h_keys(size, invalid_key);
  thrust::host_vector<ValueT> h_values(size, invalid_value);


  for(size_t i = 0; i < size; ++i) {
    
    bool in_set = true;
    KeyT new_key;
    while(in_set) {
      new_key = dist(gen);
      in_set = false;
      for(size_t j = 0; j < i; ++j) {
        if(h_keys[j] == new_key) {
          in_set = true;
          break;
        }
      }
    }

    h_keys[i] = new_key;
    h_values[i] = 1;
  }
  
  thrust::device_vector<KeyT> d_keys = h_keys;
  thrust::device_vector<ValueT> d_values = h_values;

  if constexpr (supportsVersioning) {
    tree.insert(d_keys.data().get(),
                d_values.data().get(),
                size,
                0x0,
                in_place);
  } else {
    tree.insert(d_keys.data().get(), d_values.data().get(), size, 0x0);
  }
  cuda_try(cudaDeviceSynchronize());
}

struct bench_rates {
  float ops_rate;
};

template <typename BTree,
          bool supportsVersioning,
          typename KeyT,
          typename ValueT,
          typename PairT>
bench_rates bench_versioned(unsigned population,
                            unsigned insert_rate, 
                            unsigned remove_rate, 
                            unsigned range_query_rate, 
                            unsigned rq_size, 
                            size_t ops,
                            bool in_place, 
                            KeyT key_range) {
    
    //thrust::device_vector<KeyT> &d_keys, 
    //                        thrust::device_vector<ValueT> &d_values,
    //                        SizeT initial_tree_size,
    //                        thrust::device_vector<KeyT> &d_lower_bound,
    //                        thrust::device_vector<KeyT> &d_upper_bound,
    //                        thrust::device_vector<PairT> &d_results,
    //                        SizeT average_range_length,
    //                        std::vector<KeyT> &h_keys,
    //                        std::vector<KeyT> &h_lower_bound,
    //                        bool in_place,
    //                        bool validate_result,
    //                        bool validate_tree_structure,
    //                        const SetT &ref_set_v0,
    //                        const SetT &ref_set_v1,
    //                        std::size_t num_experiments,
    //                        Function0 &to_value,
    //                        Function1 &to_upper_bound) {


  const KeyT invalid_key     = std::numeric_limits<KeyT>::max();
  const ValueT invalid_value = std::numeric_limits<ValueT>::max();
  const PairT invalid_pair(invalid_key, invalid_value);

  BTree tree;
  populate<BTree, supportsVersioning, KeyT, ValueT>(tree, key_range, population, in_place);

  std::default_random_engine gen;
  std::uniform_int_distribution<> percent{0, 99}; // generate 0-99 inclusive
  std::uniform_int_distribution<KeyT> dist(0, key_range - 1); // generate in key range
  if(dist.max() - dist.min() + 1 != key_range) {
    std::cerr << "Incorrect distribution of keys" << std::endl;
    exit(1);
  }

  // create batches to execute
 
  thrust::host_vector<KeyT> h_insert_keys; 
  thrust::host_vector<KeyT> h_erase_keys; 
  thrust::host_vector<KeyT> h_get_keys; 
  thrust::host_vector<KeyT> h_lower_bound; 
  thrust::host_vector<KeyT> h_upper_bound; 

  for(size_t i = 0; i < ops; ++i) {
    int p = percent(gen);

    if(p < insert_rate) {
      // INSERT 
      h_insert_keys.push_back(dist(gen));
    } else if (p < insert_rate + remove_rate) {
      // REMOVE
      h_erase_keys.push_back(dist(gen));
    } else if (p < insert_rate + remove_rate + range_query_rate) {
      // RQ
      auto lower = dist(gen);
      h_lower_bound.push_back(lower);
      h_upper_bound.push_back(lower + rq_size - 1);
    } else {
      // GET
      h_get_keys.push_back(dist(gen));
    }
  } 

  cudaStream_t streamA{0};
  cudaStream_t streamB{0};

  cuda_try(cudaStreamCreate(&streamB));

  cuda_try(cudaProfilerStart());
  // now we execute our batches

  thrust::device_vector<KeyT> d_insert_keys = h_insert_keys; 
  thrust::device_vector<ValueT> d_insert_values(h_insert_keys.size(), 1); 
  thrust::device_vector<KeyT> d_erase_keys = h_erase_keys; 
  thrust::device_vector<KeyT> d_get_keys = h_get_keys; 
  thrust::device_vector<ValueT> d_get_values(h_get_keys.size(), invalid_value); 
  thrust::device_vector<KeyT> d_lower_bound = h_lower_bound; 
  thrust::device_vector<KeyT> d_upper_bound = h_upper_bound; 
  thrust::device_vector<PairT> d_range_results(h_lower_bound.size() * rq_size, invalid_pair);
  
  auto start = std::chrono::high_resolution_clock::now();
  tree.concurrent_insert_range(d_insert_keys.data().get(),
                               d_insert_values.data().get(),
                               d_insert_keys.size(),
                               d_lower_bound.data().get(),
                               d_upper_bound.data().get(),
                               d_lower_bound.size(),
                               d_range_results.data().get(),
                               rq_size,
                               streamA);

  tree.concurrent_find_erase(d_get_keys.data().get(),
                             d_get_values.data().get(),
                             d_get_keys.size(),
                             d_erase_keys.data().get(),
                             d_erase_keys.size(),
                             streamB);

  cuda_try(cudaStreamSynchronize(streamA));
  cuda_try(cudaStreamSynchronize(streamB));
  auto end = std::chrono::high_resolution_clock::now();

  cuda_try(cudaProfilerStop());
  std::cerr << ops / std::chrono::duration<double>(end - start).count() << std::endl;

  cuda_try(cudaStreamDestroy(streamB));

  return {ops / std::chrono::duration<double>(end - start).count() / 1e6};
}

int main(int argc, char **argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);

  uint32_t population =
      get_arg_value<uint32_t>(arguments, "population").value_or(500'000);
  uint32_t num_operations       = get_arg_value<uint32_t>(arguments, "ops").value_or(1'000'000);

  uint32_t key_range = get_arg_value<uint32_t>(arguments, "key-range").value_or(1'000'000);
  unsigned insert_rate = get_arg_value<uint32_t>(arguments, "inserts").value_or(5);
  unsigned range_query_rate = get_arg_value<uint32_t>(arguments, "range-queries").value_or(10);
  unsigned remove_rate = get_arg_value<uint32_t>(arguments, "removes").value_or(5);

  uint32_t average_range_length = get_arg_value<uint32_t>(arguments, "range-length").value_or(50);

  int device_id = get_arg_value<int>(arguments, "device").value_or(0);

  std::string output_dir = get_arg_value<std::string>(arguments, "output-dir").value_or("./");

  int device_count;
  cudaGetDeviceCount(&device_count);
  cudaDeviceProp devProp;
  if (device_id < device_count) {
    cudaSetDevice(device_id);
    cudaGetDeviceProperties(&devProp, device_id);
    std::cout << "Device[" << device_id << "]: " << devProp.name << std::endl;
  } else {
    std::cout << "No capable CUDA device found." << std::endl;
    std::terminate();
  }

  std::string device_name(devProp.name);
  std::replace(device_name.begin(), device_name.end(), ' ', '-');

  std::cout << "Benchmarking...\n";
  std::cout << std::boolalpha;
  std::cout << "key-range = " << key_range << ",\n";
  std::cout << "population = " << population << ",\n";
  std::cout << "num_operations = " << num_operations << ", ";
  std::cout << "insert_rate = " << insert_rate << ", ";
  std::cout << "range_query_rate = " << range_query_rate << ", ";
  std::cout << "remove_rate = " << remove_rate << "\n";
  std::cout << "get_rate = " << 100 - insert_rate - range_query_rate - remove_rate << ", ";
  std::cout << "range-length = " << average_range_length << "\n";

  std::cout << "------------------------\n";
  std::cout << "Generating input...\n";

  using key_type                 = uint32_t;
  using value_type               = uint32_t;
  using pair_type                = pair_type<key_type, value_type>;

  //unsigned seed = 0;
  //std::random_device rd;
  //std::mt19937_64 rng(seed);
  //// device vectors
  //auto d_keys   = thrust::device_vector<key_type>(num_keys, invalid_key);
  //auto d_values = thrust::device_vector<value_type>(num_keys, invalid_value);

  //auto d_range_lower = thrust::device_vector<key_type>(num_range_query, invalid_key);
  //auto d_range_upper = thrust::device_vector<key_type>(num_range_query, invalid_key);
  //auto d_range_results =
  //    thrust::device_vector<pair_type>(num_range_query * average_range_length, invalid_pair);

  //// host vectors
  //auto h_keys = rkg::generate_keys<key_type>(num_keys, rng, rkg::distribution_type::unique_random);
  //auto h_range_lower = std::vector<key_type>(num_range_query, invalid_key);

  //rkg::prep_experiment_range_query(h_keys, initial_tree_size, h_range_lower, num_range_query, rng);

  //// copy to device
  //d_keys        = h_keys;
  //d_range_lower = h_range_lower;

  //// assign values and upper bound
  //thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), to_value);
  //thrust::transform(thrust::device,
  //                  d_range_lower.begin(),
  //                  d_range_lower.end(),
  //                  d_range_upper.begin(),
  //                  to_upper_bound);

  //std::set<key_type> cpu_ref_set_v0;  // contains initial tree
  //std::set<key_type> cpu_ref_set_v1;  // contains inserted keys
  //if (validate_result) {
  //  std::cout << "Building CPU reference sets...\n";
  //  cpu_ref_set_v0.insert(h_keys.begin(), h_keys.begin() + initial_tree_size);
  //  cpu_ref_set_v1.insert(h_keys.begin() + initial_tree_size, h_keys.end());
  //}

  static constexpr int branching_factor = 16;
  using node_type           = GpuBTree::node_type<key_type, value_type, branching_factor>;
  using slab_allocator_type = device_allocator::SlabAllocLight<node_type, 8, 128 * 64, 16, 128>;
  using bump_allocator_type = device_bump_allocator<node_type>;

  using blink_tree_slab_type =
      GpuBTree::gpu_blink_tree<key_type, value_type, branching_factor, slab_allocator_type>;
  using blink_tree_bump_type =
      GpuBTree::gpu_blink_tree<key_type, value_type, branching_factor, bump_allocator_type>;

  using vblink_tree_slab_type =
      GpuBTree::gpu_versioned_btree<key_type, value_type, branching_factor, slab_allocator_type>;
  using vblink_tree_bump_type =
      GpuBTree::gpu_versioned_btree<key_type, value_type, branching_factor, bump_allocator_type>;

  //std::string report_dir = output_dir + '/' + device_name + "/versioned_insert_range/";
  //std::filesystem::create_directories(report_dir);

  //std::string filename = "rates_initial" + std::to_string(int(initial_tree_size / 1e6)) +
  //                       "M_update" + std::to_string(int(update_ratio * 100)) + "_range_length" +
  //                       std::to_string(average_range_length) + ".csv";
  //bool output_file_exist = std::filesystem::exists(report_dir + filename);
  //std::fstream result_output(report_dir + filename, std::ios::app);
  //if (!output_file_exist) {
  //  result_output << "initial_tree_size" << ',';
  //  result_output << "num_insertions" << ',';
  //  result_output << "num_range_query" << ',';
  //  result_output << "num_experiments" << ',';

  //  result_output << "vblink_slab_out_of_place_insert" << ',';
  //  result_output << "vblink_slab_out_of_place_concurrent_ops" << ',';

  //  result_output << "blink_slab_insert" << ',';
  //  result_output << "blink_slab_concurrent_ops" << ',';
  //  result_output << '\n';
  //}

  //result_output << initial_tree_size << ',';
  //result_output << num_insertions << ',';
  //result_output << num_range_query << ',';
  //result_output << num_experiments << ',';

  std::cout << "Running experiment...\n";
  {
    auto rates = bench_versioned<vblink_tree_slab_type, true, key_type, value_type, pair_type>(population, 
                            insert_rate, 
                            remove_rate, 
                            range_query_rate, 
                            average_range_length, 
                            num_operations,
                            true,
                            key_range);

    std::cout << "VBlink Tree: " << rates.ops_rate << std::endl;
  }

  {
    auto rates = bench_versioned<blink_tree_slab_type, false, key_type, value_type, pair_type>(population,
                            insert_rate, 
                            remove_rate, 
                            range_query_rate, 
                            average_range_length, 
                            num_operations, 
                            true,
                            key_range);


    std::cout << "Blink Tree: " << rates.ops_rate << std::endl;
  }

  //result_output << '\n';
}
