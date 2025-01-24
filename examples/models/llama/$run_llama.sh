#!/bin/bash

llama3_2_1b=Llama-3.2-1B-Instruct
llama3_2_3b=Llama-3.2-3B-Instruct
llama3_1_8b=meta-llama-3.1-8b-instruct

model=${MODEL:-$llama3_2_1b}
models_dir=${MODELS_DIR:-../models}
model_type=${MODEL_TYPE:-llama3_2}

model_path=$models_dir/$model
checkpoint=$model_path/original/consolidated.00.pth
tokenizer=$model_path/original/tokenizer.model
params=$model_path/original/params.json
output_name="${model}_kv_sdpa_xnn_qe_4_32.pte"


export_model=false
build_runner=false
n_threads=-1
while [[ $# -gt 0 ]]; do
  case $key in
    --export)
      export_model=true
      shift 1
      ;;
    --build)
      build_runner=true
      shift 1
      ;;
    --threads)
      n_threads=$2
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done


if [[ "$export_model" == "true" ]]; then
  printf "\n\n>> Export $model_path to $output_name ...\n"

  set -x
  python -m examples.models.llama.export_llama \
    --model $model_type \
    --checkpoint "$checkpoint" \
    -p "$params" \
    -kv \
    --use_sdpa_with_kv_cache \
    -X \
    -qmode 8da4w \
    --group_size 128 \
    -d fp16 \
    --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
    --embedding-quantize 4,32 \
    --output_name="$output_name"
  set +x
fi


if [[ "$build_runner" == "true" ]]; then
  printf "\n\n>> Build executorch with optimized CPU performance ...\n\n"
  set -x
  cmake -DPYTHON_EXECUTABLE=python \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DEXECUTORCH_ENABLE_LOGGING=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
      -Bcmake-out .
  cmake --build cmake-out -j16 --target install --config Release
  set +x

  printf "\n\n>> Build llama runner ...\n\n"
  set -x
  cmake -DPYTHON_EXECUTABLE=python \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
      -Bcmake-out/examples/models/llama \
      examples/models/llama
  cmake --build cmake-out/examples/models/llama -j16 --config Release
  set +x
fi

# Run model
printf "\n\n>> Run model\n\n"
set -x
cmake-out/examples/models/llama/llama_main \
  --model_path=$output_name \
  --tokenizer_path=$tokenizer \
  --prompt="You are a helpful AI assistant that will help to" \
  --cpu_threads=$n_threads
set +x
