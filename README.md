## Cloning

```bash
git lfs install

```


## Install CUDA  13.0
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13-0

echo 'export PATH=/usr/local/cuda-13.0/bin/:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Test
nvcc --version
```

```bash
sudo apt install -y python3-venv python3-dev build-essential
python3 -m venv gpu-env
source gpu-env/bin/activate
pip install --upgrade pip
pip install numpy pyyaml
pip install pycuda
```


## Deploy Aerial
### Requirements

Before running `run_aerial.sh`, the validation of available GPUs [in line 132](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/cuPHY-CP/container/run_aerial.sh#L132C1-L132C65) does not identify correctly in our environment. As it searches for string "3D controller: NVIDIA" and our environment shows "VGA compatible controller: NVIDIA". I changed the line to search for "NVIDIA" instead.

```bash
git clone https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git --recurse-submodules

cd aerial-cuda-accelerated-ran

# Optional (may be needed for large files)
git lfs install
git lfs pull

# Pull the Aerial container from NGC
docker pull nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb

# Start interactive development container
./cuPHY-CP/container/run_aerial.sh

# Inside container: Build SDK
./testBenches/phase4_test_scripts/build_aerial_sdk.sh
```

From another terminal check the running container name and extract the cuBB dir.

```bash
docker cp <Container name>:/opt/nvidia/cuBB cuBB

#safe to stop the aerial container now
cd cuBB
pip install hpccm
```

### pyAerial

```bash
export cuBB_SDK=`pwd`
AERIAL_BASE_IMAGE=<container image file> $cuBB_SDK/pyaerial/container/build.sh

$cuBB_SDK/pyaerial/container/run.sh
cd $cuBB_SDK
cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/devkit -DNVIPC_FMTLOG_ENABLE=OFF -DCMAKE_CUDA_ARCHITECTURES="86" -DASIM_CUPHY_SRS_OUTPUT_FP32=ON
cmake --build build -t _pycuphy pycuphycpp
./pyaerial/scripts/install_dev_pkg.sh
```


```bash
$ nvidia-smi --query-gpu=compute_cap --format=csv
    compute_cap
    8.6
```


# FlexRIC

```bash
  wget -qO- https://apt.repos.intel.com/oneapi/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg

  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

  sudo apt update
  sudo apt install intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic

  source /opt/intel/oneapi/compiler/2023.2.4/env/vars.sh
  source /opt/intel/oneapi/setvars.sh
  source /opt/intel/oneapi/ipp/2022.3/env/vars.sh
```

```C
#include <ipps.h>
```

Changed to:
```C
#include <ipp/ipps.h>
```

Rename `Is16vec16`, `I16vec16`, `I8vec32` and `I32vec8` to something else.

Run `make install -j` after `make -j` in FlexRAN installation tutorial.

### Agora install

Requires [Agora](https://github.com/Agora-wireless/Agora)

```bash
sudo apt install libarmadillo-dev libsoapysdr-dev libgflags-dev libgtest-dev python3-dev python3-numpy nlohmann-json3-dev

# Check if NUMPY is installed
python3 -c "import numpy; print(numpy.get_include())"
# Case true:
export CPATH=$(python3 -c "import numpy; print(numpy.get_include())"):$CPATH
```
Build Agora
```bash
cd Agora
mkdir build
cd build
cmake ..
make -j
```

Run new experiment:
```bash
./test_ldpc_experiment /home/william/repos/cuBB/pyaerial/notebooks/output/mcs_00_snr_00
```

## Run Experiment

### Generate dataset

In `line 577`  of Agora/CMakeLists.txt extends:

```MAKEFILE
set(LDPC_TESTS test_ldpc test_ldpc_mod test_ldpc_baseband)
```
to

```MAKEFILE
set(LDPC_TESTS test_ldpc test_ldpc_mod test_ldpc_baseband export_llr_dataset)
```


```bash
./build/export_llr_dataset   --conf_file=/home/william/repos/Agora/files/config/ci/tddconfig-sim-ul.json   --export_root=ldpc_llr_export   --num_codeblocks=10000   --mcs_start=0   --mcs_end=28   --snr_min_db=-5   --snr_max_db=25   --num_snr=30   --batch_codeblocks=512   --mother_code_rate=0.333333333333   --rate_match_rv=0   --rate_match_granularity_re=12

./build/benchmark_ldpc   --input_root=ldpc_llr_export/mcs_12/snr_20   --output_csv=flexran_ldpc_benchmark_mcs13_snr00.csv   --warmup_repeats=25   --repeats=100   --verify=true
```