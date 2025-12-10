curl
wget

curl -fsSL https://install.julialang.org | sh
juliaup add 1.11
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
mv bazelisk-linux-amd64 /usr/local/bin/bazelisk
export DEBIAN_FRONTEND=noninteractive
apt-get install -y bzip2 ca-certificates clang cmake curl earlyoom file g++ gcc git gnupg gzip jq lld make python3 python3-numpy tar xz-utils zip zstd

julia --project -e 'using Pkg; Pkg.instantiate()' # needed only the first time to install dependencies for this script
LD_LIBRARY_PATH=/usr/local/cuda/compat/:$LD_LIBRARY_PATH julia -O0 --color=yes --project build_local.jl --backend=cuda
cp /workspaces/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so ../libReactantExtra.so