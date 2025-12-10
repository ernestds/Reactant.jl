#include <iostream>

// #include "mlir-c/IR.h"
#include "mlir-c/Support.h"

// #include "mlir/CAPI/IR.h"
// #include "mlir/CAPI/Wrap.h"
// #include "mlir/Pass/PassManager.h"

// #include "Enzyme/MLIR/Dialect/Dialect.h"
// #include "Enzyme/MLIR/Dialect/Ops.h"
// #include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
// #include "Enzyme/MLIR/Passes/Passes.h"

// #include "mlir/CAPI/Support.h"
// #include "mlir/Conversion/Passes.h"
// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Async/IR/Async.h"
// #include "mlir/Dialect/Complex/IR/Complex.h"
// #include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
// #include "mlir/Dialect/DLTI/DLTI.h"
// #include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/LLVMIR/NVVMDialect.h"
// #include "mlir/Dialect/Linalg/IR/Linalg.h"
// #include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
// #include "mlir/Dialect/Math/IR/Math.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/OpenMP/OpenMPDialect.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Dialect/Transform/Transforms/Passes.h"
// #include "mlir/InitAllPasses.h"
// #include "mlir/Parser/Parser.h"
// #include "mlir/Pass/PassRegistry.h"
// #include "mlir/Transforms/Passes.h"
// #include "src/enzyme_ad/jax/Dialect/Dialect.h"
// #include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"
// #include "src/enzyme_ad/jax/Passes/Passes.h"
// #include "src/enzyme_ad/jax/RegistryUtils.h"
// #include "llvm/Support/TargetSelect.h"

// #include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
// #include "stablehlo/dialect/ChloOps.h"
// #include "stablehlo/dialect/StablehloOps.h"
// #include "stablehlo/transforms/Passes.h"
// #include "stablehlo/transforms/optimization/Passes.h"

// #include "absl/log/globals.h"
// #include "absl/log/initialize.h"

// #include "xla/mlir/utils/type_util.h"
// #include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

// #include "tsl/platform/init_main.h"
// #include "tsl/profiler/lib/profiler_session.h"
// #include "tsl/profiler/lib/traceme.h"
// #include "xla/python/profiler_utils.h"
// #include "xla/tsl/profiler/rpc/client/capture_profile.h"
// #include "xla/tsl/profiler/rpc/profiler_server.h"

// #include "xla/python/ifrt/hlo/hlo_program.h"
// #include "llvm/ExecutionEngine/ExecutionEngine.h"
// #include "llvm/MC/TargetRegistry.h"
// #include "llvm/Support/Process.h"
// #include "llvm/TargetParser/Host.h"

// #include "llvm-c/TargetMachine.h"

// PJRT
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#if defined(REACTANT_CUDA) || defined(REACTANT_ROCM)
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#endif
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/pjrt/se_raw_buffer.h"

#include "xla/python/dlpack_types.h"
#include "xla/python/strides.h"

// #include "xla/hlo/ir/hlo_computation.h"
// #include "xla/hlo/ir/hlo_instruction.h"
// #include "xla/hlo/ir/hlo_instructions.h"
// #include "xla/hlo/ir/hlo_module.h"
// #include "xla/hlo/parser/hlo_parser.h"
// #include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
// #include "xla/hlo/translate/stablehlo.h"

// // CPU collectives
// #if defined(__linux__)
// #include "gloo/transport/tcp/attr.h"
// #include "gloo/transport/tcp/device.h"
// #include "xla/backends/cpu/collectives/gloo_collectives.h"
// #include "xla/backends/cpu/collectives/gloo_kv_store.h"
// #include "xla/backends/cpu/collectives/mpi_collectives.h"
// #elif defined(__APPLE__)
// #include "gloo/transport/uv/device.h"
// #include "xla/backends/cpu/collectives/gloo_collectives.h"
// #include "xla/backends/cpu/collectives/gloo_kv_store.h"
// #include "xla/backends/cpu/collectives/mpi_collectives.h"
// #endif // defined(__linux__)

// // shardy
// #include "shardy/dialect/sdy/ir/dialect.h"
// #include "shardy/dialect/sdy/transforms/export/passes.h"
// #include "shardy/dialect/sdy/transforms/import/passes.h"
// #include "shardy/dialect/sdy/transforms/passes.h"
// #include "shardy/dialect/sdy/transforms/propagation/passes.h"
// #include
// "shardy/dialect/sdy/transforms/propagation/user_priority_propagation.h"
// #include "shardy/integrations/c/attributes.h"
// #include "xla/pjrt/mlir_to_hlo.h"
// #include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
// #include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.h"
// #include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"

// // IFRT
// #include "xla/python/ifrt/array.h"
// #include "xla/python/ifrt/attribute_map.h"
// #include "xla/python/ifrt/client.h"
// #include "xla/python/ifrt/compiler.h"
// #include "xla/python/ifrt/device.h"
// #include "xla/python/ifrt/device_list.h"
// #include "xla/python/ifrt/dtype.h"
// #include "xla/python/ifrt/executable.h"
// #include "xla/python/ifrt/hlo/hlo_program.h"
// #include "xla/python/ifrt/host_callback.h"
// #include "xla/python/ifrt/index.h"
// #include "xla/python/ifrt/index_domain.h"
// #include "xla/python/ifrt/ir/ifrt_ir_program.h"
// #include "xla/python/ifrt/memory.h"
// #include "xla/python/ifrt/shape.h"
// #include "xla/python/ifrt/sharding.h"
// #include "xla/python/ifrt/topology.h"
// #include "xla/python/ifrt/tuple.h"
// #include "xla/python/ifrt/value.h"

// IFRT - PJRT
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_compiler.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pjrt_ifrt/pjrt_tuple.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"

// IFRT - Proxy (RPC)
#include "xla/python/ifrt_proxy/client/registry.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"

// // Cost Analysis
// #include "xla/hlo/ir/hlo_module.h"
// #include "xla/service/hlo_cost_analysis.h"

// #if defined(REACTANT_CUDA) || defined(REACTANT_ROCM)
// #include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
// #include "xla/service/gpu/model/gpu_performance_model.h"
// #include "xla/service/gpu/model/gpu_performance_model_base.h"
// #include "xla/stream_executor/device_description.h"
// #endif

// // Broken upstream x/ref https://github.com/jax-ml/jax/issues/33344
// // #include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

// #include "llvm/Support/ExtensibleRTTI.h"
// #include <llvm/Support/FileSystem.h>
// #include <llvm/Support/raw_ostream.h>

#define REACTANT_ABI extern "C" MLIR_CAPI_EXPORTED
void (*ReactantThrowError2)(const char *) = nullptr;

template <typename T> T MyValueOrThrow(absl::StatusOr<T> v) {
  if (!v.ok()) {
    ReactantThrowError2(v.status().ToString().c_str());
  }
  return std::move(v).value();
}

#include <dlpack/dlpack.h>
struct DLPackTensor {
  ~DLPackTensor();

  // `buffer_reference` is populated if we have shared (read-only) access.
  // nb::object buffer_reference;

  // `external_reference` is always populated.
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference;

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor tensor;
};

DLPackTensor::~DLPackTensor() {
  // We must release the external reference first before deleting the array.
  // external_reference.reset();
  std::cout << "deleting DLPackTensor\n";
  // if (buffer_reference) {
  //   GlobalPyRefManager()->AddGarbage(
  //       absl::MakeSpan(&buffer_reference, /*size=*/1));
  // }
}

void DLPackTensorDeleter(DLManagedTensor *t) {
  std::cout << "deleting DLManagedTensor\n";

  if (t) {
    delete static_cast<DLPackTensor *>(t->manager_ctx);
  }
}

absl::Status VerifyDType(const DLTensor &dl_tensor) {
  if (dl_tensor.dtype.bits % 8 != 0) {
    return xla::InvalidArgument(
        "Unsupported DLPack tensor dtype: bits should be a multiple of 8, got "
        "%d",
        dl_tensor.dtype.bits);
  }

  if (dl_tensor.dtype.lanes != 1) {
    return xla::InvalidArgument(
        "Unsupported DLPack tensor dtype: lanes should be equal to 1, got %d",
        dl_tensor.dtype.lanes);
  }

  return absl::OkStatus();
}

std::vector<int64_t> GetByteStrides(const DLTensor &dl_tensor) {
  // absl::StatusOr<std::vector<int64_t>> GetByteStrides(const DLTensor&
  // dl_tensor) { TF_RETURN_IF_ERROR(VerifyDType(dl_tensor));

  // Convert element strides from the number of elements to the number of bytes.
  std::vector<int64_t> strides;
  strides.reserve(dl_tensor.ndim);
  for (int i = 0; i < dl_tensor.ndim; ++i) {
    strides.push_back(dl_tensor.strides[i] * dl_tensor.dtype.bits / 8);
  }
  return strides;
}

REACTANT_ABI int64_t teste(int64_t t) { return t; }

// REACTANT_ABI xla::PjRtBuffer *MakePjrtBuffer(xla::PjRtDevice &device,
//   ::DLManagedTensor *dlmt,
//                const xla::Shape &shape,
//                xla::PrimitiveType element_type,
//                absl::Span<int64_t const> dimensions,
//                std::optional<bool> copy = std::nullopt,
//                std::optional<std::intptr_t> stream = std::nullopt) {
absl::StatusOr<std::vector<int64_t>>
StridesToLayout(absl::Span<int64_t const> dims,
                absl::Span<int64_t const> strides) {
  CHECK_EQ(dims.size(), strides.size());
  std::vector<int64_t> minor_to_major(dims.size());
  std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
  absl::c_sort(minor_to_major, [&](int a, int b) {
    if (strides[a] < strides[b]) {
      return true;
    }
    if (strides[a] > strides[b]) {
      return false;
    }
    // If two dimensions have the same stride, prefer the major-to-minor
    // interpretation of the ordering, since that's what JAX wants.
    return b < a;
  });
  int64_t stride = 1;
  for (int64_t d : minor_to_major) {
    if (dims[d] > 1 && strides[d] != stride) {
      return xla::Unimplemented(
          "Only DLPack tensors with trivial (compact) striding are supported; "
          "i.e., tensors whose striding represents a transposition of the "
          "underlying buffer but not broadcasting. Dimensions were: [%s], "
          "strides were [%s].",
          absl::StrJoin(dims, ","), absl::StrJoin(strides, ","));
    }
    stride *= dims[d];
  }
  return minor_to_major;
}
// device::Ptr{Nothing}, rt::Ref{DLPack.DLManagedTensor}, primitive_type::Int64,
// dim::Int64, cshape::Ptr{Int64}
REACTANT_ABI xla::PjRtBuffer *
MakePjrtBuffer(xla::PjRtDevice &device, ::DLManagedTensor *dlmt, uint64_t ptype,
               size_t dim, int64_t *cshape,
               std::optional<bool> copy = std::nullopt,
               std::optional<std::intptr_t> stream = std::nullopt) {

  auto element_type = (xla::PrimitiveType)ptype;

  absl::Span<const int64_t> dimensions(cshape, dim);

  std::function<void()> on_delete_callback;
  if (dlmt->deleter) {
    on_delete_callback = [dlmt]() {
      std::cout << "deleting" << "\n";
      dlmt->deleter(dlmt);
    };
  }

  bool has_custom_layout = dlmt->dl_tensor.strides != nullptr;
  std::vector<int64_t> minor_to_major;
  if (dlmt->dl_tensor.strides &&
      absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::Span<int64_t const> strides(
        reinterpret_cast<int64_t *>(dlmt->dl_tensor.strides),
        dlmt->dl_tensor.ndim);
    minor_to_major = MyValueOrThrow(StridesToLayout(dimensions, strides));
    // TF_ASSIGN_OR_RETURN(minor_to_major, StridesToLayout(dimensions,
    // strides));
  } else {
    minor_to_major.resize(dlmt->dl_tensor.ndim);
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      element_type, dimensions, minor_to_major);
  // xla::Shape shape(element_type, dimensions);
  void *data =
      static_cast<char *>(dlmt->dl_tensor.data) + dlmt->dl_tensor.byte_offset;

  // On CPU, creating a view may fail because of unaligned data buffer
  // in which case we'll fallback to copy. On non-CPU, array-api copy
  // semantics is handled in dlpack._place_array function.
  // copy.
  bool fallback_to_copy =
      !copy.has_value() && dlmt->dl_tensor.device.device_type == kDLCPU;
  std::cout << copy.has_value() << "AAAAAA" << copy.value_or(false) << "\n";
  std::cout << copy.has_value() << "AAAAAA" << copy.value_or(false) << "\n";
  // Create a view.
  if (!copy.value_or(false)) {
    std::cout << "here" << "\n";

    auto result = MyValueOrThrow(device.client()->CreateViewOfDeviceBuffer(
        data, shape, *device.default_memory_space(), on_delete_callback,
        stream));
    return (result).release();
  }
  std::cout << "2" << "\n";

  // Convert tensor strides (expressed in number of elements) to byte strides.
  std::optional<std::vector<int64_t>> byte_strides;
  if (dlmt->dl_tensor.strides) {
    // TF_ASSIGN_OR_RETURN(byte_strides, GetByteStrides(dlmt->dl_tensor));
    byte_strides = GetByteStrides(dlmt->dl_tensor);
  }

  // TF_ASSIGN_OR_RETURN(auto *memory_space, device.default_memory_space());
  auto *memory_space = MyValueOrThrow(device.default_memory_space());

  // Create a copy.
  return MyValueOrThrow(
             device.client()->BufferFromHostBuffer(
                 data, element_type, dimensions, byte_strides,
                 xla::PjRtClient::HostBufferSemantics::kMutableZeroCopy,
                 on_delete_callback, memory_space, /*device_layout=*/nullptr))
      .release();
}
// #include "Eigen/Core" 
#include "xla/backends/cpu/alignment.h"
REACTANT_ABI size_t CpuMinAlignment() {return xla::cpu::MinAlign();}
REACTANT_ABI xla::PjRtBuffer *
DataPointerToViewBuffer(void *data, xla::PjRtDevice *device, uint64_t dtype,
                        uint64_t *shape, uint64_t *minor_to_major,
                        uint64_t ndims, std::intptr_t stream_opt) {
  std::optional<std::intptr_t> stream;

  if (stream_opt != 0)
    stream = stream_opt; // has value
  else
    stream = std::nullopt;
  auto memory_space = *device->default_memory_space();

  auto cshape = (const int64_t *)shape;
  auto cminor_to_major = (const int64_t *)minor_to_major;
  absl::Span<const int64_t> span_cshape(cshape, ndims);
  absl::Span<const int64_t> span_minor_to_major(cminor_to_major, ndims);

  xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      (xla::PrimitiveType)dtype, span_cshape, span_minor_to_major);

  std::function<void()> on_delete_callback;
  on_delete_callback = []() {
    std::cout << "DEWLEGINTG\n";
  };
  std::cout << typeid(*device->client()).name() << "\n";

  auto result = MyValueOrThrow(device->client()->CreateViewOfDeviceBuffer(
      data, xla_shape, memory_space, on_delete_callback, stream));
  return (result).release();
}

REACTANT_ABI xla::PjRtBuffer *MutableZeroCopyBufferFromHostBuffer(xla::PjRtClient *client, void *data,
  uint64_t ptype, size_t dim,
  int64_t *cshape,
  xla::PjRtDevice *device) {
auto primtype = (xla::PrimitiveType)ptype;
absl::Span<const int64_t> shape(cshape, dim);
xla::PjRtClient::HostBufferSemantics semantics =
xla::PjRtClient::HostBufferSemantics::kMutableZeroCopy;
// xla::Layout layout(col_major(dim));
// auto buffer = xla::MyValueOrThrow(client->BufferFromHostBuffer(data,
// primtype, shape, /*byte_strides*/{},  semantics, /*ondone*/{}, device,
// &layout));
const xla::Layout *layout = nullptr;
auto buffer = MyValueOrThrow(client->BufferFromHostBuffer(
data, primtype, shape, /*byte_strides*/ {}, semantics, /*ondone*/ {},
*device->default_memory_space(), layout));
auto bres = buffer.release();
return bres;
}

REACTANT_ABI xla::PjRtBuffer *DLManagedTensorToBufferInternal(
    xla::PjRtDevice &device, ::DLManagedTensor *dlmt, bool copy,
    std::optional<std::intptr_t> _stream = std::nullopt) {
  std::optional<std::intptr_t> stream = std::nullopt;
  //   if (_stream) {
  //     stream = *_stream;
  // }
  // std::cout << "1" << "\n";

  auto element_type =
      MyValueOrThrow(xla::DLDataTypeToPrimitiveType(dlmt->dl_tensor.dtype));
  // std::cout << "2" << "\n";
  auto cshape = reinterpret_cast<int64_t *>(dlmt->dl_tensor.shape);
  // std::cout << "3" << "\n";
  int64_t dim = dlmt->dl_tensor.ndim;

  absl::Span<const int64_t> dimensions(cshape, dim);

  std::function<void()> on_delete_callback;
  if (dlmt->deleter) {

    on_delete_callback = [dlmt]() {
      // std::cout << "deleting" << "\n";
      // std::cout << "deleting buffer with deleter but remember to uncomment
      // the actual deleter part" << "\n"; this crashes when importing from
      // pytorch and jax? doesn't crash if we just let the DLManagedTensor
      // finalizer call it. dlmt->deleter(dlmt);
      //
      // std::cout << "deleter was executed from xla buffer destructor" << "\n";
    };
  } else {
    // crash if on_delete_callback is left unset
    on_delete_callback = [dlmt]() {
      // std::cout << "deleting buffer without any deleter" << "\n";
    };
  }

  // std::cout << "4" << "\n";
  // bool has_custom_layout = dlmt->dl_tensor.strides != nullptr;
  std::vector<int64_t> minor_to_major;
  if (dlmt->dl_tensor.strides &&
      absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::Span<int64_t const> strides(
        reinterpret_cast<int64_t *>(dlmt->dl_tensor.strides),
        dlmt->dl_tensor.ndim);
    minor_to_major = MyValueOrThrow(StridesToLayout(dimensions, strides));
    //   if (minor_to_major[0] == 0) {
    //     std::reverse(minor_to_major.begin(), minor_to_major.end());
    // }
  } else {

    minor_to_major.resize(dlmt->dl_tensor.ndim);
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      element_type, dimensions, minor_to_major);
  // std::cout << "5" << "\n";
  // xla::Shape shape(element_type, dimensions);
  void *data =
      static_cast<char *>(dlmt->dl_tensor.data) + dlmt->dl_tensor.byte_offset;

  // On CPU, creating a view may fail because of unaligned data buffer
  // in which case we'll fallback to copy. On non-CPU, array-api copy
  // semantics is handled in dlpack._place_array function.
  // copy.
  // bool fallback_to_copy =
  //     !copy.has_value() && dlmt->dl_tensor.device.device_type == kDLCPU;

  // Create a view.
  // std::cout << "copy "<< copy << "\n";
  // std::cout << "stream "<< stream << "\n";

  if (!copy || copy) {
    // std::cout << "here" << "\n";

    auto result = MyValueOrThrow(device.client()->CreateViewOfDeviceBuffer(
        data, shape, *device.default_memory_space(), on_delete_callback,
        stream));
    return (result).release();

    // if (!(result.status().code() == absl::StatusCode::kInvalidArgument &&
    //       fallback_to_copy)) {
    //   return MyValueOrThrow(result).release();
    // }
  }
  std::cout << "2" << "\n";

  // Convert tensor strides (expressed in number of elements) to byte strides.
  std::optional<std::vector<int64_t>> byte_strides;
  if (dlmt->dl_tensor.strides) {
    // TF_ASSIGN_OR_RETURN(byte_strides, GetByteStrides(dlmt->dl_tensor));
    byte_strides = GetByteStrides(dlmt->dl_tensor);
  }

  // TF_ASSIGN_OR_RETURN(auto *memory_space, device.default_memory_space());
  auto *memory_space = MyValueOrThrow(device.default_memory_space());

  // Create a copy.
  return MyValueOrThrow(
             device.client()->BufferFromHostBuffer(
                 data, element_type, dimensions, byte_strides,
                 xla::PjRtClient::HostBufferSemantics::kMutableZeroCopy,
                 {}, memory_space, /*device_layout=*/nullptr))
      .release();
}

// want from this:
// an external reference (and await buffers ready?), externel reference is
// needed so we can do: external_referece.reset() when we delete our "capsule"
// => return the deleter function
// an OpaqueDeviceMemoryDataPointer

// struct DLTensor
//     data::Ptr{Cvoid} => OpaqueDeviceMemoryDataPointer
//     ctx::DLDevice
//     ndim::Cint
//     dtype::DLDataType
//     shape::Ptr{Clonglong}
//     strides::Ptr{Clonglong}
//     byte_offset::Culonglong
// end
#include "xla/pjrt/pjrt_common.h"

absl::StatusOr<DLDeviceType>
DLDeviceTypeForDevice(const xla::PjRtDevice &device) {
  if (device.client()->platform_id() == xla::CpuId()) {
    return kDLCPU;
  } else if (device.client()->platform_id() == xla::CudaId()) {
    return kDLCUDA;
  } else if (device.client()->platform_id() == xla::RocmId()) {
    return kDLROCM;
  }
  return xla::InvalidArgument("Device %s cannot be used as a DLPack device.",
                              device.DebugString());
}
absl::StatusOr<DLDevice> DLDeviceForDevice(const xla::PjRtDevice &device) {
  DLDevice context;
  context.device_type = MyValueOrThrow(DLDeviceTypeForDevice(device));
  context.device_id = device.local_hardware_id().value();
  return context;
}

REACTANT_ABI std::intptr_t
GetBufferComputeStream(xla::PjRtBuffer *pjrt_buffer) {
  // absl::bit_cast<std::intptr_t>(
  //   se_stream->platform_specific_handle().stream)
  return absl::bit_cast<std::intptr_t>(
      MyValueOrThrow(tensorflow::down_cast<xla::PjRtStreamExecutorDevice *>(
                         pjrt_buffer->device())
                         ->GetLocalDeviceState())
          ->compute_stream()
          ->platform_specific_handle()
          .stream);
}

REACTANT_ABI void WaitUntilBufferReadyOnStream(xla::PjRtBuffer *pjrt_buffer,
                                               std::intptr_t stream) {
  static_cast<xla::CommonPjRtBuffer *>(pjrt_buffer)
      ->GetBufferWithHold(xla::CommonPjRtBuffer::ScopedHold::kUsage)
      .buffer()
      ->WaitUntilBufferReadyOnStream(stream);

  // auto se_client = static_cast<xla::PjRtStreamExecutorClient
  // *>(pjrt_buffer->device()->client()); auto thread_pool =
  // se_client->thread_pool(); TF_ASSIGN_OR_RETURN(definition_stream,
  //   local_device->GetStreamFromExternalStream(*stream));
  //   TF_RETURN_IF_ERROR(AllocateAndRecordEvent(definition_events.back(),
  //   local_device, definition_stream));
  // TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
  //   device->GetLocalDeviceState());
  // xla::PjRtStreamExecutorDevice* se_device =
  // tensorflow::down_cast<xla::PjRtStreamExecutorDevice*>(pjrt_buffer->device());

  // xla::LocalDeviceState* device_state =
  // MyValueOrThrow(se_device->GetLocalDeviceState());

  // auto local_device->GetStreamFromExternalStream(*stream)
  // absl::InlinedVector<BufferSequencingEventRef, 2> definition_events;
  // definition_events.emplace_back(
  //     BufferSequencingEvent::Create(this->thread_pool()));
  //   auto local_device_state =
  //   tensorflow::down_cast<xla::PjRtStreamExecutorDevice*>(pjrt_buffer->device())->GetLocalDeviceState();
  //   local_device_state.com
  // auto definition_event_or = device_state->GetEventForComputeStreamSyncPoint(
  //   device_state->GetNextComputeStreamSyncPoint(), se_client->thread_pool());
  //   auto definition_event = tsl::MakeRef<xla::PjRtStreamExecutorDeviceEvent>(
  //     *definition_event_or, "PjRtStreamExecutorLoadedExecutable", "Execute");
  //   static_cast<xla::CommonPjRtBuffer*>(pjrt_buffer)->GetBufferWithHold(xla::CommonPjRtBuffer::ScopedHold::kUsage).ConvertUsageHold(definition_event);
}

DLManagedTensor *
BufferToDLPackManagedTensorInternal(xla::PjRtBuffer *pjrt_buffer,
                                    std::optional<std::intptr_t> stream) {

  if (pjrt_buffer->IsTuple()) {
    ReactantThrowError2(
        "BufferToDLPackManagedTensor is not implemented for tuple buffers");
    // xla::Unimplemented(
    //     "BufferToDLPackManagedTensor is not implemented for tuple "
    //     "buffers.");
  }
  if (pjrt_buffer->has_dynamic_dimensions()) {
    ReactantThrowError2("DynamicShape is not implemented in DLPack");
  }
  // pjrt_buffer->data;
  auto pack = std::make_unique<DLPackTensor>();
  DLTensor &dt = pack->tensor.dl_tensor;
  {
    // pack->external_reference =
    // MyValueOrThrow(pjrt_buffer->AcquireExternalReference());
    // TF_ASSIGN_OR_RETURN(pack->external_reference,
    //                     pjrt_buffer->AcquireExternalReference());
    if (stream) {
      static_cast<xla::CommonPjRtBuffer *>(pjrt_buffer)
          ->GetBufferWithHold(xla::CommonPjRtBuffer::ScopedHold::kUsage)
          .buffer()
          ->WaitUntilBufferReadyOnStream(*stream);

    } else {

      // void CommonPjRtBuffer::WaitForOutstandingUsageHolds() ?? or sth like
      // that
    }
  }

  // pack->buffer_reference = nb::borrow<nb::object>(py_buffer);
  // pjrt_buffer->WaitUntilBufferReadyOnStream(stream);

  // std::cout << typeid(*pjrt_buffer).name() << "\n";
  dt.data = (void *)MyValueOrThrow(pjrt_buffer->client()->UnsafeBufferPointer(
      pjrt_buffer)); // PjRtClient::UnsafeBufferPointer which does contain a
                     // external reference? so why does the above results in
                     // error? => because the destructor of this ones external
                     // reference is called when unsafeptrfunction return
  // options: use this unsafebuffer
  // use the other one and delete memory reference of the dlpacked array and
  // create another dlpack array or sth, since this function here is pretty
  // cheap to execute ~500ns by using the UnsafeBufferPoiner, the original
  // pjrt_buffer might get destroyed in any buffer donation std::cout << "after
  // dt.data\n"; return (void *)unsafe; dt.data =
  // pack->external_reference->OpaqueDeviceMemoryDataPointer();
  pack->tensor.manager_ctx = pack.get();
  pack->tensor.deleter = DLPackTensorDeleter;
  dt.device = MyValueOrThrow(DLDeviceForDevice(*pjrt_buffer->device()));
  // TF_ASSIGN_OR_RETURN(dt.device, DLDeviceForDevice(*pjrt_buffer->device()));
  dt.device.device_id = pjrt_buffer->device()->local_hardware_id().value();
  dt.ndim = pjrt_buffer->dimensions().size();
  dt.dtype = MyValueOrThrow(
      xla::PrimitiveTypeToDLDataType(pjrt_buffer->element_type()));
  // TF_ASSIGN_OR_RETURN(dt.dtype,
  //                     PrimitiveTypeToDLDataType(pjrt_buffer->element_type()));

  pack->shape = std::vector<int64_t>(pjrt_buffer->dimensions().begin(),
                                     pjrt_buffer->dimensions().end());

  // TODO(b/327524065): use PjRtLayout directly instead of xla::Layout
  xla::Layout xla_layout = pjrt_buffer->layout()->xla_layout();
  pack->strides = xla::StridesForShape(pjrt_buffer->element_type(),
                                       pjrt_buffer->dimensions(), xla_layout);

  dt.shape = reinterpret_cast<std::int64_t *>(pack->shape.data());
  dt.strides = reinterpret_cast<std::int64_t *>(pack->strides.data());
  dt.byte_offset = 0;

  return &pack.release()->tensor;
}

REACTANT_ABI DLManagedTensor *
BufferToDLPackManagedTensor(xla::PjRtBuffer *pjrt_buffer,
                            std::optional<std::intptr_t> stream_opt) {
  std::optional<std::intptr_t> stream;
  if (stream_opt) {
    stream = *stream_opt;
  }
  return BufferToDLPackManagedTensorInternal(pjrt_buffer, stream);
}