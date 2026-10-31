"""Independent CUDA VMM twin-allocation helper for prepare-time ILC experiments.

One VA is remapped between ordinary/compressible physical allocations only
after context synchronization. Tensor views retain shared ownership. This
module imports no GPU package; the builder embeds these source constants.
"""

CPP=r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void ilc_check(CUresult rc,const char* call) {
  if(rc!=CUDA_SUCCESS) {
    const char *name=nullptr,*text=nullptr;
    cuGetErrorName(rc,&name);cuGetErrorString(rc,&text);
    throw std::runtime_error(std::string(call)+": "+(name?name:"unknown")+" "+(text?text:""));
  }
}
struct IlcContext {
  explicit IlcContext(CUcontext context){ilc_check(cuCtxPushCurrent(context),"cuCtxPushCurrent");}
  ~IlcContext(){CUcontext old;cuCtxPopCurrent(&old);}
};
struct IlcArena {
  CUcontext context=nullptr;
  CUdevice device=0;
  CUdeviceptr address=0;
  CUmemGenericAllocationHandle handle[2]={0,0};
  size_t logical=0,physical=0,granularity=0;
  int mapped=-1;
  int actual_flag[2]={-1,-1};
  ~IlcArena() {
    if(!context)return;
    if(cuCtxPushCurrent(context)!=CUDA_SUCCESS)return;
    // Destruction is outside timed calls. Never unmap a running graph.
    cuCtxSynchronize();
    if(address && mapped>=0)cuMemUnmap(address,physical);
    if(address)cuMemAddressFree(address,physical);
    for(int i=0;i<2;++i)if(handle[i])cuMemRelease(handle[i]);
    CUcontext old;cuCtxPopCurrent(&old);
  }
};
using IlcOwner=std::shared_ptr<IlcArena>;
static IlcOwner ilc_owner(const pybind11::capsule& cap) {
  const char* name=cap.name();
  TORCH_CHECK(name && std::string(name)=="muse_ilc_arena","wrong ILC capsule");
  return *static_cast<IlcOwner*>(cap.get_pointer());
}
std::vector<int64_t> ilc_capabilities() {
  CUdevice d;ilc_check(cuCtxGetDevice(&d),"cuCtxGetDevice");
  int vmm=0,compression=0,major=0,minor=0;
  ilc_check(cuDeviceGetAttribute(&vmm,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,d),"VMM attribute");
  ilc_check(cuDeviceGetAttribute(&compression,CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,d),"compression attribute");
  ilc_check(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,d),"major attribute");
  ilc_check(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,d),"minor attribute");
  return {(int64_t)d,vmm,compression,major,minor};
}
pybind11::capsule ilc_create(int64_t bytes) {
  TORCH_CHECK(bytes>0 && bytes<(1LL<<60),"invalid ILC byte count");
  auto a=std::make_shared<IlcArena>();a->logical=(size_t)bytes;
  ilc_check(cuCtxGetCurrent(&a->context),"cuCtxGetCurrent");
  TORCH_CHECK(a->context,"no current CUDA context");
  ilc_check(cuCtxGetDevice(&a->device),"cuCtxGetDevice");
  CUmemAllocationProp props[2]={};size_t g[2]={};
  for(int i=0;i<2;++i) {
    props[i].type=CU_MEM_ALLOCATION_TYPE_PINNED;
    props[i].location.type=CU_MEM_LOCATION_TYPE_DEVICE;
    props[i].location.id=a->device;
    props[i].allocFlags.compressionType=i?CU_MEM_ALLOCATION_COMP_GENERIC:CU_MEM_ALLOCATION_COMP_NONE;
    ilc_check(cuMemGetAllocationGranularity(&g[i],&props[i],CU_MEM_ALLOC_GRANULARITY_MINIMUM),"cuMemGetAllocationGranularity");
  }
  TORCH_CHECK(g[0] && g[1],"zero VMM granularity");
  a->granularity=std::lcm(g[0],g[1]);
  a->physical=((a->logical-1)/a->granularity+1)*a->granularity;
  for(int i=0;i<2;++i) {
    ilc_check(cuMemCreate(&a->handle[i],a->physical,&props[i],0),"cuMemCreate");
    CUmemAllocationProp actual={};
    ilc_check(cuMemGetAllocationPropertiesFromHandle(&actual,a->handle[i]),"cuMemGetAllocationPropertiesFromHandle");
    a->actual_flag[i]=actual.allocFlags.compressionType;
  }
  ilc_check(cuMemAddressReserve(&a->address,a->physical,a->granularity,0,0),"cuMemAddressReserve");
  auto holder=new IlcOwner(a);
  return pybind11::capsule(holder,"muse_ilc_arena",[](void* p){delete static_cast<IlcOwner*>(p);});
}
void ilc_map(pybind11::capsule cap,int64_t mode) {
  auto a=ilc_owner(cap);TORCH_CHECK(mode==0 || mode==1,"bad ILC mode");
  IlcContext context(a->context);
  ilc_check(cuCtxSynchronize(),"cuCtxSynchronize before remap");
  if(a->mapped==mode)return;
  if(a->mapped>=0){ilc_check(cuMemUnmap(a->address,a->physical),"cuMemUnmap");a->mapped=-1;}
  ilc_check(cuMemMap(a->address,a->physical,0,a->handle[mode],0),"cuMemMap");
  a->mapped=(int)mode;
  CUmemAccessDesc access={};access.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id=a->device;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ilc_check(cuMemSetAccess(a->address,a->physical,&access,1),"cuMemSetAccess");
}
at::Tensor ilc_view(pybind11::capsule cap) {
  auto a=ilc_owner(cap);TORCH_CHECK(a->mapped>=0,"ILC view is unmapped");
  IlcContext context(a->context);
  return at::from_blob(reinterpret_cast<void*>(a->address),{(int64_t)a->logical},
      [a](void*){},at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA,a->device)));
}
std::vector<int64_t> ilc_info(pybind11::capsule cap) {
  auto a=ilc_owner(cap);
  return {(int64_t)a->address,(int64_t)a->logical,(int64_t)a->physical,
          (int64_t)a->granularity,a->actual_flag[0],a->actual_flag[1],a->mapped};
}
void ilc_copy(at::Tensor destination,at::Tensor source);
'''

CUDA=r'''
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
__global__ void ilc_copy_bytes_kernel(uint8_t* out,const uint8_t* in,int64_t bytes) {
  int64_t vector=blockIdx.x*blockDim.x+threadIdx.x;
  int64_t vectors=bytes/16;
  for(int64_t i=vector;i<vectors;i+=(int64_t)blockDim.x*gridDim.x)
    reinterpret_cast<uint4*>(out)[i]=reinterpret_cast<const uint4*>(in)[i];
  // Up to15tail bytes, no speculative read/write beyond the tensor.
  if(vector<bytes%16)out[vectors*16+vector]=in[vectors*16+vector];
}
void ilc_copy(at::Tensor destination,at::Tensor source) {
  TORCH_CHECK(source.is_cuda() && destination.is_cuda(),"ILC copy requires CUDA tensors");
  TORCH_CHECK(source.device()==destination.device(),"ILC copy device mismatch");
  TORCH_CHECK(source.is_contiguous() && destination.is_contiguous(),"ILC copy requires contiguous tensors");
  int64_t bytes=source.numel()*source.element_size();
  TORCH_CHECK(destination.numel()*destination.element_size()==bytes,"ILC byte size mismatch");
  TORCH_CHECK((uintptr_t)source.data_ptr()%16==0 && (uintptr_t)destination.data_ptr()%16==0,"unaligned ILC copy");
  int64_t blocks=(bytes+4095)/4096;blocks=blocks<1?1:(blocks>132*8?132*8:blocks);
  ilc_copy_bytes_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
      (uint8_t*)destination.data_ptr(),(const uint8_t*)source.data_ptr(),bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
