; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Classic scalar (4B / float) stride-2 access -> 2-way bank conflict.
; Should be padded with L=32, N=1 in float units (size 256 + 8 = 264).

target triple = "nvptx64-nvidia-cuda"

@shared = internal addrspace(3) global [256 x float] undef

; CHECK: [Padding] Candidate L=32 N=1 (units=4B)
; CHECK: [Padding] >>> Recommended pad period: L=32 N=1
; CHECK: applying L=32 N=1 in ElemTy units (4B)

; CHECK: @shared.padded = internal addrspace(3) global [264 x float]

define void @test_scalar_stride2(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tid64 = sext i32 %tid to i64
  %idx = mul nsw i64 %tid64, 2
  %gep = getelementptr inbounds [256 x float], ptr addrspace(3) @shared, i64 0, i64 %idx
  %v = load float, ptr addrspace(3) %gep
  store float %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
