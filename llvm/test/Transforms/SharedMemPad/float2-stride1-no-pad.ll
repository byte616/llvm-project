; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; float2 stride 1 (byte stride 8) -> phase-splitting model: 2 phases of 16
; threads, each phase covers 16 threads x 2 banks = 32 distinct banks -> 1-way.
; No padding should be applied.

target triple = "nvptx64-nvidia-cuda"

@shared = internal addrspace(3) global [256 x float] undef

; CHECK: 1-way bank conflict
; CHECK: [Padding] No conflicting even strides. No padding needed.
; CHECK-NOT: @shared.padded

define void @test_float2_stride1(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tid64 = sext i32 %tid to i64
  %idx = mul nsw i64 %tid64, 2
  %gep = getelementptr inbounds [256 x float], ptr addrspace(3) @shared, i64 0, i64 %idx
  %v = load <2 x float>, ptr addrspace(3) %gep
  store <2 x float> %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
