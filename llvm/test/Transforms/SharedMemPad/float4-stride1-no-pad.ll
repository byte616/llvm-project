; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; float4 stride 1 (byte stride 16) should produce *no* bank conflict due to
; phase splitting: a warp is served in 4 phases x 8 threads, each phase has
; 8 threads x 4 consecutive banks = 32 distinct banks -> 1-way.
;
; Regression test: without phase splitting the model would (incorrectly)
; report 4-way conflict and try to pad.  With the new model the analysis
; reports 1-way and skips padding.

target triple = "nvptx64-nvidia-cuda"

@shared = internal addrspace(3) global [256 x float] undef

; CHECK: 1-way bank conflict
; CHECK: [Padding] No conflicting even strides. No padding needed.

; The original global must remain untouched (no .padded variant created).
; CHECK: @shared = internal addrspace(3) global [256 x float]
; CHECK-NOT: @shared.padded

define void @test_float4_stride1(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tid64 = sext i32 %tid to i64
  %idx = mul nsw i64 %tid64, 4
  %gep = getelementptr inbounds [256 x float], ptr addrspace(3) @shared, i64 0, i64 %idx
  %v = load <4 x float>, ptr addrspace(3) %gep
  store <4 x float> %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
