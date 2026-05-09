; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Scalar stride-2 access (2-way bank conflict) wrapped in a loop.
; FlatIdx = tid*2 + i is an affine recurrence in %i, so the pass should
; rewrite it to a phi-based IV named with the "flat.iv" prefix.

target triple = "nvptx64-nvidia-cuda"

@shared = internal addrspace(3) global [256 x float] undef

; CHECK: [Padding] Candidate L=32 N=1 (units=4B)
; CHECK: [Padding] >>> Recommended pad period: L=32 N=1
; CHECK: @shared.padded = internal addrspace(3) global [264 x float]

; The rewritten loop header should carry a phi whose name starts with
; "flat.iv" (emitted by SCEVExpander).
; CHECK-LABEL: define void @test_flat_iv_in_loop
; CHECK: loop:
; CHECK: phi i64{{.*}}%flat.iv

define void @test_flat_iv_in_loop(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tid64 = sext i32 %tid to i64
  %stride2 = mul nsw i64 %tid64, 2
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %idx = add nsw i64 %stride2, %i
  %gep = getelementptr inbounds [256 x float], ptr addrspace(3) @shared, i64 0, i64 %idx
  %v = load float, ptr addrspace(3) %gep
  %out.gep = getelementptr inbounds float, ptr %out, i64 %i
  store float %v, ptr %out.gep
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, 16
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
