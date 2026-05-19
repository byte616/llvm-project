; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Same div/rem idiom as divrem-fold.ll but with an *odd* stride s=3.
; The fold should still recover the linear stride 3, but because 3 is
; coprime with 32 there is no bank conflict and no padding should be
; emitted (no @arr.padded global, no [Padding] Candidate line).

target triple = "nvptx64-nvidia-cuda"

@arr = internal addrspace(3) global [32 x [32 x float]] undef

; CHECK: Load Tx:3 Ty:0 (1-way bank conflict)
; CHECK-NOT: [Padding] Candidate
; CHECK: [Padding] No conflicting even strides. No padding needed.
; CHECK-NOT: @arr.padded

define void @test_divrem_fold_s3_odd(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %x = mul i32 %tid, 3
  %row = lshr i32 %x, 5
  %col = and i32 %x, 31
  %row64 = zext i32 %row to i64
  %col64 = zext i32 %col to i64
  %gep = getelementptr inbounds [32 x [32 x float]],
         ptr addrspace(3) @arr, i64 0, i64 %row64, i64 %col64
  %v = load float, ptr addrspace(3) %gep
  store float %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
