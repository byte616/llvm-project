; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; D1 (= 16) != L (= 32): the fast-path precondition fails, so the pass must
; fall back to the original flatten + pad rewrite.  Verify that we do *not*
; report dim-expansion and that the new global is a 1-D padded array.

target triple = "nvptx64-nvidia-cuda"

@arr = internal addrspace(3) global [64 x [16 x float]] undef

; CHECK: [Padding] Candidate L=32 N=1
; CHECK-NOT: [Transform] Dim-expanded (2D fast path)
; CHECK: [Transform] Flattened & padded:
; CHECK: @arr.padded = internal addrspace(3) global [{{[0-9]+}} x float]

define void @test_divrem_fold_inner16(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %x = shl i32 %tid, 1                                    ; x = 2 * tid
  %row = lshr i32 %x, 4                                   ; x / 16
  %col = and i32 %x, 15                                   ; x % 16
  %row64 = sext i32 %row to i64
  %col64 = sext i32 %col to i64
  %gep = getelementptr inbounds [64 x [16 x float]],
         ptr addrspace(3) @arr, i64 0, i64 %row64, i64 %col64
  %v = load float, ptr addrspace(3) %gep
  store float %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
