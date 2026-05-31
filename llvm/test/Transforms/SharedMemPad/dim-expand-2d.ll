; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Verify the 2-D dim-expand fast path.
;
; Shape is [32 x [32 x float]] with stride 2 -> bank conflict, picks L=32, N=1.
; Because D1 (= 32) equals L, the pass should bypass the flatten+pad rewrite
; and instead just resize the inner dimension to D1+N (= 33), keeping the
; original (row, col) GEP indices unchanged.

target triple = "nvptx64-nvidia-cuda"

@arr = internal addrspace(3) global [32 x [32 x float]] undef

; CHECK: [Transform] Dim-expanded (2D fast path): arr.padded [32 x [32+1 x T]]

; New global: 2-D, inner dim widened by N=1.
; CHECK: @arr.padded = internal addrspace(3) global [32 x [33 x float]] undef

; The rewritten GEP uses the *new* type with the *original* row/col indices --
; no flat/udiv/lshr arithmetic is emitted.
; CHECK-LABEL: define void @test_dim_expand_2d(
; CHECK: %[[GEP:.*]] = getelementptr inbounds [32 x [33 x float]], ptr addrspace(3) @arr.padded, i64 0, i64 %{{.*}}, i64 %{{.*}}
; CHECK: load float, ptr addrspace(3) %[[GEP]]
; CHECK-NOT: udiv
; CHECK-NOT: lshr i64 %{{.*}}, 5

define void @test_dim_expand_2d(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %x = shl i32 %tid, 1                                    ; x = 2 * tid
  %row = lshr i32 %x, 5                                   ; x / 32
  %col = and i32 %x, 31                                   ; x % 32
  %row64 = sext i32 %row to i64
  %col64 = sext i32 %col to i64
  %gep = getelementptr inbounds [32 x [32 x float]],
         ptr addrspace(3) @arr, i64 0, i64 %row64, i64 %col64
  %v = load float, ptr addrspace(3) %gep
  store float %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
