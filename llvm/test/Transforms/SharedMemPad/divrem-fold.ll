; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Test the (A udiv C) + and(A, C-1) GEP idiom recovery.
;
; Source pattern (typical CUDA code):
;   __shared__ float arr[32][32];
;   int x = 2 * tid;
;   ... = arr[x / 32][x % 32];
;
; After InstCombine x/32 -> lshr 5 and x%32 -> and 31.  Without the fold,
; the StrideVisitor sees an opaque udiv (-> UNKNOWN) and the pass bails out.
; With the fold the linear stride should be recovered as 2, producing a
; 2-way bank conflict and the standard L=32 N=1 padding.

target triple = "nvptx64-nvidia-cuda"

@arr = internal addrspace(3) global [32 x [32 x float]] undef

; CHECK: [Padding] Candidate L=32 N=1 (units=4B)
; CHECK: [Padding] >>> Recommended pad period: L=32 N=1
; D1 == L (=32), so the 2-D dim-expand fast path applies and produces
; [32 x [33 x float]] instead of a flat 1-D padded array.
; CHECK: [Transform] Dim-expanded (2D fast path): arr.padded [32 x [32+1 x T]]
; CHECK: @arr.padded = internal addrspace(3) global [32 x [33 x float]]

define void @test_divrem_fold(ptr %out) {
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
