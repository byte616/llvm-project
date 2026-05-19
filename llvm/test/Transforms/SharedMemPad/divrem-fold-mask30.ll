; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Regression test for the InstCombine-narrowed mask form.
;
; clang -O2 compiles
;     int x   = 6 * tid;
;     int row = x / 32;
;     int col = x % 32;
;     sh[row][col] = ...
; into LLVM IR that uses `and i32 %x, 30` (not 31), because 6 is even and
; therefore the low bit of 6*tid is known zero.  SCEV in turn represents
; the rem term as
;     8 * (zext i4 (3 * (trunc i32 %tid to i4)) to i64)
; which is *not* pointer-equal to what `getURemExpr(6*zext(tid), 32)`
; returns in its canonical form.  The fold has to enumerate the
; gcd-factored variant (g=2, j=4) for the match to succeed.

target triple = "nvptx64-nvidia-cuda"

@arr = internal addrspace(3) global [32 x [32 x float]] undef

; CHECK: Load Tx:6 Ty:0 (2-way bank conflict)
; CHECK: [Padding] Candidate L=96 N=1 (units=4B)
; CHECK: [Padding] >>> Recommended pad period: L=96 N=1
; CHECK: @arr.padded = internal addrspace(3) global [{{[0-9]+}} x float]

define void @test_divrem_fold_mask30(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %x = mul nuw nsw i32 %tid, 6
  %row = lshr i32 %x, 5
  %col = and i32 %x, 30                ; <- narrowed by InstCombine
  %row64 = zext nneg i32 %row to i64
  %col64 = zext nneg i32 %col to i64
  %gep = getelementptr inbounds [32 x [32 x float]],
         ptr addrspace(3) @arr, i64 0, i64 %row64, i64 %col64
  %v = load float, ptr addrspace(3) %gep
  store float %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
