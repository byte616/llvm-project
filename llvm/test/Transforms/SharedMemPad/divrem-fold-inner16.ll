; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Verifies the div/rem fold works for a non-square 2D array where the
; inner dimension (= modulus C) is 16 instead of 32.
;
;   __shared__ float arr[64][16];
;   int x = 2 * tid;
;   ... = arr[x / 16][x % 16];
;
; Flat index = (x/16)*16 + (x%16) = x = 2*tid (1D), expect 2-way bank
; conflict and L=lcm(2, 32)=32, N=1.

target triple = "nvptx64-nvidia-cuda"

@arr = internal addrspace(3) global [64 x [16 x float]] undef

; CHECK: Load Tx:2 Ty:0 (2-way bank conflict)
; CHECK: [Padding] Candidate L=32 N=1 (units=4B)
; CHECK: [Padding] >>> Recommended pad period: L=32 N=1
; CHECK: @arr.padded = internal addrspace(3) global [{{[0-9]+}} x float]

define void @test_divrem_fold_inner16(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %x = shl i32 %tid, 1
  %row = lshr i32 %x, 4         ; x / 16
  %col = and i32 %x, 15         ; x % 16
  %row64 = zext i32 %row to i64
  %col64 = zext i32 %col to i64
  %gep = getelementptr inbounds [64 x [16 x float]],
         ptr addrspace(3) @arr, i64 0, i64 %row64, i64 %col64
  %v = load float, ptr addrspace(3) %gep
  store float %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
