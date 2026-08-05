; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s
;
; Verify that a 64-column 2-D shared access whose row/col arithmetic was
; lowered into power-of-two lshr/and/select patterns is still recognized as
; a linear stride-4 access across warp lanes.
;
; CHECK: Load Tx:4 Ty:0 (4-way bank conflict)
; CHECK: [Padding] >>> Recommended pad period: L=32 N=1

target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@arr = internal addrspace(3) global [64 x [64 x float]] undef, align 4

define void @test_divrem_fold_s6(ptr %output) {
entry:
  %tid = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lane = and i32 %tid, 31
  %row.cmp = icmp ugt i32 %lane, 15
  %mul = shl nuw nsw i32 %lane, 2
  %rem.sub = add nsw i32 %mul, -64
  %rem.cmp = icmp ult i32 %lane, 16
  %col = select i1 %rem.cmp, i32 %mul, i32 %rem.sub
  %row = zext i1 %row.cmp to i64
  %col64 = zext i32 %col to i64
  %gep = getelementptr inbounds nuw [64 x [64 x float]], ptr addrspace(3) @arr, i64 0, i64 %row, i64 %col64
  %v = load volatile float, ptr addrspace(3) %gep, align 4
  store float %v, ptr %output, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
