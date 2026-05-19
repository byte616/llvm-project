; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Same div/rem idiom as divrem-fold.ll with s=6 (non-power-of-two stride
; sharing gcd 2 with the 32-bank width).  Expect Tx=6, 2-way conflict,
; padding L=lcm(6,32)=96 N=1.

target triple = "nvptx64-nvidia-cuda"

@arr = internal addrspace(3) global [32 x [32 x float]] undef

; CHECK: Load Tx:6 Ty:0 (2-way bank conflict)
; CHECK: [Padding] Candidate L=96 N=1 (units=4B)
; CHECK: [Padding] >>> Recommended pad period: L=96 N=1
; CHECK: @arr.padded = internal addrspace(3) global [{{[0-9]+}} x float]

define void @test_divrem_fold_s6(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %x = mul i32 %tid, 6
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
