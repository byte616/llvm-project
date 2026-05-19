; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Same idiom as divrem-fold.ll but with a non-power-of-two stride s=24.
;
; Source:
;   __shared__ float arr[32][32];
;   int x = 24 * tid;
;   ... = arr[x / 32][x % 32];
;
; Flat index reduces to 24*tid (1D), so we expect Tx=24 and the standard
; padding L=lcm(24, 32)=96 N=1 (in 4B/float units).

target triple = "nvptx64-nvidia-cuda"

@arr = internal addrspace(3) global [32 x [32 x float]] undef

; CHECK: [Padding] Candidate L=96 N=1 (units=4B)
; CHECK: [Padding] >>> Recommended pad period: L=96 N=1
; CHECK: @arr.padded = internal addrspace(3) global [{{[0-9]+}} x float]

define void @test_divrem_fold_s24(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %x = mul i32 %tid, 24
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
