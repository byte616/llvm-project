; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; float2 stride 2 (byte stride 16) -> 2-way conflict in phase 0
; (banks 0,4,8,12,16,20,24,28 then 0,4,... = 2-way).
;
; Candidate L = lcm(2, 32/2) = lcm(2, 16) = 16 (in float2 units).
; GV is float[] (ElemTySize=4, AccessSize=8), so apply-time Scale = 8/4 = 2.
; Final padding: L=32, N=2 in float units; size = 512 + 2*(512/32) = 544.

target triple = "nvptx64-nvidia-cuda"

@shared = internal addrspace(3) global [512 x float] undef

; CHECK: 2-way bank conflict
; CHECK: [Padding] Candidate L=16 N=1 (units=8B)
; CHECK: applying L=32 N=2 in ElemTy units (4B)

; CHECK: @shared.padded = internal addrspace(3) global [544 x float]

define void @test_float2_stride2(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tid64 = sext i32 %tid to i64
  %idx = mul nsw i64 %tid64, 4
  %gep = getelementptr inbounds [512 x float], ptr addrspace(3) @shared, i64 0, i64 %idx
  %v = load <2 x float>, ptr addrspace(3) %gep
  store <2 x float> %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
