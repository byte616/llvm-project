; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; float4 stride 2 (byte stride 32) -> 2-way conflict in phase 0
; (banks 0,8,16,24,0,8,16,24).
;
; The candidate L is computed in float4 units: L = lcm(2, 32/4) = lcm(2, 8) = 8.
; The GV is a float[] (ElemTySize=4, AccessSize=16), so the apply-time scale is
; 16/4 = 4, giving L=32, N=4 in float units.  Padded size = 512 + 4*16 = 576.
;
; This validates:
;   - phase splitting (S=2 detected as 2-way, not 8-way),
;   - wide-type L formula (uses 32/ElemBanks = 8, not 32),
;   - apply-time unit scaling (float4 candidate -> float[] storage).

target triple = "nvptx64-nvidia-cuda"

@shared = internal addrspace(3) global [512 x float] undef

; CHECK: 2-way bank conflict
; CHECK: [Padding] Candidate L=8 N=1 (units=16B)
; CHECK: [Padding] >>> Recommended pad period: L=8 N=1 (access units, 16B)
; CHECK: applying L=32 N=4 in ElemTy units (4B)

; CHECK: @shared.padded = internal addrspace(3) global [576 x float]

define void @test_float4_stride2(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tid64 = sext i32 %tid to i64
  %idx = mul nsw i64 %tid64, 8
  %gep = getelementptr inbounds [512 x float], ptr addrspace(3) @shared, i64 0, i64 %idx
  %v = load <4 x float>, ptr addrspace(3) %gep
  store <4 x float> %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
