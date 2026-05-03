; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Mixed access widths to the same shared memory.  Both accesses have stride
; that produces 2-way bank conflict:
;   - float access, byte stride 8  -> candidate (L=32, N=1, EB=1)  (4B padding)
;   - float4 access, byte stride 32 -> candidate (L=8,  N=1, EB=4) (16B padding)
;
; The 4B-padding candidate would misalign the float4 access (16B alignment
; required), so the alignment filter must drop it.  Only the 16B-padding
; candidate survives.

target triple = "nvptx64-nvidia-cuda"

@shared = internal addrspace(3) global [512 x float] undef

; The float-derived candidate must be dropped by the alignment filter.
; CHECK: [Padding] Drop candidate (L=32,N=1,EB=1): N_bytes=4 not aligned to max access size 16B

; The float4-derived candidate must survive and be applied.
; CHECK: [Padding] Candidate L=8 N=1 (units=16B)
; CHECK: applying L=32 N=4 in ElemTy units (4B)

; CHECK: @shared.padded = internal addrspace(3) global [576 x float]

define void @test_mixed_alignment(ptr %out_f, ptr %out_f4) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tid64 = sext i32 %tid to i64

  ; float access, stride 2 (byte stride 8) -> 2-way conflict.
  %idx_f = mul nsw i64 %tid64, 2
  %gep_f = getelementptr inbounds [512 x float], ptr addrspace(3) @shared, i64 0, i64 %idx_f
  %vf = load float, ptr addrspace(3) %gep_f
  store float %vf, ptr %out_f

  ; float4 access, stride 2 (byte stride 32) -> 2-way conflict.
  %idx_f4 = mul nsw i64 %tid64, 8
  %gep_f4 = getelementptr inbounds [512 x float], ptr addrspace(3) @shared, i64 0, i64 %idx_f4
  %vf4 = load <4 x float>, ptr addrspace(3) %gep_f4
  store <4 x float> %vf4, ptr %out_f4

  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
