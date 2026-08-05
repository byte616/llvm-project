; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Verifies that for 4-byte (float) accesses with bank conflict, the candidate
; generator emits the fixed extra L set {32, 96, 160, 224, 352, 416, 480} in
; addition to the per-stride lcm-derived candidate.  Without this extra set,
; a stride-2 access would only produce L=32.  The cost model is still expected
; to pick L=32 (best score), so the final padding is unchanged from
; scalar-stride2.ll.

target triple = "nvptx64-nvidia-cuda"

@shared = internal addrspace(3) global [256 x float] undef

; CHECK-DAG: [Padding] Candidate L=32 N=1 (units=4B)
; CHECK-DAG: [Padding] Candidate L=96 N=1 (units=4B)
; CHECK-DAG: [Padding] Candidate L=160 N=1 (units=4B)
; CHECK-DAG: [Padding] Candidate L=224 N=1 (units=4B)
; CHECK-DAG: [Padding] Candidate L=352 N=1 (units=4B)
; CHECK-DAG: [Padding] Candidate L=416 N=1 (units=4B)
; CHECK-DAG: [Padding] Candidate L=480 N=1 (units=4B)
; CHECK:     [Padding] >>> Recommended pad period: L=32 N=1

define void @test_scalar_stride2(ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tid64 = sext i32 %tid to i64
  %idx = mul nsw i64 %tid64, 2
  %gep = getelementptr inbounds [256 x float], ptr addrspace(3) @shared, i64 0, i64 %idx
  %v = load float, ptr addrspace(3) %gep
  store float %v, ptr %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
