; RUN: opt -passes=shared-mem-pad -cuda-blockdim-file=%S/Inputs/blockdim.json -S < %s 2>&1 | FileCheck %s

; Verify the 2-D closed-form fast path (the default `-shared-mem-pad-2d-mode`).
;
; Shape is [32 x [32 x float]] with stride 2 -> 2-way bank conflict, picks
; L=32, N=1.  Because D1 (= 32) equals L, the pass should:
;   * keep a 1-D padded layout: [32*(32+1) x float] = [1056 x float]
;   * rewrite each GEP to compute padded = (D1+N)*row + col = 33*row + col
;     directly, skipping the flatten path's `flat + N*(flat/L)` chain
;     (no udiv / lshr / `flat = 32*r + c` add).

target triple = "nvptx64-nvidia-cuda"

@arr = internal addrspace(3) global [32 x [32 x float]] undef

; CHECK: [Transform] Closed-form padded (2D fast path): arr.padded [1056 x T] (stride=33, L=32, N=1)
; CHECK: @arr.padded = internal addrspace(3) global [1056 x float] undef

; CHECK-LABEL: define void @test_closed_form_2d(
; The IR should multiply row by 33 (the padded stride), add col, and emit
; a 1-D GEP into the new padded global -- with no udiv / lshr-by-5 / the
; flatten path's intermediate `flat = 32*r + c` add.
; CHECK:     %[[ROW:.*]] = mul i64 %{{.*}}, 33
; CHECK:     %[[PAD:.*]] = add i64 %[[ROW]], %{{.*}}
; CHECK:     %[[GEP:.*]] = getelementptr inbounds [1056 x float], ptr addrspace(3) @arr.padded, i64 0, i64 %[[PAD]]
; CHECK:     load float, ptr addrspace(3) %[[GEP]]
; CHECK-NOT: udiv
; CHECK-NOT: lshr i64 %{{.*}}, 5

define void @test_closed_form_2d(ptr %out) {
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
