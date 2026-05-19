#include "llvm/Transforms/Utils/SharedMemPad.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BlockGridDimensionAnalysis.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <vector>

using namespace llvm;
#define DEBUG_TYPE "shared-mem-pad"

// Command-line option pointing to the JSON file produced by Phase 1
// (running -passes=cuda-blockdim-extract on the host IR).
// If provided, blockDim is read from the JSON for accurate bank conflict
// analysis. Without it, analysis is skipped for kernels with 2D/3D access.
static cl::opt<std::string> BlockDimFile(
    "cuda-blockdim-file",
    cl::desc("JSON file with CUDA kernel block dimensions (produced by "
             "running 'opt -passes=cuda-blockdim-extract' on host IR)"),
    cl::value_desc("filename"), cl::init(""));

// target machine: NV GPU
// 1. Calculate the access stride of shared memory between all the threads in
// the single warp
// 2. If stride S is odd, there is no bank conflict --> do nothing
// 3. If stride S is even, there exists bank conflict --> every L (lcm(32, S))
// elements pad 1 element

// Original index: tx * S + b --> after padding, index should be tx * S + b +
// (tx * S) / L We can optimize (tx * S) / L as (tx >> d) where d is 32/gcd(32,
// S) (power-of-2)

// If there is only one stride S for the shared memory variable SM
// --> turn the dimension of SM into 1D (size will be original size + last
// element )
// --> turn all the load/store instructions address to the padded address

// If there are multiple strides for the shared memory variable SM
// --> to be continued

// File-scope helpers: gcd / lcm on int64_t.  Used by both the bank-conflict
// simulator and the candidate-generation logic.
static int64_t SMPGcd(int64_t A, int64_t B) {
  A = std::abs(A);
  B = std::abs(B);
  while (B != 0) {
    int64_t R = A % B;
    A = B;
    B = R;
  }
  return A;
}
static int64_t SMPLcm(int64_t A, int64_t B) {
  if (A == 0 || B == 0)
    return 0;
  return (A / SMPGcd(A, B)) * B;
}
// guess 1: duplicate the shared memory variable?
// guess 2: use loop iteration to determine solve which// Enums to define the
// stride case
enum class StrideCase {
  BROADCAST,   // Stride is 0 (all threads access same memory)
  STRIDE_ODD,  // Stride is odd (e.g., 1, 3, 5... no conflict)
  STRIDE_EVEN, // Stride is even (e.g., 2, 4, 6... potential bank conflict)
  UNKNOWN_CASE // Non-linear, indirect access, or complex expression
};

// Struct to encapsulate stride dimensions
struct DimStrides {
  int64_t Tx = 0;
  int64_t Ty = 0;
  int64_t Tz = 0;
  bool IsUnknown = false;

  void add(const DimStrides &O) {
    if (O.IsUnknown) {
      IsUnknown = true;
      return;
    }
    Tx += O.Tx;
    Ty += O.Ty;
    Tz += O.Tz;
  }

  void multiply(int64_t F) {
    Tx *= F;
    Ty *= F;
    Tz *= F;
  }
};

// Data structure to hold access information for a specific instruction
struct AccessInfo {
  Instruction *Inst;   // the load or store instruction
  DimStrides Strides;  // the multi-dimensional strides
  StrideCase CaseType; // the classification
  uint64_t Weight;     // estimated access frequency
  int ConflictCount;   // estimated max bank conflict count
  unsigned ElemBanks;  // number of 4-byte banks per element (1=float, 2=float2, 4=float4)
};

// Data structure to hold all accesses for a specific shared memory variable
struct SharedMemVarInfo {
  Value *BaseVar;                // The shared memory variable (Alloca / Global)
  std::vector<AccessInfo> Loads; // List of Load accesses
  std::vector<AccessInfo> Stores; // List of Store accesses
  bool HasUnknownAccess =
      false; // if there exists unknown access, we cannot pad
};

// SCEV Visitor to traverse the expression tree and find the coefficient of tid
// (x, y, or z)
class StrideVisitor : public SCEVVisitor<StrideVisitor, DimStrides> {
  ScalarEvolution &SE;
  Value *TidX;
  Value *TidY;
  Value *TidZ;

public:
  StrideVisitor(ScalarEvolution &SE, Value *X, Value *Y, Value *Z)
      : SE(SE), TidX(X), TidY(Y), TidZ(Z) {}

  DimStrides visitUnknown(const SCEVUnknown *U) {
    DimStrides Res;
    if (TidX && U->getValue() == TidX)
      Res.Tx = 1;
    else if (TidY && U->getValue() == TidY)
      Res.Ty = 1;
    else if (TidZ && U->getValue() == TidZ)
      Res.Tz = 1;
    return Res;
  }

  // Peel off any number of zero/sign-extend wrappers, returning the
  // innermost SCEV.  Used by the div/rem fold so that the matcher works
  // even when the address arithmetic widens `x` to i64 (sext/zext gets
  // pushed inside udiv/and by SCEV in some configurations).
  static const SCEV *peelExtensions(const SCEV *S) {
    while (true) {
      if (auto *ZE = dyn_cast<SCEVZeroExtendExpr>(S))
        S = ZE->getOperand();
      else if (auto *SX = dyn_cast<SCEVSignExtendExpr>(S))
        S = SX->getOperand();
      else if (auto *TR = dyn_cast<SCEVTruncateExpr>(S))
        S = TR->getOperand();
      else
        break;
    }
    return S;
  }

  // Match S == K * (A udiv C) with K, C positive constants.
  // K==1 case: S itself is the SCEVUDivExpr (no enclosing Mul).
  // Looks through zext/sext/trunc wrappers on the Mul operands so that
  // the inner UDivExpr is found regardless of any bitwidth normalization
  // SCEV performed when widening to pointer size.
  static bool matchKMulUDiv(const SCEV *S, int64_t &K, int64_t &C,
                            const SCEV *&A) {
    S = peelExtensions(S);
    if (auto *UD = dyn_cast<SCEVUDivExpr>(S)) {
      auto *DivC = dyn_cast<SCEVConstant>(UD->getRHS());
      if (!DivC) return false;
      K = 1;
      C = DivC->getAPInt().getSExtValue();
      A = UD->getLHS();
      return C > 0;
    }
    auto *M = dyn_cast<SCEVMulExpr>(S);
    if (!M) return false;
    const SCEVConstant *KC = nullptr;
    const SCEVUDivExpr *UD = nullptr;
    for (const SCEV *Op : M->operands()) {
      const SCEV *PO = peelExtensions(Op);
      if (auto *C2 = dyn_cast<SCEVConstant>(PO)) {
        if (KC) return false;
        KC = C2;
      } else if (auto *U = dyn_cast<SCEVUDivExpr>(PO)) {
        if (UD) return false;
        UD = U;
      } else {
        return false;
      }
    }
    if (!KC || !UD) return false;
    auto *DivC = dyn_cast<SCEVConstant>(UD->getRHS());
    if (!DivC) return false;
    K = KC->getAPInt().getSExtValue();
    C = DivC->getAPInt().getSExtValue();
    A = UD->getLHS();
    return K > 0 && C > 0;
  }

  DimStrides visitAddExpr(const SCEVAddExpr *Expr) {
    // ----- Pre-pass: collapse the GEP idiom `arr[x / C][x % C]` -----
    //
    // For typical 2-D shared accesses such as
    //     __shared__ float arr[H][C];
    //     int x = stride * tid;
    //     ... = arr[x / C][x % C];
    // the address SCEV produced by GEP is
    //     base + K1 * (A udiv C) + RemTerm
    // where K1 is the row pitch in bytes and RemTerm represents
    // `K2 * (A urem C)` in some form — most commonly an opaque
    //     K2 * SCEVUnknown(and A, C-1)                   (post-InstCombine),
    // or, after SCEV's own canonicalization of bit-mask `and`,
    //     K2 * SCEVZeroExtend(SCEVTruncate(...)).
    //
    // The relation K1 = K2 * C makes the whole thing equal to
    //     base + K2 * A.
    //
    // We recognise the pair by *constructing* the canonical rem term via
    // SCEV's `getURemExpr` and letting SCEV's value-numbering tell us
    // whether that exact SCEV node is already an operand of the AddExpr.
    // Because SCEV uniques nodes, both forms (rem-from-and-IR and
    // rem-from-getURemExpr) collapse to the same pointer when they are
    // mathematically identical, so a pointer-equal lookup suffices.
    for (unsigned i = 0; i < Expr->getNumOperands(); ++i) {
      const SCEV *Opi = Expr->getOperand(i);
      int64_t K1 = 0, C = 0;
      const SCEV *A = nullptr;
      if (!matchKMulUDiv(Opi, K1, C, A)) continue;
      if (C <= 0 || K1 % C != 0) continue;
      int64_t K2 = K1 / C;
      // Build the canonical K2 * urem(A, C) plus all gcd-factored
      // variants and look for any of them among the remaining operands.
      //
      // SCEV uniques nodes by structure, so a pointer comparison only
      // succeeds when the candidate has the exact same shape as what
      // SCEV used when first constructing the AddExpr.  Two equivalent
      // shapes show up in practice:
      //   1. The canonical urem form, produced by SCEV when it
      //      processes an `and X, 2^k-1` IR instruction (or an `urem`
      //      directly): `K2 * zext_ik(c * trunc_ik(B))` with k=log2(C).
      //   2. A gcd-factored form, produced when InstCombine has
      //      narrowed the mask to `and X, M` with M = (2^k-1) & ~low,
      //      because the low bits of X are known zero (e.g. clang turns
      //      `(6*tid) & 31` into `(6*tid) & 30` since 6 is even).  SCEV
      //      then represents the rem as
      //          (K2*g) * zext_ij((c/g) * trunc_ij(B))
      //      with g = gcd(c, C), j = log2(C/g), C/g a power of two.
      //
      // We enumerate divisors of gcd(LinCoef, C) so that at least one
      // candidate matches whichever shape SCEV picked.
      auto BuildExpected = [&](const SCEV *RemA,
                               int64_t RemC) -> const SCEV * {
        const SCEV *URem =
            SE.getURemExpr(RemA, SE.getConstant(RemA->getType(), RemC));
        int64_t Scale = K1 / RemC; // K2 * (C / RemC)
        return Scale == 1
                   ? URem
                   : SE.getMulExpr(SE.getConstant(RemA->getType(), Scale),
                                   URem);
      };

      SmallVector<const SCEV *, 4> Candidates;
      Candidates.push_back(BuildExpected(A, C));

      // Try gcd-factored variants when A is a constant-times-rest mul.
      if (auto *AM = dyn_cast<SCEVMulExpr>(A)) {
        if (auto *CC = dyn_cast<SCEVConstant>(AM->getOperand(0))) {
          int64_t LinCoef = CC->getAPInt().getSExtValue();
          if (LinCoef > 0) {
            int64_t G = std::gcd<int64_t>(std::abs(LinCoef), C);
            for (int64_t Div = 2; Div <= G; ++Div) {
              if (G % Div != 0) continue;
              if (LinCoef % Div != 0 || C % Div != 0) continue;
              // Build A' = (LinCoef/Div) * rest, C' = C/Div.
              SmallVector<const SCEV *, 4> RestOps;
              RestOps.push_back(
                  SE.getConstant(A->getType(), LinCoef / Div));
              for (unsigned k = 1; k < AM->getNumOperands(); ++k)
                RestOps.push_back(AM->getOperand(k));
              const SCEV *AReduced =
                  RestOps.size() == 1 ? RestOps[0] : SE.getMulExpr(RestOps);
              Candidates.push_back(BuildExpected(AReduced, C / Div));
            }
          }
        }
      }

      const SCEV *Match = nullptr;
      unsigned MatchJ = 0;
      for (unsigned j = 0; j < Expr->getNumOperands() && !Match; ++j) {
        if (i == j) continue;
        for (const SCEV *Cand : Candidates) {
          if (Expr->getOperand(j) == Cand) {
            Match = Cand;
            MatchJ = j;
            break;
          }
        }
      }
      if (Match) {
        unsigned j = MatchJ;
        // Match!  Rebuild the AddExpr with the udiv operand and the rem
        // operand replaced by the equivalent linear form K2 * A.
        SmallVector<const SCEV *, 4> NewOps;
        for (unsigned k = 0; k < Expr->getNumOperands(); ++k)
          if (k != i && k != j)
            NewOps.push_back(Expr->getOperand(k));
        NewOps.push_back(
            (K2 == 1) ? A : SE.getMulExpr(SE.getConstant(A->getType(), K2), A));
        return visit(SE.getAddExpr(NewOps));
      }
    }

    // ----- No fold applied: fall back to per-operand visitation -----
    DimStrides Total;
    for (const SCEV *Op : Expr->operands()) {
      DimStrides OpStride = visit(Op);
      if (OpStride.IsUnknown) {
        Total.IsUnknown = true;
        return Total;
      }
      Total.add(OpStride);
    }
    return Total;
  }

  DimStrides visitMulExpr(const SCEVMulExpr *Expr) {
    int64_t ConstantFactor = 1;
    DimStrides TidStrides;
    int TidOperands = 0;

    for (const SCEV *Op : Expr->operands()) {
      if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Op)) {
        ConstantFactor *= C->getAPInt().getSExtValue();
      } else {
        DimStrides S = visit(Op);
        if (S.IsUnknown)
          return S;

        if (S.Tx != 0 || S.Ty != 0 || S.Tz != 0) {
          TidStrides = S;
          TidOperands++;
        }
      }
    }

    if (TidOperands > 1) {
      DimStrides Unknown;
      Unknown.IsUnknown = true;
      return Unknown;
    }

    TidStrides.multiply(ConstantFactor);
    return TidStrides;
  }

  DimStrides visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    DimStrides StartStride = visit(Expr->getStart());
    DimStrides StepStride = visit(Expr->getStepRecurrence(SE));

    if (StartStride.IsUnknown || StepStride.IsUnknown) {
      StartStride.IsUnknown = true;
      return StartStride;
    }

    // If tid.x is part of the step (e.g., i += tid.x), then the stride changes
    // every iteration.
    if (StepStride.Tx != 0 || StepStride.Ty != 0 || StepStride.Tz != 0) {
      StartStride.IsUnknown = true;
      return StartStride;
    }

    return StartStride;
  }

  DimStrides visitTruncateExpr(const SCEVTruncateExpr *Expr) {
    return visit(Expr->getOperand());
  }
  DimStrides visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    return visit(Expr->getOperand());
  }
  DimStrides visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    return visit(Expr->getOperand());
  }

  DimStrides visitCouldNotCompute(const SCEVCouldNotCompute *Expr) {
    return {0, 0, 0, true};
  }
  DimStrides visitConstant(const SCEVConstant *Expr) {
    return {0, 0, 0, false};
  }

  DimStrides visitVScale(const SCEVVScale *Expr) { return {0, 0, 0, true}; }
  DimStrides visitPtrToIntExpr(const SCEVPtrToIntExpr *Expr) {
    return {0, 0, 0, true};
  }
  DimStrides visitUDivExpr(const SCEVUDivExpr *Expr) { return {0, 0, 0, true}; }
  DimStrides visitSequentialUMinExpr(const SCEVSequentialUMinExpr *Expr) {
    return {0, 0, 0, true};
  }

  DimStrides visitSMaxExpr(const SCEVSMaxExpr *Expr) { return {0, 0, 0, true}; }
  DimStrides visitSMinExpr(const SCEVSMinExpr *Expr) { return {0, 0, 0, true}; }
  DimStrides visitUMaxExpr(const SCEVUMaxExpr *Expr) { return {0, 0, 0, true}; }
  DimStrides visitUMinExpr(const SCEVUMinExpr *Expr) { return {0, 0, 0, true}; }
};

// Returns the total number of base (float) elements in a (nested) array type.
// e.g. [16 x [16 x float]] → 256.
static int64_t computeFlatSize(Type *Ty) {
  if (auto *AT = dyn_cast<ArrayType>(Ty))
    return (int64_t)AT->getNumElements() *
           computeFlatSize(AT->getElementType());
  return 1; // base element
}

// For GEP source type T = [A x [B x float]], returns strides for all
// non-leading indices: [B, 1].  Works for any depth of nesting.
static SmallVector<int64_t, 4> computeDimStrides(Type *SrcTy) {
  SmallVector<int64_t, 4> Strides;
  Type *Ty = SrcTy;
  while (isa<ArrayType>(Ty)) {
    Type *ElemTy = cast<ArrayType>(Ty)->getElementType();
    Strides.push_back(computeFlatSize(ElemTy));
    Ty = ElemTy;
  }
  return Strides;
}

// Replace a shared-memory GlobalVariable with a padded 1-D version.
//
// Original layout: GV  = [A x [B x float]]  (addrspace 3)
// New layout:      GV' = [FlatSize + FlatSize/L x float]  (addrspace 3)
//
// Every GEP  "gep [AxB], ptr, 0, row, col"  is rewritten to
//   flat      = row*B + col
//   padded    = flat + udiv(flat, L)
//   "gep [N], ptr', 0, padded"
//
// This walk covers unrolled GEPs automatically because every unrolled
// iteration produces an independent GEP instruction, all of them users
// of the same GlobalVariable (possibly through an addrspacecast CE).
static void applyFlattenAndPad(GlobalVariable *GV, int64_t L, int64_t N,
                               LoopInfo *LI = nullptr,
                               ScalarEvolution *SE = nullptr) {
  Module *M = GV->getParent();
  LLVMContext &Ctx = M->getContext();
  Type *I64Ty = Type::getInt64Ty(Ctx);

  // 1. Compute sizes.
  int64_t FlatSize = computeFlatSize(GV->getValueType());
  int64_t PaddedSize = FlatSize + N * (FlatSize / L);

  // Scalar element type (float), obtained by unwrapping all array layers.
  Type *ElemTy = GV->getValueType();
  while (auto *AT = dyn_cast<ArrayType>(ElemTy))
    ElemTy = AT->getElementType();

  // 2. Create new 1-D global in the same address space.
  ArrayType *NewTy = ArrayType::get(ElemTy, PaddedSize);
  auto *NewGV =
      new GlobalVariable(*M, NewTy, /*isConstant=*/false, GV->getLinkage(),
                         UndefValue::get(NewTy), GV->getName() + ".padded",
                         /*InsertBefore=*/nullptr, GV->getThreadLocalMode(),
                         GV->getAddressSpace());
  NewGV->setAlignment(GV->getAlign());
  NewGV->setUnnamedAddr(GV->getUnnamedAddr());

  // 3. Collect all GEP users that transitively address GV.
  //    Three kinds:
  //    a) GEP instructions   (runtime indices, common case)
  //    b) CE GEPs            (all indices are constants; appear after unrolling
  //                           when the optimizer folds indices to constants)
  //    c) Direct Load/Store  (access to base pointer, i.e., flat index 0)
  //    Non-GEP CEs (addrspacecast, bitcast) are just passed through.
  SmallVector<GetElementPtrInst *, 32> GEPsToReplace;
  SmallVector<ConstantExpr *, 16> CEGEPsToReplace;
  SmallVector<Instruction *, 8> DirectAccessesToReplace; // Load/Store to base

  // ThroughGEP: If the call chain has passed through any GEP (instruction or ConstantExpr),
  // then the Load/Store on that chain should not be treated as a direct base-pointer access
  // and must not be added to DirectAccessesToReplace;
  // otherwise, the padded-index pointer already produced by the GEP would be overwritten
  // with a flat-index-0 pointer.
  std::function<void(Value *, bool)> Collect = [&](Value *V, bool ThroughGEP) {
    for (User *U : V->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        GEPsToReplace.push_back(GEP);
      } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->getOpcode() == Instruction::GetElementPtr) {
          // CE is itself a GEP — collect it for constant-folded replacement.
          CEGEPsToReplace.push_back(CE);
          // Recurse *through the GEP*: downstream load/stores reached via
          // this CE GEP must not be treated as direct base accesses.
          Collect(CE, /*ThroughGEP=*/true);
        } else {
          // Non-GEP CE (addrspacecast, bitcast): pass-through.
          Collect(CE, ThroughGEP);
        }
      } else if (auto *Cast = dyn_cast<AddrSpaceCastInst>(U)) {
        Collect(Cast, ThroughGEP);
      } else if (auto *Cast = dyn_cast<BitCastInst>(U)) {
        Collect(Cast, ThroughGEP);
      } else if (auto *LI = dyn_cast<LoadInst>(U)) {
        // Only a direct base-pointer load (flat index 0) if no GEP in chain.
        if (!ThroughGEP)
          DirectAccessesToReplace.push_back(LI);
      } else if (auto *SI = dyn_cast<StoreInst>(U)) {
        // Direct store to base pointer (flat index 0).
        // Require: (a) V is the pointer operand (not the stored value), and
        // (b) no GEP in the chain.
        if (!ThroughGEP && SI->getPointerOperand() == V) {
          DirectAccessesToReplace.push_back(SI);
        }
      }
      // Note: PHINode and SelectInst are not handled — those would require
      // more complex analysis to track all incoming values.
    }
  };
  Collect(GV, /*ThroughGEP=*/false);

  // Helper: given a CE GEP, return the flat element index as a constant.
  //   Two forms are handled:
  //   - Element form:  getelementptr [AxBxfloat], ptr, 0, row, col
  //   - Byte-offset:   getelementptr i8, ptr, i64 <byteoffset>
  //     (produced by the optimizer when all indices are constant).
  const DataLayout &DL = M->getDataLayout();
  uint64_t ElemSize = DL.getTypeAllocSize(ElemTy); // bytes per element

  auto GetFlatIdxFromCEGEP = [&](ConstantExpr *CE) -> int64_t {
    Type *SrcTy = CE->getOperand(0)->getType(); // ptr type — not element type
    // Reconstruct the element type from the CE's source element type field.
    // For a GEP CE, getOperand(0) is the base pointer; the source element type
    // is embedded in the CE's sub-class data (accessed via
    // cast<GEPOperator>).
    auto *GEPOp = cast<GEPOperator>(CE);
    Type *GEPSrcTy = GEPOp->getSourceElementType();
    (void)SrcTy;

    if (GEPSrcTy->isIntegerTy(8)) {
      // Byte-offset form.  operand(1) is the byte offset.
      int64_t ByteOff = cast<ConstantInt>(CE->getOperand(1))->getSExtValue();
      return ByteOff / (int64_t)ElemSize;
    }

    // Element form: indices are [0, i1, i2, ...]; skip the leading 0.
    SmallVector<int64_t, 4> Strides = computeDimStrides(GEPSrcTy);
    int64_t Flat = 0;
    // CE operands: op[0]=ptr, op[1]=idx0(0), op[2]=row, op[3]=col, ...
    for (unsigned i = 2; i < CE->getNumOperands(); ++i) {
      int64_t IdxVal = cast<ConstantInt>(CE->getOperand(i))->getSExtValue();
      int64_t Stride = (i - 2 < Strides.size()) ? Strides[i - 2] : 1;
      Flat += IdxVal * Stride;
    }
    return Flat;
  };

  // 4a. Transform each GEP instruction.
  // One SCEVExpander shared across all GEPs so that phi nodes created for
  // the first GEP can be reused by subsequent GEPs whose SCEVs match or are
  // constant offsets of one another.  This avoids creating a separate IV
  // per unrolled copy of an access.
  std::unique_ptr<SCEVExpander> SharedExp;
  if (LI && SE) {
    SharedExp = std::make_unique<SCEVExpander>(*SE, M->getDataLayout(),
                                               "flat.iv");
    SharedExp->disableCanonicalMode();
  }
  // Cache of (SCEV of already-materialized FlatIdx, materialized Value).
  // Used to coalesce unrolled copies: if a later GEP's SCEV differs from
  // a cached one only by a loop-invariant constant, reuse the base and
  // add the constant offset instead of creating a new phi.
  SmallVector<std::pair<const SCEV *, Value *>, 8> FlatIVCache;

  for (auto *GEP : GEPsToReplace) {
    IRBuilder<> B(GEP);
    Type *GEPSrcTy = GEP->getSourceElementType();

    Value *FlatIdx;

    // Handle byte-offset form: getelementptr i8, ptr, i64 <byteoffset>
    if (GEPSrcTy->isIntegerTy(8) && GEP->getNumIndices() == 1) {
      Value *ByteOff = GEP->getOperand(1);
      if (ByteOff->getType() != I64Ty) {
        if (auto *CI = dyn_cast<ConstantInt>(ByteOff))
          ByteOff = ConstantInt::get(I64Ty, CI->getSExtValue());
        else
          ByteOff = B.CreateSExt(ByteOff, I64Ty, "byteoff.ext");
      }
      // Convert byte offset to element index.
      // When ElemSize is a power of two, lower the sdiv to an arithmetic
      // right shift (byte offsets from valid GEPs are exactly divisible).
      if (isPowerOf2_64(ElemSize))
        FlatIdx = B.CreateAShr(
            ByteOff, ConstantInt::get(I64Ty, Log2_64(ElemSize)), "elem.idx");
      else
        FlatIdx = B.CreateSDiv(ByteOff, ConstantInt::get(I64Ty, ElemSize),
                               "elem.idx");
    } else {
      // Element form: getelementptr [AxB], ptr, 0, row, col, ...
      SmallVector<int64_t, 4> Strides = computeDimStrides(GEPSrcTy);

      // Compute flat index from multi-dim GEP indices.
      // GEP operands: [ptr, idx0=0, idx1=row, idx2=col, ...]
      // We skip idx0 (the outer "array-of-arrays" base index, always 0).
      FlatIdx = ConstantInt::get(I64Ty, 0);
      for (unsigned int i = 1; i < GEP->getNumIndices(); ++i) {
        Value *Idx = GEP->getOperand(i + 1); // op[0]=ptr, so op[i+1] = idx[i]
        // Sign-extend / zero-extend to i64.
        if (Idx->getType() != I64Ty) {
          if (auto *CI = dyn_cast<ConstantInt>(Idx))
            Idx = ConstantInt::get(I64Ty, CI->getSExtValue());
          else
            Idx = B.CreateSExt(Idx, I64Ty, "idx.ext");
        }
        // Strides[i-1]: stride for the i-th non-leading index.
        int64_t Stride = (i - 1 < Strides.size()) ? Strides[i - 1] : 1;
        // Strength-reduce: stride==1 -> no mul, power-of-two -> shl.
        Value *Term;
        if (Stride == 1)
          Term = Idx;
        else if (isPowerOf2_64(Stride))
          Term = B.CreateShl(Idx, ConstantInt::get(I64Ty, Log2_64(Stride)),
                             "fmul");
        else
          Term = B.CreateMul(Idx, ConstantInt::get(I64Ty, Stride), "fmul");
        FlatIdx = B.CreateAdd(FlatIdx, Term, "flat");
      }
    }

    // Try to rewrite FlatIdx into a phi-based induction-variable form when
    // it is an affine recurrence in some enclosing loop. This makes the
    // downstream pad arithmetic (and the entire address) follow an IV
    // increment pattern, which significantly reduces per-iteration work
    // (LSR/IndVars usually do not turn the closed form into an IV on their
    // own).  Fallback: keep the current closed-form FlatIdx.
    if (SharedExp) {
      if (auto *FlatInst = dyn_cast<Instruction>(FlatIdx)) {
        if (Loop *Lp = LI->getLoopFor(FlatInst->getParent())) {
          if (Lp->getLoopPreheader()) {
            const SCEV *S = SE->getSCEV(FlatInst);
            if (S && !isa<SCEVCouldNotCompute>(S) &&
                SE->containsAddRecurrence(S)) {
              // Look for an already-expanded SCEV in the same loop whose
              // difference from S is a loop-invariant integer constant.
              // If found, reuse its phi and add the constant offset so
              // unrolled copies of the same access share ONE induction
              // variable instead of creating N parallel phis.
              //
              // Canonical-base policy: keep the SCEV with the *smallest*
              // start as the materialized phi.  This ensures the most
              // common k=0 unroll copy (which usually appears in the main
              // body block) uses the phi directly with no offset add,
              // while k=1..N copies pay a single positive add.  If we
              // discover a smaller start later, we *swap*: re-materialize
              // the new canonical and rewrite the old base's existing
              // uses through `new_base + |diff|`.
              Value *NewFlat = nullptr;
              auto *SAR = dyn_cast<SCEVAddRecExpr>(S);
              for (size_t i = 0; i < FlatIVCache.size(); ++i) {
                const SCEV *Cached = FlatIVCache[i].first;
                Value *CachedBase = FlatIVCache[i].second;
                auto *CachedAR = dyn_cast<SCEVAddRecExpr>(Cached);
                if (!CachedAR || !SAR ||
                    CachedAR->getLoop() != SAR->getLoop())
                  continue;
                const SCEV *Diff = SE->getMinusSCEV(S, Cached);
                auto *C = dyn_cast<SCEVConstant>(Diff);
                if (!C)
                  continue;
                const APInt &D = C->getAPInt();
                if (D == 0) {
                  NewFlat = CachedBase;
                } else if (D.isNonNegative()) {
                  // S = Cached + D (D > 0).  Derive directly.
                  IRBuilder<> BB(FlatInst);
                  NewFlat = BB.CreateAdd(
                      CachedBase, ConstantInt::get(I64Ty, D), "flat.iv.off");
                } else {
                  // S < Cached.  Make S the new canonical.
                  // 1) Materialize NewBase from S.
                  // 2) Replace the cached base's uses with `NewBase + |D|`.
                  // 3) Update the cache entry to (S, NewBase).
                  Value *NewBase = SharedExp->expandCodeFor(
                      S, I64Ty, &*Lp->getHeader()->getFirstInsertionPt());
                  if (auto *CBInst = dyn_cast<Instruction>(CachedBase)) {
                    // Insert the replacement after CBInst (or at the
                    // start of the loop body if CBInst is the loop-
                    // header phi).  Do NOT delete CBInst: SCEVExpander
                    // holds AssertingVHs to materialized values and
                    // would assert on deletion.  Downstream DCE will
                    // remove it once it is truly dead.
                    Instruction *InsertPt =
                        isa<PHINode>(CBInst)
                            ? &*CBInst->getParent()->getFirstInsertionPt()
                            : CBInst->getNextNode();
                    IRBuilder<> BB(InsertPt);
                    Value *Replacement = BB.CreateAdd(
                        NewBase, ConstantInt::get(I64Ty, -D),
                        "flat.iv.off");
                    CBInst->replaceUsesWithIf(
                        Replacement, [&](Use &U) {
                          return U.getUser() != Replacement;
                        });
                  }
                  FlatIVCache[i] = {S, NewBase};
                  NewFlat = NewBase;
                }
                break;
              }
              if (!NewFlat)
                NewFlat = SharedExp->expandCodeFor(S, I64Ty, FlatInst);
              if (NewFlat && NewFlat != FlatInst) {
                FlatInst->replaceAllUsesWith(NewFlat);
                RecursivelyDeleteTriviallyDeadInstructions(FlatInst);
                FlatIdx = NewFlat;
                // Reset the IRBuilder insert point, since FlatInst is gone.
                B.SetInsertPoint(GEP);
              }
              if (NewFlat) {
                // Only push if we either (a) created a fresh phi, or
                // (b) swapped to a new canonical.  In the constant-offset
                // derive case NewFlat is the offset add, not the base, so
                // we don't add it to the cache to avoid stacking offsets.
                bool IsBase = false;
                if (auto *Phi = dyn_cast<PHINode>(NewFlat))
                  IsBase = (LI->getLoopFor(Phi->getParent()) == SAR->getLoop());
                if (IsBase) {
                  // Ensure not already present.
                  bool Found = false;
                  for (auto &KV : FlatIVCache)
                    if (KV.second == NewFlat) {
                      Found = true;
                      break;
                    }
                  if (!Found)
                    FlatIVCache.push_back({S, NewFlat});
                }
              }
            }
          }
        }
      }
    }

    // Apply padding: padded = flat + N * (flat / L).
    // When L is a power of two, lower the udiv to a logical right shift.
    Value *Pad;
    if (isPowerOf2_64(L))
      Pad = B.CreateLShr(FlatIdx, ConstantInt::get(I64Ty, Log2_64(L)), "pad");
    else
      Pad = B.CreateUDiv(FlatIdx, ConstantInt::get(I64Ty, L), "pad");
    // When N is a power of two, lower the mul to a logical left shift.
    Value *ScaledPad;
    if (N == 1)
      ScaledPad = Pad;
    else if (isPowerOf2_64(N))
      ScaledPad =
          B.CreateShl(Pad, ConstantInt::get(I64Ty, Log2_64(N)), "scaled.pad");
    else
      ScaledPad = B.CreateMul(Pad, ConstantInt::get(I64Ty, N), "scaled.pad");
    Value *PaddedIdx = B.CreateAdd(FlatIdx, ScaledPad, "padded");

    // Build a pointer to the new global in the right address space.
    unsigned PtrAS =
        GEP->getPointerOperand()->getType()->getPointerAddressSpace();
    Value *NewPtr;
    if (NewGV->getAddressSpace() == PtrAS) {
      NewPtr = NewGV;
    } else {
      NewPtr =
          ConstantExpr::getAddrSpaceCast(NewGV, PointerType::get(Ctx, PtrAS));
    }

    // Emit new 1-D GEP.
    Value *Idxs[] = {ConstantInt::get(I64Ty, 0), PaddedIdx};
    Value *NewGEP = B.CreateGEP(NewTy, NewPtr, Idxs, GEP->getName() + ".padded",
                                GEP->isInBounds());

    GEP->replaceAllUsesWith(NewGEP);
    GEP->eraseFromParent();
  }

  // 4b. Transform each ConstantExpr GEP.
  //     Because all indices are constants, the padded index is also a constant
  //     — no instruction insertion needed.
  for (auto *CE : CEGEPsToReplace) {
    int64_t Flat = GetFlatIdxFromCEGEP(CE);
    if (Flat < 0 || Flat >= FlatSize) {
      errs() << "  [Transform] WARNING: CE GEP flat index " << Flat
             << " out of range for " << GV->getName() << ", skipping.\n";
      continue;
    }
    int64_t PaddedIdx = Flat + N * (Flat / L);

    // Build a constant pointer to NewGV in the right address space.
    unsigned PtrAS = CE->getType()->getPointerAddressSpace();
    Constant *NewPtr;
    if (NewGV->getAddressSpace() == PtrAS) {
      NewPtr = NewGV;
    } else {
      NewPtr =
          ConstantExpr::getAddrSpaceCast(NewGV, PointerType::get(Ctx, PtrAS));
    }

    // Build new CE GEP: getelementptr [N x float], ptr, 0, padded_idx.
    Constant *NewIdxs[] = {ConstantInt::get(I64Ty, 0),
                           ConstantInt::get(I64Ty, PaddedIdx)};
    Constant *NewCE = ConstantExpr::getGetElementPtr(NewTy, NewPtr, NewIdxs,
                                                     /*InBounds=*/true);

    CE->replaceAllUsesWith(NewCE);
    CE->destroyConstant();
  }

  // 4c. Transform direct Load/Store to base pointer (flat index 0).
  //     These access element 0, which after padding is still element 0.
  //     We need to redirect them to use the new padded global.
  for (auto *Inst : DirectAccessesToReplace) {
    IRBuilder<> B(Inst);

    // Get the pointer operand and its address space
    Value *OldPtr;
    if (auto *LI = dyn_cast<LoadInst>(Inst)) {
      OldPtr = LI->getPointerOperand();
    } else {
      OldPtr = cast<StoreInst>(Inst)->getPointerOperand();
    }
    unsigned PtrAS = OldPtr->getType()->getPointerAddressSpace();

    // Build a pointer to element 0 of the new global
    Value *NewPtr;
    if (NewGV->getAddressSpace() == PtrAS) {
      NewPtr = NewGV;
    } else {
      NewPtr =
          ConstantExpr::getAddrSpaceCast(NewGV, PointerType::get(Ctx, PtrAS));
    }

    // Flat index 0, padded index = 0 + 0/L = 0
    Value *Idxs[] = {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 0)};
    Value *NewGEP = B.CreateGEP(NewTy, NewPtr, Idxs, "base.padded");

    // Replace the pointer operand in the load/store
    if (auto *LI = dyn_cast<LoadInst>(Inst)) {
      LI->setOperand(LI->getPointerOperandIndex(), NewGEP);
    } else {
      auto *SI = cast<StoreInst>(Inst);
      SI->setOperand(SI->getPointerOperandIndex(), NewGEP);
    }
  }

  // 5. Remove the old global (all instruction uses replaced; constant-expr
  //    users are now dead and will be cleaned up).
  GV->removeDeadConstantUsers();
  if (GV->use_empty())
    GV->eraseFromParent();

  errs() << "  [Transform] Flattened & padded: " << GV->getName() << " → size "
         << FlatSize << " + " << N << "*" << FlatSize / L << " = " << PaddedSize
         << " (L=" << L << ", N=" << N << ")\n";
}

// For wide accesses (>4B per thread), hardware splits a warp's request into
// multiple phases.  Each phase serves up to 32 banks (128B):
//   - float  (4B):  1 phase  of 32 threads, 1 bank/thread
//   - float2 (8B):  2 phases of 16 threads, 2 banks/thread
//   - float4 (16B): 4 phases of  8 threads, 4 banks/thread
// Bank conflicts only occur **within a single phase**.  Conflicts across
// phases do not serialize because they happen in different cycles.
static int computeBankConflict(DimStrides S, int BlockDimX, int BlockDimY,
                               int ElemBanks = 1) {
  int ThreadsPerPhase = std::max(1, 32 / ElemBanks);
  int NumPhases = ElemBanks;
  int MaxConflictAcrossPhases = 0;

  for (int P = 0; P < NumPhases; ++P) {
    int Banks[32] = {0};
    int MaxConflict = 0;
    std::set<int64_t> Address;
    for (int t = 0; t < ThreadsPerPhase; ++t) {
      int T = P * ThreadsPerPhase + t; // lane id within the warp
      int ThreadX = T % BlockDimX;
      int ThreadY = T / BlockDimX;
      int64_t ElementIndex = ThreadX * S.Tx + ThreadY * S.Ty;
      if (Address.count(ElementIndex))
        continue; // broadcast: same address served once, no conflict
      Address.insert(ElementIndex);
      // Each element occupies ElemBanks consecutive 4-byte banks.
      // Count them all -- a phase has bank conflict iff any 4-byte bank is
      // hit by more than one thread.
      for (int b = 0; b < ElemBanks; ++b) {
        int64_t BankUnit = ElementIndex * ElemBanks + b;
        int Bank = (int)(((BankUnit % 32) + 32) % 32);
        Banks[Bank]++;
        MaxConflict = std::max(MaxConflict, Banks[Bank]);
      }
    }
    MaxConflictAcrossPhases =
        std::max(MaxConflictAcrossPhases, MaxConflict);
  }
  return MaxConflictAcrossPhases;
}

// Compute average bank conflict across one full cycle of warps for stride S
// **with** padding period L applied.
//
// When we pad every L logical elements (insert 1 extra physical element), the
// physical address of logical element i is:
//   physical(i) = i + floor(i / L)
//
// We simulate `CycleWarps` warps -- the smallest W such that the floor()
// term in `physical = logical + N*floor(logical/L)` advances by a constant
// per warp.  After such W, every subsequent warp's bank pattern is just a
// uniform shift of warp-0's pattern (same conflict count), so 1 cycle is
// enough -- and for our standard candidates (L = lcm(S, 32/ElemBanks)) we
// can prove L | 32*S, hence CycleWarps == 1 in practice.  The loop is kept
// for robustness against non-standard L values.
//
// Returns a double so the caller can compare fractional averages.
static double computeBankConflictWithPadding(DimStrides S, int BlockDimX,
                                             int BlockDimY, int64_t PaddingL,
                                             int64_t PadN = 1,
                                             int ElemBanks = 1) {

  // Dominant stride: prefer Tx, then Ty, then Tz.
  int64_t DomStride = (S.Tx != 0)   ? std::abs(S.Tx)
                      : (S.Ty != 0) ? std::abs(S.Ty)
                                    : std::abs(S.Tz);
  if (DomStride == 0)
    return 1.0; // broadcast -- always 1 (no conflict)

  // Cycle length in warps: smallest W such that L | W*32*|S|.
  //   W = lcm(32*|S|, L) / (32*|S|) = L / gcd(L, 32*|S|)
  // Examples:
  //   float,  S=5, L=96  -> 96 / gcd(96,160) = 96/32  = 3  warps
  //   float,  S=3, L=96  -> 96 / gcd(96, 96) = 96/96  = 1  warp
  //   float4, S=4, L=8   -> 8  / gcd(8, 128) = 8/8    = 1  warp
  // For standard candidates (L=lcm(S, 32/ElemBanks)) this is always 1 because
  // L | 32*S; reduces to the classic L/32 whenever 32 | L (4-byte cases).
  int64_t CycleWarps = std::max<int64_t>(
      PaddingL / SMPGcd(PaddingL, 32 * DomStride), 1);

  // Same phase-splitting model as computeBankConflict: a warp's wide access
  // is served in `NumPhases` phases of `ThreadsPerPhase` threads each.
  // Bank conflicts are only counted within a phase.
  int ThreadsPerPhase = std::max(1, 32 / ElemBanks);
  int NumPhases = ElemBanks;

  double TotalConflict = 0.0;
  for (int64_t W = 0; W < CycleWarps; ++W) {
    int WarpMaxConflict = 0;
    for (int P = 0; P < NumPhases; ++P) {
      int Banks[32] = {0};
      int MaxConflict = 0;
      std::set<int64_t> Seen;
      for (int t = 0; t < ThreadsPerPhase; ++t) {
        int T = P * ThreadsPerPhase + t;
        int ThreadX = T % BlockDimX;
        int ThreadY = T / BlockDimX;
        int64_t Logical =
            W * 32 * DomStride + std::abs(ThreadX * S.Tx + ThreadY * S.Ty);
        if (Seen.count(Logical))
          continue;
        Seen.insert(Logical);
        int64_t Physical = Logical + PadN * (Logical / PaddingL);
        // Each element occupies ElemBanks consecutive banks.
        for (int b = 0; b < ElemBanks; ++b) {
          int64_t BankUnit = Physical * ElemBanks + b;
          int Bank = (int)(((BankUnit % 32) + 32) % 32);
          Banks[Bank]++;
          MaxConflict = std::max(MaxConflict, Banks[Bank]);
        }
      }
      WarpMaxConflict = std::max(WarpMaxConflict, MaxConflict);
    }
    TotalConflict += WarpMaxConflict;
  }
  return TotalConflict / (double)CycleWarps;
}

PreservedAnalyses SharedMemPass::run(Function &F, FunctionAnalysisManager &AM) {
  // Check if the target is NVPTX
  Triple T(F.getParent()->getTargetTriple());
  if (!T.isNVPTX()) {
    return PreservedAnalyses::all();
  }

  // --- Load blockDim from JSON file if -cuda-blockdim-file was specified ---
  // Priority: JSON file > reqntidx attr > heuristic fallback
  int BlockDimX = 32, BlockDimY = 1, BlockDimZ = 1;
  bool HasKnownBlockDim = false;
  bool Modified = false;

  if (!BlockDimFile.empty()) {
    auto ResultOrErr =
        autotex::BlockGridDimensionAnalysisJSONImporter::fromFile(BlockDimFile);
    if (!ResultOrErr) {
      errs() << "[SharedMemPass] Warning: failed to load blockdim JSON: "
             << toString(ResultOrErr.takeError()) << "\n";
    } else {
      // The JSON is keyed by the *stub* function name (host side).
      // The device kernel name typically matches after stripping the
      // __device_stub__ prefix added by clang.
      StringRef DeviceName = F.getName();
      auto &BGDs = *ResultOrErr;

      // Strategy: try direct name match first.
      auto It = BGDs.find(DeviceName.str());

      // Strategy: suffix match.
      // Device function mangling:  _Z<N><name><types>
      // Stub function mangling:    _Z<M>__device_stub__<name><types>
      // Both share the suffix "<name><types>" after stripping "_Z" + digits.
      // Example:
      //   device: _Z12lud_internalPfii  → suffix: lud_internalPfii
      //   stub:   _Z22__device_stub__lud_internalPfii → ends_with that suffix
      if (It == BGDs.end()) {
        StringRef Suffix = DeviceName;
        Suffix.consume_front("_Z");
        while (!Suffix.empty() && isdigit(Suffix.front()))
          Suffix = Suffix.drop_front();
        if (!Suffix.empty()) {
          for (auto &[StubName, BGD] : BGDs) {
            if (StringRef(StubName).ends_with(Suffix)) {
              It = BGDs.find(StubName);
              errs() << "[SharedMemPass] Matched device '" << DeviceName
                     << "' to stub '" << StubName << "' via suffix '" << Suffix
                     << "'\n";
              break;
            }
          }
        }
      }

      if (It != BGDs.end()) {
        auto &BD = It->second.BlockDim;
        if (BD.X)
          BlockDimX = static_cast<int>(*BD.X);
        if (BD.Y)
          BlockDimY = static_cast<int>(*BD.Y);
        if (BD.Z)
          BlockDimZ = static_cast<int>(*BD.Z);
        HasKnownBlockDim = true;
        errs() << "[SharedMemPass] Loaded blockDim from JSON for '"
               << DeviceName << "': (" << BlockDimX << ", " << BlockDimY << ", "
               << BlockDimZ << ")\n";
      } else {
        errs() << "[SharedMemPass] No JSON entry found for '" << DeviceName
               << "'\n";
      }
    }
  }

  // Get ScalarEvolution Analysis and BlockFrequency Analysis
  ScalarEvolution &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  BlockFrequencyInfo &BFI = AM.getResult<BlockFrequencyAnalysis>(F);

  // Attempt to find the values representing threadIdx.x, y, z
  // In NVPTX, they are typically calls to @llvm.nvvm.read.ptx.sreg.tid.*()
  std::vector<Value *> TidValues;
  for (Instruction &I : instructions(F)) {
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      if (Function *CalledFn = CI->getCalledFunction()) {
        StringRef Name = CalledFn->getName();
        if (Name == "llvm.nvvm.read.ptx.sreg.tid.x" ||
            Name == "llvm.nvvm.read.ptx.sreg.tid.y" ||
            Name == "llvm.nvvm.read.ptx.sreg.tid.z") {
          TidValues.push_back(CI);
        }
      }
    }
  }

  // Map to hold our findings for each shared memory variable
  std::map<Value *, SharedMemVarInfo> SharedMemInfoMap;

  // Check if the function uses any shared memory
  // In NVPTX, shared memory is represented by Address Space 3
  bool UsesSharedMemory = false;
  for (Instruction &I : instructions(F)) {
    Value *PtrOp = nullptr;
    bool IsLoad = false;

    if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
      PtrOp = LI->getPointerOperand();
      IsLoad = true;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
      PtrOp = SI->getPointerOperand();
    }

    if (PtrOp) {
      // getUnderlyingObject will look through GEPs, BitCasts, and
      // AddrSpaceCasts to find the original allocation (e.g., the
      // GlobalVariable for shared memory)
      Value *BaseVar = getUnderlyingObject(PtrOp);

      // check if the base variable is allocated in shared memory (address space
      // 3)
      if (BaseVar->getType()->getPointerAddressSpace() == 3) {
        UsesSharedMemory = true;

        DimStrides S = {0, 0, 0, true};
        StrideCase Case = StrideCase::UNKNOWN_CASE;
        int ConflictCount = 32;
        unsigned ElemBanks = 1;

        if (TidValues.empty()) {
          // If the function doesn't even use thread IDs,
          // it might be broadcast (stride 0) or unanalyzable.
          // Let's be safe and mark it unknown.
          Case = StrideCase::UNKNOWN_CASE;
        } else {
          Value *TidX = nullptr, *TidY = nullptr, *TidZ = nullptr;
          for (Value *V : TidValues) {
            StringRef Name = cast<CallInst>(V)->getCalledFunction()->getName();
            if (Name == "llvm.nvvm.read.ptx.sreg.tid.x")
              TidX = V;
            else if (Name == "llvm.nvvm.read.ptx.sreg.tid.y")
              TidY = V;
            else if (Name == "llvm.nvvm.read.ptx.sreg.tid.z")
              TidZ = V;
          }

          // Get the primitive size in bytes for the element being accessed
          Type *AccessType =
              IsLoad ? cast<LoadInst>(&I)->getType()
                     : cast<StoreInst>(&I)->getValueOperand()->getType();
          const DataLayout &DL = F.getParent()->getDataLayout();
          unsigned ElementSizeBytes = DL.getTypeStoreSize(AccessType);
          // Number of 4-byte banks occupied by one element
          // (float=1, float2/double=2, float4=4)
          ElemBanks = std::max(1u, ElementSizeBytes / 4u);

          // Use SCEV to calculate the stride
          const SCEV *Expr = SE.getSCEV(PtrOp);
          StrideVisitor Visitor(SE, TidX, TidY, TidZ);
          S = Visitor.visit(Expr);

          if (S.IsUnknown) {
            Case = StrideCase::UNKNOWN_CASE;
          } else {
            // Convert byte strides to element strides
            S.Tx /= ElementSizeBytes;
            S.Ty /= ElementSizeBytes;
            S.Tz /= ElementSizeBytes;

            // Use blockDim from JSON (loaded at function start).
            // If Ty/Tz are used but blockDim is unknown, we cannot determine
            // how threads distribute across X and Y, so mark UNKNOWN.
            if (!HasKnownBlockDim) {
              Case = StrideCase::UNKNOWN_CASE;
            } else {
              // For pure 1D access (only Tx), warp always fills X-dimension,
              // so BlockDimX=32 is safe even when blockDim is not known.
              int MaxConflict =
                  computeBankConflict(S, BlockDimX, BlockDimY, ElemBanks);
              ConflictCount = MaxConflict;

              if (S.Tx == 0 && S.Ty == 0 && S.Tz == 0) {
                Case = StrideCase::BROADCAST;
                ConflictCount = 1;
              } else if (MaxConflict <= 1) {
                Case = StrideCase::STRIDE_ODD; // No conflict
                ConflictCount = 1;
              } else {
                Case = StrideCase::STRIDE_EVEN;
              }
            }
          }
        }

        std::optional<uint64_t> ProfileCount =
            BFI.getBlockProfileCount(I.getParent());
        uint64_t Weight = ProfileCount.has_value()
                              ? ProfileCount.value()
                              : BFI.getBlockFreq(I.getParent()).getFrequency();

        AccessInfo Info = {&I, S, Case, Weight, ConflictCount, ElemBanks};
        SharedMemInfoMap[BaseVar].BaseVar = BaseVar;
        if (Case == StrideCase::UNKNOWN_CASE) {
          SharedMemInfoMap[BaseVar].HasUnknownAccess = true;
        }
        if (IsLoad)
          SharedMemInfoMap[BaseVar].Loads.push_back(Info);
        else
          SharedMemInfoMap[BaseVar].Stores.push_back(Info);
      }
    }
  }

  // If no instruction in this function accesses shared memory, skip it
  if (!UsesSharedMemory) {
    return PreservedAnalyses::all();
  }

  // Print our extracted stride information
  errs() << "--- Shared Memory Access Analysis ---\n";
  for (auto &Pair : SharedMemInfoMap) {
    Value *BaseVar = Pair.first;
    SharedMemVarInfo &Info = Pair.second;

    errs() << "Variable: " << *BaseVar << "\n";
    if (Info.HasUnknownAccess) {
      errs()
          << "  [WARNING] Variable has UNKNOWN accesses. Cannot safely pad.\n";
    }

    std::map<std::pair<std::string, int>, uint64_t> LoadWeights;
    std::map<std::pair<std::string, int>, uint64_t> StoreWeights;

    auto FormatStride = [](DimStrides S, int Conflicts) {
      std::string Str = "Tx:" + std::to_string(S.Tx) +
                        " Ty:" + std::to_string(S.Ty) + " (" +
                        std::to_string(Conflicts) + "-way bank conflict)";
      return Str;
    };

    for (auto &Acc : Info.Loads) {
      LoadWeights[{FormatStride(Acc.Strides, Acc.ConflictCount),
                   (int)Acc.CaseType}] += Acc.Weight;
    }
    for (auto &Acc : Info.Stores) {
      StoreWeights[{FormatStride(Acc.Strides, Acc.ConflictCount),
                    (int)Acc.CaseType}] += Acc.Weight;
    }

    for (auto &Pair : LoadWeights) {
      errs() << "  Load " << Pair.first.first << " (Case: " << Pair.first.second
             << ") - Estimated Accesses: " << Pair.second << "\n";
    }
    for (auto &Pair : StoreWeights) {
      errs() << "  Store " << Pair.first.first
             << " (Case: " << Pair.first.second
             << ") - Estimated Accesses: " << Pair.second << "\n";
    }

    // -----------------------------------------------------------------------
    // Optimal Padding L Selection (Full Cost Model)
    //
    // For each candidate L = lcm(|S|, 32) (from even strides with conflict),
    // compute the net score across ALL accesses:
    //
    //   score(L) = Σ_i  W_i/TotalW × (conflict_before_i - conflict_after_i(L))
    //
    // conflict_after is computed by simulating physical bank access with
    // padding period L (physical(i) = i + floor(i/L)).
    // Positive score = net improvement; negative = net harm.
    // We pick the L with the highest score (> 0 means padding helps overall).
    // -----------------------------------------------------------------------
    if (!Info.HasUnknownAccess) {
      struct StrideEntry {
        int64_t Stride; // dominant stride in elements
        int ConflictBefore;
        uint64_t Weight;
        DimStrides FullStrides; // needed for 2D conflict recomputation
        unsigned ElemBanks;     // banks per element (1=float, 2=float2, 4=float4)
      };
      std::vector<StrideEntry> Entries;

      auto CollectEntries = [&](const std::vector<AccessInfo> &Accesses) {
        for (const auto &Acc : Accesses) {
          // Pick the dominant stride: prefer Tx, then Ty, then Tz
          int64_t S = Acc.Strides.Tx != 0   ? Acc.Strides.Tx
                      : Acc.Strides.Ty != 0 ? Acc.Strides.Ty
                                            : Acc.Strides.Tz;
          Entries.push_back(
              {S, Acc.ConflictCount, Acc.Weight, Acc.Strides, Acc.ElemBanks});
        }
      };
      CollectEntries(Info.Loads);
      CollectEntries(Info.Stores);

      uint64_t TotalWeight = 0;
      for (auto &E : Entries)
        TotalWeight += E.Weight;

      if (TotalWeight == 0) {
        errs() << "  [Padding] No accesses found.\n";
      } else {
        // Collect candidate (L, N) pairs from strides that currently
        // have bank conflicts beyond the minimum for their element width.
        // For each stride, L = lcm(|S|, 32/ElemBanks).
        //
        // ElemBanks = element_size / 4: how many 4-byte banks one element
        // spans.  The effective number of distinct bank positions is
        // 32/ElemBanks (e.g., 32 for float, 16 for float2, 8 for float4).
        //
        // N = ceil(32 / blockDim.x): the number of distinct threadY values
        // within a single warp.  Each ty sub-group independently creates
        // the same stride-S access pattern at a different base offset.
        // Padding N per L ensures sufficient bank separation between all
        // sub-groups.
        //
        // This formula requires blockDim.x to be a power of 2 (so it
        // evenly divides 32 and all sub-groups have equal size).
        // For non-power-of-2 blockDim.x, we fall back to N=1 (classic
        // padding), which still partially resolves bank conflicts.
        //
        // Examples:
        //   blockDim=(32,1) → N=1  (one sub-group, classic case)
        //   blockDim=(16,16)→ N=2  (two ty values per warp)
        //   blockDim=(8,4)  → N=4  (four ty values per warp)
        bool IsPow2BDX = BlockDimX > 0 && (BlockDimX & (BlockDimX - 1)) == 0;
        int64_t PadN = IsPow2BDX ? std::max<int64_t>(32 / BlockDimX, 1) : 1;
        // Each candidate is tagged with the ElemBanks of the access that
        // produced it, because L/N are expressed in *that access type's*
        // element units (e.g. float4 units when ElemBanks=4).  We need
        // this later to convert L/N into the GV's ElemTy units.
        std::set<std::tuple<int64_t, int64_t, unsigned>> Candidates;
        for (auto &E : Entries) {
          int64_t AbsStride = std::abs(E.Stride);
          // Skip if there is no bank conflict, or stride is 0 (broadcast).
          // (The phase-splitting model already accounts for the fact that
          // wide accesses can achieve 1-way -- e.g. float4 stride 1 -- so
          // the floor is 1, not ElemBanks.)
          if (AbsStride == 0 || E.ConflictBefore <= 1)
            continue;
          // L = lcm(|S|, 32/ElemBanks): period in elements before bank
          // pattern repeats.  32/ElemBanks is the number of distinct bank
          // positions available for this element width.
          int64_t NumEffBanks = 32 / E.ElemBanks;
          int64_t IdealL = SMPLcm(AbsStride, NumEffBanks);
          if (IdealL <= 0)
            continue;
          Candidates.insert({IdealL, PadN, E.ElemBanks});

          // ---------------------------------------------------------------
          // Extra candidates for 2D row-major access patterns.
          //
          // Picking only the "dominant stride" loses information when both
          // Tx and Ty are non-zero.  For the two canonical patterns:
          //
          //   arr[ty][tx]:  Cx = 1, Cy = W   -> L = W,  N = bx
          //   arr[tx][ty]:  Cx = W, Cy = 1   -> L = W,  N = (32/EB)/bx
          //
          // where W is the innermost array-dim size (= row pitch in
          // access-type elements).  These are added on top of the dominant-
          // stride candidate; the cost model below will discard them if
          // they do not actually reduce conflicts.
          //
          // Assumptions for the formulas to be optimal:
          //   * bx divides 32/EB
          //   * W is a multiple of 32/EB    (true for typical 32-wide tiles)
          //   * No mixed coefficients (pure Cx*tx + Cy*ty)
          // When the assumptions break, the cost model still acts as a
          // safety net and the candidate is simply ignored.
          // ---------------------------------------------------------------
          int64_t TxA = std::abs(E.FullStrides.Tx);
          int64_t TyA = std::abs(E.FullStrides.Ty);
          // arr[ty][tx]: row pitch W = |Ty|, shift each ty-subgroup by bx.
          if (TxA == 1 && TyA > 1 && BlockDimX > 0 &&
              (int64_t)BlockDimX < NumEffBanks) {
            int64_t LTT = TyA;
            int64_t NTT = BlockDimX;
            if (NTT > 0 && NTT < LTT)
              Candidates.insert({LTT, NTT, E.ElemBanks});
          }
          // arr[tx][ty]: row pitch W = |Tx|, shift each tx-subgroup by Q.
          if (TyA == 1 && TxA > 1 && BlockDimX > 0 && IsPow2BDX) {
            int64_t LTT = TxA;
            int64_t Q = std::max<int64_t>(NumEffBanks / BlockDimX, 1);
            if (Q > 0 && Q < LTT)
              Candidates.insert({LTT, Q, E.ElemBanks});
          }
        }

        // Alignment filter: when the same shared variable is accessed by
        // mixed widths (e.g. both float and float4), the padding inserted
        // must be a multiple of the *largest* access size, otherwise the
        // wider accesses would land on misaligned addresses after padding.
        //
        // Concretely, both L_bytes and N_bytes must be multiples of
        // MaxAccessSize = MaxEB * 4.  Because L_bytes = L * EB * 4 and
        // L is a multiple of 32/EB (by construction), L*EB is a multiple
        // of 32, so L_bytes is always aligned.  Only N needs checking:
        //   N_bytes = N * EB * 4  must be multiple of  MaxEB * 4
        //   <=>  N * EB  must be multiple of  MaxEB
        unsigned MaxEB = 1;
        for (auto &E : Entries)
          MaxEB = std::max(MaxEB, E.ElemBanks);
        if (MaxEB > 1 && !Candidates.empty()) {
          std::set<std::tuple<int64_t, int64_t, unsigned>> Filtered;
          for (auto &[L, N, EB] : Candidates) {
            if ((N * (int64_t)EB) % (int64_t)MaxEB == 0) {
              Filtered.insert({L, N, EB});
            } else {
              errs() << "  [Padding] Drop candidate (L=" << L << ",N=" << N
                     << ",EB=" << EB << "): N_bytes=" << (N * EB * 4)
                     << " not aligned to max access size " << (MaxEB * 4)
                     << "B\n";
            }
          }
          Candidates = std::move(Filtered);
        }

        if (Candidates.empty()) {
          errs() << "  [Padding] No conflicting even strides. No padding "
                    "needed.\n";
        } else {
          // Evaluate each candidate (L, N) across ALL accesses.
          struct LNScore {
            int64_t L;
            int64_t N;
            unsigned ElemBanks; // units that L/N are expressed in
            double Score;
          };
          std::vector<LNScore> Scores;
          for (auto &[L, N, EB] : Candidates) {
            double Score = 0.0;
            for (auto &E : Entries) {
              double W = (double)E.Weight / (double)TotalWeight;
              // Convert (L, N) from "EB element" units to "E.ElemBanks
              // element" units, so the simulator sees them in the same
              // unit as E.FullStrides.
              //   L_bytes      = L * EB * 4
              //   L_in_E_units = L_bytes / (E.ElemBanks * 4)
              //                = L * EB / E.ElemBanks
              // For our standard candidates L is a multiple of 32/EB, so
              // L*EB is a multiple of 32 and divisible by any
              // E.ElemBanks in {1,2,4} -> exact integer.
              int64_t LE = L * (int64_t)EB / (int64_t)E.ElemBanks;
              int64_t NE = N * (int64_t)EB / (int64_t)E.ElemBanks;
              double After = computeBankConflictWithPadding(
                  E.FullStrides, BlockDimX, BlockDimY, LE, NE, E.ElemBanks);
              Score += W * ((double)E.ConflictBefore - After);
            }
            Scores.push_back({L, N, EB, Score});
            errs() << "  [Padding] Candidate L=" << L << " N=" << N
                   << " (units=" << (EB * 4) << "B)"
                   << " Score=" << format("%.4f", Score) << "\n";
          }

          // Pick (L, N) with maximum score (must be > 0 to be beneficial).
          // Among equal scores, prefer smaller N (less memory overhead).
          int64_t BestL = 0, BestN = 0;
          unsigned BestElemBanks = 1;
          double BestScore = 0.0;
          for (auto &S : Scores) {
            if (S.Score > BestScore ||
                (S.Score == BestScore && S.N < BestN)) {
              BestScore = S.Score;
              BestL = S.L;
              BestN = S.N;
              BestElemBanks = S.ElemBanks;
            }
          }
          if (BestL > 0) {
            // L and N are in "access type" units (e.g. float4 if
            // BestElemBanks=4).  applyFlattenAndPad operates in the GV's
            // innermost ElemTy units.  Convert via the byte ratio:
            //   Scale = AccessSizeBytes / ElemTySizeBytes
            //         = (BestElemBanks * 4) / ElemTySize
            // Examples:
            //   GV=float[],  access=float4  -> Scale = 16/4 = 4
            //   GV=float4[], access=float4  -> Scale = 16/16 = 1
            //   GV=float[],  access=float   -> Scale =  4/4 = 1
            if (auto *GV = dyn_cast<GlobalVariable>(BaseVar)) {
              const DataLayout &DL = F.getParent()->getDataLayout();
              Type *ElemTy = GV->getValueType();
              while (auto *AT = dyn_cast<ArrayType>(ElemTy))
                ElemTy = AT->getElementType();
              uint64_t ElemTySize = DL.getTypeStoreSize(ElemTy);
              uint64_t AccessSize = (uint64_t)BestElemBanks * 4u;
              int64_t Scale = (ElemTySize > 0 && AccessSize >= ElemTySize)
                                  ? (int64_t)(AccessSize / ElemTySize)
                                  : 1;
              int64_t ScaledL = BestL * Scale;
              int64_t ScaledN = BestN * Scale;
              errs() << "  [Padding] >>> Recommended pad period: L=" << BestL
                     << " N=" << BestN << " (access units, " << AccessSize
                     << "B); applying L=" << ScaledL << " N=" << ScaledN
                     << " in ElemTy units (" << ElemTySize << "B)"
                     << " Score=" << format("%.4f", BestScore) << "\n";
              auto &LI = AM.getResult<LoopAnalysis>(F);
              auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
              applyFlattenAndPad(GV, ScaledL, ScaledN, &LI, &SE);
              Modified = true;
            }
          } else {
            errs()
                << "  [Padding] >>> No padding beneficial (all scores <= 0)\n";
          }
        }
      }
    }
  }

  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
