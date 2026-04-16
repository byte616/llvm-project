#include "llvm/Transforms/Utils/SharedMemPad.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BlockGridDimensionAnalysis.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <cstdint>
#include <functional>
#include <map>
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

  DimStrides visitAddExpr(const SCEVAddExpr *Expr) {
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
static void applyFlattenAndPad(GlobalVariable *GV, int64_t L, int64_t N) {
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

  std::function<void(Value *)> Collect = [&](Value *V) {
    for (User *U : V->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        GEPsToReplace.push_back(GEP);
      } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->getOpcode() == Instruction::GetElementPtr) {
          // CE is itself a GEP — collect it for constant-folded replacement.
          CEGEPsToReplace.push_back(CE);
        }
        // Always recurse: downstream GEP *instructions* may use this CE
        // as their pointer operand (e.g. CE GEP → GEP instruction chain).
        Collect(CE);
      } else if (auto *Cast = dyn_cast<AddrSpaceCastInst>(U)) {
        Collect(Cast);
      } else if (auto *Cast = dyn_cast<BitCastInst>(U)) {
        Collect(Cast);
      } else if (auto *LI = dyn_cast<LoadInst>(U)) {
        // Direct load from base pointer (flat index 0)
        DirectAccessesToReplace.push_back(LI);
      } else if (auto *SI = dyn_cast<StoreInst>(U)) {
        // Direct store to base pointer (flat index 0)
        // Make sure V is the pointer operand, not the value being stored
        if (SI->getPointerOperand() == V) {
          DirectAccessesToReplace.push_back(SI);
        }
      }
      // Note: PHINode and SelectInst are not handled — those would require
      // more complex analysis to track all incoming values.
    }
  };
  Collect(GV);

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
      // Convert byte offset to element index
      FlatIdx =
          B.CreateSDiv(ByteOff, ConstantInt::get(I64Ty, ElemSize), "elem.idx");
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
        FlatIdx = B.CreateAdd(
            FlatIdx, B.CreateMul(Idx, ConstantInt::get(I64Ty, Stride), "fmul"),
            "flat");
      }
    }

    // Apply padding: padded = flat + N * udiv(flat, L).
    Value *Pad = B.CreateUDiv(FlatIdx, ConstantInt::get(I64Ty, L), "pad");
    Value *ScaledPad = (N == 1) ? Pad
                                : B.CreateMul(Pad, ConstantInt::get(I64Ty, N),
                                              "scaled.pad");
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

static int computeBankConflict(DimStrides S, int BlockDimX, int BlockDimY) {
  int Banks[32] = {0};
  int MaxConflict = 0;
  std::set<int64_t> Address;
  for (int T = 0; T < 32; ++T) {
    int ThreadX = T % BlockDimX;
    int ThreadY = T / BlockDimX;
    int64_t ElementIndex = ThreadX * S.Tx + ThreadY * S.Ty;
    if (!Address.count(ElementIndex)) {
      Address.insert(ElementIndex);
      // Correct modulo for negative numbers: ((x % 32) + 32) % 32
      int Bank = (int)(((ElementIndex % 32) + 32) % 32);
      Banks[Bank]++;
      MaxConflict = std::max(MaxConflict, Banks[Bank]);
    }
  }
  return MaxConflict;
}

// Compute average bank conflict across one full cycle of warps for stride S
// **with** padding period L applied.
//
// When we pad every L logical elements (insert 1 extra physical element), the
// physical address of logical element i is:
//   physical(i) = i + floor(i / L)
//
// We simulate lcm(32*|S|, L) / (32*|S|) warps -- the minimum number of warps
// needed for the bank pattern to repeat -- and return the average per-warp
// max-bank-conflict count.
//
// Returns a double so the caller can compare fractional averages.
static double computeBankConflictWithPadding(DimStrides S, int BlockDimX,
                                             int BlockDimY, int64_t PaddingL,
                                             int64_t PadN = 1) {

  // Dominant stride: prefer Tx, then Ty, then Tz.
  int64_t DomStride = (S.Tx != 0)   ? std::abs(S.Tx)
                      : (S.Ty != 0) ? std::abs(S.Ty)
                                    : std::abs(S.Tz);
  if (DomStride == 0)
    return 1.0; // broadcast -- always 1 (no conflict)

  // Cycle length in warps: one full padding period covers S*L logical elements.
  // Each warp covers 32*S elements, so: (S*L) / (32*S) = L/32.
  int64_t CycleWarps = PaddingL / 32;

  double TotalConflict = 0.0;
  for (int64_t W = 0; W < CycleWarps; ++W) {
    int Banks[32] = {0};
    int MaxConflict = 0;
    std::set<int64_t> Seen;
    for (int T = 0; T < 32; ++T) {
      int ThreadX = T % BlockDimX;
      int ThreadY = T / BlockDimX;
      // Use absolute value of dominant stride for warp offset calculation,
      // but preserve sign in thread-local calculation for correct relative
      // positioning
      int64_t Logical =
          W * 32 * DomStride + std::abs(ThreadX * S.Tx + ThreadY * S.Ty);
      if (Seen.count(Logical))
        continue;
      Seen.insert(Logical);
      int64_t Physical = Logical + PadN * (Logical / PaddingL);
      // Correct modulo for negative numbers (though Logical should be
      // non-negative now)
      int Bank = (int)(((Physical % 32) + 32) % 32);
      Banks[Bank]++;
      MaxConflict = std::max(MaxConflict, Banks[Bank]);
    }
    TotalConflict += MaxConflict;
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
              int MaxConflict = computeBankConflict(S, BlockDimX, BlockDimY);
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

        AccessInfo Info = {&I, S, Case, Weight, ConflictCount};
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
      };
      std::vector<StrideEntry> Entries;

      auto CollectEntries = [&](const std::vector<AccessInfo> &Accesses) {
        for (const auto &Acc : Accesses) {
          // Pick the dominant stride: prefer Tx, then Ty, then Tz
          int64_t S = Acc.Strides.Tx != 0   ? Acc.Strides.Tx
                      : Acc.Strides.Ty != 0 ? Acc.Strides.Ty
                                            : Acc.Strides.Tz;
          Entries.push_back({S, Acc.ConflictCount, Acc.Weight, Acc.Strides});
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
        auto GCD = [](int64_t A, int64_t B) -> int64_t {
          A = std::abs(A);
          B = std::abs(B);
          while (B) {
            A %= B;
            std::swap(A, B);
          }
          return A;
        };
        auto LCM = [&GCD](int64_t A, int64_t B) -> int64_t {
          if (A == 0 || B == 0)
            return 0;
          return (A / GCD(A, B)) * B;
        };

        // Collect candidate (L, N) pairs from even strides that currently
        // conflict.  For each stride, L = lcm(|S|, 32).
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
        std::set<std::pair<int64_t, int64_t>> Candidates; // {L, N}
        for (auto &E : Entries) {
          int64_t AbsStride = std::abs(E.Stride);
          if (AbsStride <= 1 || AbsStride % 2 != 0)
            continue;
          int64_t IdealL = LCM(AbsStride, 32);
          if (IdealL <= 0)
            continue;
          Candidates.insert({IdealL, PadN});
        }

        if (Candidates.empty()) {
          errs() << "  [Padding] No conflicting even strides. No padding "
                    "needed.\n";
        } else {
          // Evaluate each candidate (L, N) across ALL accesses.
          struct LNScore {
            int64_t L;
            int64_t N;
            double Score;
          };
          std::vector<LNScore> Scores;
          for (auto &[L, N] : Candidates) {
            double Score = 0.0;
            for (auto &E : Entries) {
              double W = (double)E.Weight / (double)TotalWeight;
              double After = computeBankConflictWithPadding(
                  E.FullStrides, BlockDimX, BlockDimY, L, N);
              Score += W * ((double)E.ConflictBefore - After);
            }
            Scores.push_back({L, N, Score});
            errs() << "  [Padding] Candidate L=" << L << " N=" << N
                   << " Score=" << format("%.4f", Score) << "\n";
          }

          // Pick (L, N) with maximum score (must be > 0 to be beneficial).
          // Among equal scores, prefer smaller N (less memory overhead).
          int64_t BestL = 0, BestN = 0;
          double BestScore = 0.0;
          for (auto &S : Scores) {
            if (S.Score > BestScore ||
                (S.Score == BestScore && S.N < BestN)) {
              BestScore = S.Score;
              BestL = S.L;
              BestN = S.N;
            }
          }
          if (BestL > 0) {
            errs() << "  [Padding] >>> Recommended pad period: L=" << BestL
                   << " N=" << BestN
                   << " (Score=" << format("%.4f", BestScore) << ")\n";
            // Apply the transformation: flatten the array and insert
            // padded index = flat + N * floor(flat/L) for every GEP.
            if (auto *GV = dyn_cast<GlobalVariable>(BaseVar)) {
              applyFlattenAndPad(GV, BestL, BestN);
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
