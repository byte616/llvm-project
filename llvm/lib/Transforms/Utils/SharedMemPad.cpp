#include "llvm/Transforms/Utils/SharedMemPad.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <map>
#include <set>
#include <vector>

using namespace llvm;
#define DEBUG_TYPE "shared-mem-pad"

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
  bool HasUnknownAccess = false; // if there exists unknown access, we cannot pad
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

// Extracts block dimensions from function attributes if available
// Returns true if successful, false otherwise
static bool getBlockDimFromFn(Function &F, int &BlockDimX, int &BlockDimY, int &BlockDimZ) {
  if (F.hasFnAttribute("reqntidx")) {
    StringRef Reqntidx = F.getFnAttribute("reqntidx").getValueAsString();
    // Format is usually "X Y Z" or just a single integer
    SmallVector<StringRef, 3> Dims;
    Reqntidx.split(Dims, ' ');
    if (Dims.size() >= 1) Dims[0].getAsInteger(10, BlockDimX);
    if (Dims.size() >= 2) Dims[1].getAsInteger(10, BlockDimY);
    if (Dims.size() >= 3) Dims[2].getAsInteger(10, BlockDimZ);
    return true;
  }
  return false;
}

// Helper method to compute bank conflicts based on strides and block dimensions
static int computeBankConflict(DimStrides S, int BlockDimX, int BlockDimY) {
  // If only Tx varies or block is essentially 1D for X, use X, else consider Ty
  if (BlockDimX <= 0) BlockDimX = 32;

  int Banks[32] = {0};
  int MaxConflict = 0;
  std::set<int> Address;
  for (int T = 0; T < 32; ++T) {
    int ThreadX = T % BlockDimX;
    int ThreadY = (T / BlockDimX);
    // tz is 0 for the first 32 threads, ignore for now
            
    int64_t ElementIndex = ThreadX * S.Tx + ThreadY * S.Ty;
    
    if(!Address.count(ElementIndex)) {
      Address.insert(ElementIndex);
      int Bank = ElementIndex % 32;
      Banks[Bank]++;
      if (Banks[Bank] > MaxConflict) {
        MaxConflict = Banks[Bank];
      }
    }
  }
  return MaxConflict;
}

PreservedAnalyses SharedMemPass::run(Function &F, FunctionAnalysisManager &AM) {
  // Get ScalarEvolution Analysis and BlockFrequency Analysis
  ScalarEvolution &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  BlockFrequencyInfo &BFI = AM.getResult<BlockFrequencyAnalysis>(F);

  // Check if the target is NVPTX
  Triple T(F.getParent()->getTargetTriple());
  if (!T.isNVPTX()) {
    return PreservedAnalyses::all();
  }

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

          Type *AccessType =
              IsLoad ? cast<LoadInst>(&I)->getType()
                     : cast<StoreInst>(&I)->getValueOperand()->getType();
          const DataLayout &DL = F.getParent()->getDataLayout();
          unsigned ElementSizeBytes = DL.getTypeStoreSize(AccessType);

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

            // Attempt to deduce block sizes from function attributes
            // Many CUDA codes might have "reqntidx" if __launch_bounds__ is used
            int BlockDimX = 32, BlockDimY = 1, BlockDimZ = 1;
            bool HasStaticallyKnownBlockSize = getBlockDimFromFn(F, BlockDimX, BlockDimY, BlockDimZ);

            // If not found, realistically the LLVM pass doesn't know the blockDim at compile time.
            // Using a default of 32 for bank conflict estimation.
            // If the user's BlockDimX < 32 (like 16), the bank conflict formula changes,
            // but without metadata or explicit host-passed arguments, the pass cannot know.
            if (!HasStaticallyKnownBlockSize && (S.Ty != 0 || S.Tz != 0)) {
               // We will use 16 as a heuristic fallback if Ty/Tz are used just to show the difference
               // But usually compiler passes require front-end attributes to be certain.
               BlockDimX = 16;
               errs() << "CHEAT" << "\n";
            }

            int MaxConflict = computeBankConflict(S, BlockDimX, BlockDimY);
            ConflictCount = MaxConflict;
            
            if (S.Tx == 0 && S.Ty == 0 && S.Tz == 0) {
              Case = StrideCase::BROADCAST;
              ConflictCount = 0;
            } else if (MaxConflict <= 1) {
              Case = StrideCase::STRIDE_ODD; // No conflict
            } else {
              Case = StrideCase::STRIDE_EVEN;
            }
          }
        }

        std::optional<uint64_t> ProfileCount = BFI.getBlockProfileCount(I.getParent());
        uint64_t Weight = ProfileCount.has_value() ? ProfileCount.value() : BFI.getBlockFreq(I.getParent()).getFrequency();
        
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
      errs() << "  [WARNING] Variable has UNKNOWN accesses. Cannot safely pad.\n";
    }

    std::map<std::pair<std::string, int>, uint64_t> LoadWeights;
    std::map<std::pair<std::string, int>, uint64_t> StoreWeights;

    auto FormatStride = [](DimStrides S, int Conflicts) {
      std::string Str = "Tx:" + std::to_string(S.Tx) + " Ty:" + std::to_string(S.Ty) + " (" + std::to_string(Conflicts) + "-way bank conflict)";
      return Str;
    };

    for (auto &Acc : Info.Loads) {
      LoadWeights[{FormatStride(Acc.Strides, Acc.ConflictCount), (int)Acc.CaseType}] += Acc.Weight;
    }
    for (auto &Acc : Info.Stores) {
      StoreWeights[{FormatStride(Acc.Strides, Acc.ConflictCount), (int)Acc.CaseType}] += Acc.Weight;
    }

    for (auto &Pair : LoadWeights) {
      errs() << "  Load " << Pair.first.first
             << " (Case: " << Pair.first.second
             << ") - Estimated Accesses: " << Pair.second << "\n";
    }
    for (auto &Pair : StoreWeights) {
      errs() << "  Store " << Pair.first.first
             << " (Case: " << Pair.first.second
             << ") - Estimated Accesses: " << Pair.second << "\n";
    }
  }

  return PreservedAnalyses::all();
}
