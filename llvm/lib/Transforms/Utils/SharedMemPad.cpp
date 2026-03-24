#include "llvm/Transforms/Utils/SharedMemPad.h"
#include "llvm/Transforms/Utils/BlockGridDimensionAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
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
#include "llvm/Support/Format.h"

#include <cstdint>
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
    cl::value_desc("filename"),
    cl::init(""));

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
  // --- Load blockDim from JSON file if -cuda-blockdim-file was specified ---
  // Priority: JSON file > reqntidx attr > heuristic fallback
  int BlockDimX = 32, BlockDimY = 1, BlockDimZ = 1;
  bool HasKnownBlockDim = false;

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
              errs() << "[SharedMemPass] Matched device '"
                                << DeviceName << "' to stub '" << StubName
                                << "' via suffix '" << Suffix << "'\n";
              break;
            }
          }
        }
      }

      if (It != BGDs.end()) {
        auto &BD = It->second.BlockDim;
        if (BD.X) BlockDimX = static_cast<int>(*BD.X);
        if (BD.Y) BlockDimY = static_cast<int>(*BD.Y);
        if (BD.Z) BlockDimZ = static_cast<int>(*BD.Z);
        HasKnownBlockDim = true;
        errs() << "[SharedMemPass] Loaded blockDim from JSON for '"
                          << DeviceName << "': (" << BlockDimX << ", "
                          << BlockDimY << ", " << BlockDimZ << ")\n";
      } else {
        errs() << "[SharedMemPass] No JSON entry found for '"
                          << DeviceName << "'\n";
      }
    }
  }

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
            if (!HasKnownBlockDim && (S.Ty != 0 || S.Tz != 0)) {
              Case = StrideCase::UNKNOWN_CASE;
            } else {
              // For pure 1D access (only Tx), warp always fills X-dimension,
              // so BlockDimX=32 is safe even when blockDim is not known.
              int EffBlockDimX = HasKnownBlockDim ? BlockDimX : 32;
              int MaxConflict = computeBankConflict(S, EffBlockDimX, BlockDimY);
              ConflictCount = MaxConflict;

              if (S.Tx == 0 && S.Ty == 0 && S.Tz == 0) {
                Case = StrideCase::BROADCAST;
                ConflictCount = 0;
              } else if (MaxConflict <= 1) {
                Case = StrideCase::STRIDE_ODD; // No conflict
                ConflictCount = 0;
              } else {
                Case = StrideCase::STRIDE_EVEN;
              }
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

    // -----------------------------------------------------------------------
    // Optimal Padding L Calculation (Benefit Maximization)
    //
    // For each even-stride access (which has bank conflicts), the "ideal" pad
    // period L that eliminates conflicts for that stride is:
    //   L = lcm(S, 32)
    //
    // We group accesses by their ideal L, then pick the L with the highest:
    //   Benefit(L) = Σ  Weight_i × ConflictCount_i    (for strides w/ idealL == L)
    //
    // Only applies when the variable has no UNKNOWN accesses.
    // -----------------------------------------------------------------------
    if (!Info.HasUnknownAccess) {
      // Aggregate all accesses (loads + stores) into a single list of
      // (DominantStride, ConflictCount, Weight) tuples.
      struct StrideEntry {
        int64_t Stride;     // Dominant stride in elements (Tx if != 0, else Ty)
        int ConflictCount;
        uint64_t Weight;
      };
      std::vector<StrideEntry> Entries;

      auto CollectEntries = [&](const std::vector<AccessInfo> &Accesses) {
        for (const auto &Acc : Accesses) {
          // Use dominant stride (Tx first, then Ty)
          int64_t S = Acc.Strides.Tx != 0 ? Acc.Strides.Tx : Acc.Strides.Ty;
          Entries.push_back({S, Acc.ConflictCount, Acc.Weight});
        }
      };
      CollectEntries(Info.Loads);
      CollectEntries(Info.Stores);

      // Compute total weight for normalization (for display only)
      uint64_t TotalWeight = 0;
      for (auto &E : Entries)
        TotalWeight += E.Weight;

      if (TotalWeight == 0) {
        errs() << "  [Padding] No accesses found.\n";
      } else {
        // For each unique even stride > 1, compute its ideal L = lcm(S, 32).
        // Collect candidate L values.
        // gcd and lcm:
        auto GCD = [](int64_t A, int64_t B) -> int64_t {
          A = std::abs(A);
          B = std::abs(B);
          while (B) { A %= B; std::swap(A, B); }
          return A;
        };
        auto LCM = [&GCD](int64_t A, int64_t B) -> int64_t {
          if (A == 0 || B == 0) return 0;
          return (A / GCD(A, B)) * B;
        };

        // Map: L -> Benefit(L)
        std::map<int64_t, double> BenefitMap;

        for (auto &E : Entries) {
          // Only even strides > 1 have conflicts that need padding
          if (E.Stride <= 1 || E.Stride % 2 != 0)
            continue;

          int64_t IdealL = LCM(E.Stride, 32);
          if (IdealL <= 0)
            continue;

          // Proportion of total accesses this entry represents
          double Proportion = (double)E.Weight / (double)TotalWeight;
          BenefitMap[IdealL] += Proportion * E.ConflictCount;
        }

        if (BenefitMap.empty()) {
          errs() << "  [Padding] No conflicting even strides found. No padding needed.\n";
        } else {
          // Find L with maximum benefit
          int64_t BestL = 0;
          double BestBenefit = 0.0;
          for (auto &[L, Benefit] : BenefitMap) {
            errs() << "  [Padding] Candidate L=" << L
                   << " Benefit=" << format("%.4f", Benefit) << "\n";
            if (Benefit > BestBenefit) {
              BestBenefit = Benefit;
              BestL = L;
            }
          }
          errs() << "  [Padding] >>> Recommended pad period: L=" << BestL
                 << " (Benefit=" << format("%.4f", BestBenefit) << ")\n";
        }
      }
    }
  }

  return PreservedAnalyses::all();
}
