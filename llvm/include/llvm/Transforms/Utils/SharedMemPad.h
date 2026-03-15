#ifndef LLVM_TRANSFORMS_UTILS_SHARED_MEM_PAD_H
#define LLVM_TRANSFORMS_UTILS_SHARED_MEM_PAD_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SharedMemPass : public PassInfoMixin<SharedMemPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SHARED_MEM_PAD_H