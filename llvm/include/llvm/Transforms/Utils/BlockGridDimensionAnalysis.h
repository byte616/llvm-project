//===----------------------------------------------------------------------===//
//
// Copyright (c) 2025 Lai-YT
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef ANALYSIS_BLOCKGRIDDIMENSIONANALYSIS_H
#define ANALYSIS_BLOCKGRIDDIMENSIONANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Error.h"
#include <map>
#include <optional>
#include <string>

namespace llvm {

class Module;
class raw_ostream;

} // namespace llvm

namespace autotex {

struct Dim3 {
  std::optional<unsigned> X;
  std::optional<unsigned> Y;
  std::optional<unsigned> Z;
};

struct BlockGridDim {
  Dim3 BlockDim;
  Dim3 GridDim;
};

/// \brief A host-side analysis pass that identifies the block and grid
/// dimensions of the kernel launch.
class BlockGridDimensionAnalysis
    : public llvm::AnalysisInfoMixin<BlockGridDimensionAnalysis> {
public:
  // NOTE: Can't use StringRef due to lifetime issues.
  // std::map therefore is used because std::string isn't naturally hashable
  // with DenseMap.
  using Result = std::map<std::string, BlockGridDim>;
  Result run(llvm::Module &, llvm::ModuleAnalysisManager &);

private:
  friend llvm::AnalysisInfoMixin<BlockGridDimensionAnalysis>;
  static llvm::AnalysisKey Key;
};

class BlockGridDimensionAnalysisPrinter
    : public llvm::PassInfoMixin<BlockGridDimensionAnalysisPrinter> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  explicit BlockGridDimensionAnalysisPrinter(llvm::raw_ostream &OS) : OS(OS) {}

private:
  llvm::raw_ostream &OS;
};

/// \brief Serialize the block and grid dimensions to JSON.
class BlockGridDimensionAnalysisJSONExporter
    : public llvm::PassInfoMixin<BlockGridDimensionAnalysisJSONExporter> {
public:
  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);

  explicit BlockGridDimensionAnalysisJSONExporter(llvm::raw_ostream &OS)
      : OS(OS) {}

private:
  llvm::raw_ostream &OS;
};

/// \brief Deserialize the block and grid dimensions from JSON.
class BlockGridDimensionAnalysisJSONImporter {
public:
  /// \returns If the \p JSON is empty, it returns an empty block and grid
  /// dimension.
  static llvm::Expected<BlockGridDimensionAnalysis::Result>
  fromJSON(llvm::StringRef JSON);

  /// \returns If the file is empty, it returns an empty block and grid
  /// dimension.
  static llvm::Expected<BlockGridDimensionAnalysis::Result>
  fromFile(llvm::StringRef Filename);
};

} // namespace autotex

#endif // ANALYSIS_BLOCKGRIDDIMENSIONANALYSIS_H