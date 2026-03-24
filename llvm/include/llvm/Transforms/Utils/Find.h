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

#ifndef UTILS_FIND_H
#define UTILS_FIND_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include <type_traits>

namespace autotex {

/// \brief Find the first call to a function with the name \p FuncName in the
/// range of instructions or users \p Range; `nullptr` if not found.
template <typename RangeTy>
llvm::CallInst *findCallTo(const RangeTy &Range, llvm::StringRef FuncName) {
  if constexpr (std::is_pointer_v<std::decay_t<decltype(*Range.begin())>>) {
    for (auto *Element : Range) {
      if (auto *Call = llvm::dyn_cast<llvm::CallInst>(Element)) {
        if (Call->getCalledFunction()->getName() == FuncName) {
          return Call;
        }
      }
    }
  } else {
    for (auto &Element : Range) {
      if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&Element)) {
        if (Call->getCalledFunction()->getName() == FuncName) {
          return Call;
        }
      }
    }
  }
  return nullptr;
}

/// \brief Find the reference to the first call to a function with the name
/// \p FuncName in the range of instructions or users \p Range.
///
/// This function asserts that the call is found.
template <typename RangeTy>
llvm::CallInst &findCallRefTo(const RangeTy &Range, llvm::StringRef FuncName) {
  auto *Call = findCallTo(Range, FuncName);
  assert(Call && "Call to function not found.");
  return *Call;
}

/// \brief Find all calls to a function with the name \p FuncName in the range
/// of instructions or users \p Range; `nullptr` if not found.
template <typename RangeTy>
llvm::SmallVector<llvm::CallInst *> findAllCallsTo(const RangeTy &Range,
                                                   llvm::StringRef FuncName) {
  llvm::SmallVector<llvm::CallInst *> Calls;
  if constexpr (std::is_pointer_v<std::decay_t<decltype(*Range.begin())>>) {
    for (auto *Element : Range) {
      if (auto *Call = llvm::dyn_cast<llvm::CallInst>(Element);
          Call && Call->getCalledFunction()->getName() == FuncName) {
        Calls.push_back(Call);
      }
    }
  } else {
    for (auto &Element : Range) {
      if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&Element);
          Call && Call->getCalledFunction()->getName() == FuncName) {
        Calls.push_back(Call);
      }
    }
  }
  return Calls;
}

} // namespace autotex

#endif // UTILS_FIND_H