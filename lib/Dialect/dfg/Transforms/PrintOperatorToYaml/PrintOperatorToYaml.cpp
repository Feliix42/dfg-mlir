/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGPRINTOPERATORTOYAML
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace dfg;

struct Item {
    std::string Name;
    int Value;
};

namespace llvm {
namespace yaml {

template<>
struct MappingTraits<Item> {
    static void mapping(IO &io, Item &item)
    {
        io.mapRequired("name", item.Name);
        io.mapRequired("value", item.Value);
    }
};

template<>
struct SequenceTraits<std::vector<Item>> {
    static size_t size(IO &io, std::vector<Item> &seq) { return seq.size(); }

    static Item &element(IO &io, std::vector<Item> &seq, size_t index)
    {
        if (index >= seq.size()) seq.resize(index + 1);
        return seq[index];
    }
};

} // namespace yaml
} // namespace llvm

namespace {
struct DfgPrintOperatorToYamlPass
        : public dfg::impl::DfgPrintOperatorToYamlBase<
              DfgPrintOperatorToYamlPass> {
    void runOnOperation() override;
};
} // namespace

void DfgPrintOperatorToYamlPass::runOnOperation()
{
    std::vector<Item> items = {
        {"Item1", 100},
        {"Item2", 200},
        {"Item3", 300}
    };

    // Create an output stream
    llvm::raw_fd_ostream outStream(
        1,
        false); // 1 is the file descriptor for stdout

    // Emit YAML
    llvm::yaml::Output yout(outStream);
    yout << items;
}

std::unique_ptr<Pass> mlir::dfg::createDfgPrintOperatorToYamlPass()
{
    return std::make_unique<DfgPrintOperatorToYamlPass>();
}
