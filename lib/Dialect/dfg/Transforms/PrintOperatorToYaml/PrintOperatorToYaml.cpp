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

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGPRINTOPERATORTOYAML
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace dfg;

// YAML file content struct

struct GraphInfo {
    std::string name;
};

struct GraphNodePort {
    std::string name;
    std::string type;
};

struct GraphNode {
    std::string name;
    std::vector<GraphNodePort> ports;
    int execCycles;
};

struct GraphChannel {
    std::string name;
    std::string srcNode;
    std::string srcPort;
    std::string dstNode;
    std::string dstPort;
    int initToken;
};

struct GraphYaml {
    GraphInfo graphInfo;
    std::vector<GraphNode> nodes;
    std::vector<GraphChannel> channels;
};

namespace llvm {
namespace yaml {

template<>
struct MappingTraits<GraphInfo> {
    static void mapping(IO &io, GraphInfo &graphInfo)
    {
        io.mapRequired("name", graphInfo.name);
    }
};

template<>
struct MappingTraits<GraphNodePort> {
    static void mapping(IO &io, GraphNodePort &graphNodePort)
    {
        io.mapRequired("name", graphNodePort.name);
        io.mapRequired("type", graphNodePort.type);
    }
};

template<>
struct MappingTraits<GraphNode> {
    static void mapping(IO &io, GraphNode &graphNode)
    {
        io.mapRequired("name", graphNode.name);
        io.mapRequired("ports", graphNode.ports);
        io.mapRequired("exec_cycles", graphNode.execCycles);
    }
};

template<>
struct MappingTraits<GraphChannel> {
    static void mapping(IO &io, GraphChannel &graphChannel)
    {
        io.mapRequired("name", graphChannel.name);
        io.mapRequired("srcNode", graphChannel.srcNode);
        io.mapRequired("srcPort", graphChannel.srcPort);
        io.mapRequired("dstNode", graphChannel.dstNode);
        io.mapRequired("dstPort", graphChannel.dstPort);
        io.mapRequired("initToken", graphChannel.initToken);
    }
};

template<>
struct MappingTraits<GraphYaml> {
    static void mapping(IO &io, GraphYaml &graphYaml)
    {
        io.mapRequired("graph", graphYaml.graphInfo);
        io.mapRequired("nodes", graphYaml.nodes);
        io.mapRequired("channels", graphYaml.channels);
    }
};

template<>
struct SequenceTraits<std::vector<GraphNodePort>> {
    static size_t size(IO &io, std::vector<GraphNodePort> &seq)
    {
        return seq.size();
    }

    static GraphNodePort &
    element(IO &io, std::vector<GraphNodePort> &seq, size_t index)
    {
        if (index >= seq.size()) seq.resize(index + 1);
        return seq[index];
    }
};

template<>
struct SequenceTraits<std::vector<GraphNode>> {
    static size_t size(IO &io, std::vector<GraphNode> &seq)
    {
        return seq.size();
    }

    static GraphNode &element(IO &io, std::vector<GraphNode> &seq, size_t index)
    {
        if (index >= seq.size()) seq.resize(index + 1);
        return seq[index];
    }
};

template<>
struct SequenceTraits<std::vector<GraphChannel>> {
    static size_t size(IO &io, std::vector<GraphChannel> &seq)
    {
        return seq.size();
    }

    static GraphChannel &
    element(IO &io, std::vector<GraphChannel> &seq, size_t index)
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
    auto module = dyn_cast<ModuleOp>(getOperation());

    // TODO: get nodes and channels from an operator
    std::vector<GraphNode> nodes = {
        {    "src",                  {{"output", "out"}}, 1},
        {"compute", {{"input", "in"}, {"output", "out"}}, 1},
        {   "sink",                    {{"input", "in"}}, 1}
    };
    std::vector<GraphChannel> channels = {
        {"ch0",     "src", "output", "compute", "input", 0},
        {"ch1", "compute", "output",    "sink", "input", 1}
    };

    module.walk([&](OperatorOp operatorOp) {
        auto graphName = operatorOp.getSymName().str();
        GraphYaml graph = {{graphName}, nodes, channels};
        auto fileName = "graph_" + graphName + ".yaml";

        // Emit YAML into separate files
        std::error_code EC;
        llvm::raw_fd_ostream outStream(fileName, EC, llvm::sys::fs::FA_Write);
        if (EC) {
            llvm::errs() << "Error opening file " << fileName << ": "
                         << EC.message() << "\n";
            signalPassFailure();
        }
        llvm::yaml::Output yout(outStream);
        yout << graph;
        outStream.close();
    });
}

std::unique_ptr<Pass> mlir::dfg::createDfgPrintOperatorToYamlPass()
{
    return std::make_unique<DfgPrintOperatorToYamlPass>();
}
