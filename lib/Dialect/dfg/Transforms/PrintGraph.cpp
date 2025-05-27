/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/Interfaces/EdgeOp.h"
#include "dfg-mlir/Dialect/dfg/Interfaces/GraphOp.h"
#include "dfg-mlir/Dialect/dfg/Interfaces/NodeOp.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <map>
#include <memory>
#include <mlir/Analysis/CallGraph.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGPRINTGRAPH
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace dfg;

namespace {
constexpr const char* kDotNodeName =
    R"({{GraphName}}_{{GraphNumber}}_{{NodeName}}_{{NodeNumber}})";
constexpr const char* kDotNode =
    R"({{GraphName}}_{{GraphNumber}}_{{NodeName}}_{{NodeNumber}} [label="{{NodeName}}", shape=box];
)";
constexpr const char* kDotSubgraph =
    R"(subgraph cluster_{{GraphName}}_{{GraphNumber}} {
  label="{{GraphName}}";
{{Content}}
}
)";
} // namespace

namespace {
struct DfgPrintGraphPass
        : public dfg::impl::DfgPrintGraphBase<DfgPrintGraphPass> {
    void runOnOperation() override;
    void getGraph(ModuleOp module);

    // String helpers
    // replace all string with another
    std::string
    replaceAll(std::string str, const std::string &from, const std::string &to)
    {
        size_t start_pos = 0;
        while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
        return str;
    }
    // Indentation helper
    std::string indentContent(const std::string &content)
    {
        std::string indented;
        std::istringstream iss(content);
        std::string line;

        while (std::getline(iss, line)) indented += "  " + line + "\n";
        return indented;
    };
    // Remove blank lines
    std::string removeBlankLines(const std::string &input)
    {
        std::stringstream ss(input);
        std::string line;
        std::string result;
        while (std::getline(ss, line)) {
            // If this line is blank or only contains spaces
            bool isBlank = true;
            for (char c : line) {
                if (!std::isspace(c)) {
                    isBlank = false;
                    break;
                }
            }
            if (!isBlank) result += line + '\n';
        }
        return result;
    }

private:
    std::map<std::string, std::string> dotNodeNameMap;
    std::map<std::string, std::string> dotNodeMap;
    std::map<std::string, unsigned> dotNodeNumMap;
    llvm::DenseMap<Operation*, std::string> dotNodeOpMap;
    std::map<std ::string, std::string> dotGraphMap;
    std::map<std::string, unsigned> dotGraphNumMap;
    std::map<std::string, SmallVector<std::string>> dotSubgraphInputMap;
    std::map<std::string, SmallVector<std::string>> dotSubgraphOutputMap;
    std::map<std::string, SmallVector<std::string>> dotSubgraphNodeTemplateMap;
    llvm::DenseMap<Operation*, std::string> dotSubgraphNameMap;
    llvm::DenseMap<Value, std::string> graphPortMap;
    SmallVector<std::string> topGraphs;
};
} // namespace

void DfgPrintGraphPass::getGraph(ModuleOp module)
{
    // Process the graph
    for (auto &opi : module.getBodyRegion().front()) {
        if (isa<GraphOpInterface>(opi)) {
            // Each node/subgraph in a region must be defined before usage, so
            // everything is inside the map. Because it's instances of
            // node/subgraph, increase the index number. For arrows pointing to
            // subgraph, save the index of port value and it's usage in node.
            auto graphOp = cast<GraphOpInterface>(opi);
            auto graphOpAsNode = cast<NodeOpInterface>(opi);
            auto graphName = graphOp.getGraphName();
            auto isSubGraph = graphOp.isSubGraph();
            std::string graphStr(kDotSubgraph);
            graphStr = replaceAll(graphStr, "{{GraphName}}", graphName);
            std::string graphContentStr;
            // Input ports
            SmallVector<std::string> outputPortsStr;
            for (unsigned i = 0; i < graphOpAsNode.getNumInputPorts(); ++i) {
                std::string inPortStr = "in_" + std::to_string(i);
                std::string inGraphPortStr =
                    graphName + "_{{GraphNumber}}_" + inPortStr;
                std::string inPortNodeStr = inGraphPortStr + " [label=\""
                                            + inPortStr
                                            + "\", shape=ellipse];\n";
                auto inPort = graphOpAsNode.getInputPort(i);
                graphContentStr += indentContent(inPortNodeStr);
                graphPortMap.insert({inPort, inGraphPortStr});
                outputPortsStr.push_back(inGraphPortStr);
            }
            if (isSubGraph) {
                auto subGraphNameStr = graphName + "_{{GraphNumber}}";
                dotSubgraphOutputMap.insert({subGraphNameStr, outputPortsStr});
            }
            // Set the in ports of a top graph as source nodes
            std::string graphSourceNodes = "{rank=source;{{SourceNodes}}}\n";
            std::string sourceNodesStr;
            for (auto inPort : graphOpAsNode.getInputPorts()) {
                std::string inPortStr = graphPortMap[inPort];
                sourceNodesStr += (" " + inPortStr + ";");
            }
            graphContentStr += indentContent(replaceAll(
                graphSourceNodes,
                "{{SourceNodes}}",
                sourceNodesStr));
            // Output ports
            SmallVector<std::string> inputPortsStr;
            for (unsigned i = 0; i < graphOpAsNode.getNumOutputPorts(); ++i) {
                std::string outPortStr = "out_" + std::to_string(i);
                std::string outGraphPortStr =
                    graphName + "_{{GraphNumber}}_" + outPortStr;
                std::string outPortNodeStr = outGraphPortStr + " [label=\""
                                             + outPortStr
                                             + "\", shape=ellipse];\n";
                auto outPort = graphOpAsNode.getOutputPort(i);
                graphContentStr += indentContent(outPortNodeStr);
                graphPortMap.insert({outPort, outGraphPortStr});
                inputPortsStr.push_back(outGraphPortStr);
            }
            if (isSubGraph) {
                auto subGraphNameStr = graphName + "_{{GraphNumber}}";
                dotSubgraphInputMap.insert({subGraphNameStr, inputPortsStr});
            }
            // Set the out ports of a top graph as sink nodes
            std::string graphSinkNodes = "{rank=sink;{{SinkNodes}}}\n";
            std::string sinkNodesStr;
            for (auto outPort : graphOpAsNode.getOutputPorts()) {
                std::string outPortStr = graphPortMap[outPort];
                sinkNodesStr += (" " + outPortStr + ";");
            }
            graphContentStr += indentContent(
                replaceAll(graphSinkNodes, "{{SinkNodes}}", sinkNodesStr));
            // Nodes
            SmallVector<std::string> templateNodeStr;
            for (auto graphContent : graphOp.getGraphNodes()) {
                auto graphContentNode = cast<NodeOpInterface>(graphContent);
                auto graphContentNodeName = graphContentNode.getNodeName();
                auto graphContentNodeStr = dotNodeMap[graphContentNodeName];
                auto graphContentNodeNameStr =
                    dotNodeNameMap[graphContentNodeName];
                auto graphContentNodeNumber =
                    dotNodeNumMap[graphContentNodeNameStr]++;
                graphContentNodeNameStr = replaceAll(
                    graphContentNodeNameStr,
                    "{{GraphName}}",
                    graphName);
                graphContentNodeNameStr = replaceAll(
                    graphContentNodeNameStr,
                    "{{NodeNumber}}",
                    std::to_string(graphContentNodeNumber));
                graphContentNodeStr =
                    replaceAll(graphContentNodeStr, "{{GraphName}}", graphName);
                graphContentNodeStr = replaceAll(
                    graphContentNodeStr,
                    "{{NodeNumber}}",
                    std::to_string(graphContentNodeNumber));
                graphContentStr += indentContent(graphContentNodeStr);
                dotNodeOpMap.insert({graphContent, graphContentNodeNameStr});
            }
            dotSubgraphNodeTemplateMap.insert({graphName, templateNodeStr});
            // Subgraphs
            for (auto graphContent : graphOp.getGraphSubGs()) {
                auto graphContentSubgraph =
                    cast<GraphOpInterface>(graphContent);
                auto graphContentSubgraphName =
                    graphContentSubgraph.getGraphName();
                auto graphContentSubgraphNumber =
                    dotGraphNumMap[graphContentSubgraphName]++;
                std::string graphContentSubgraphStr =
                    dotGraphMap[graphContentSubgraphName];
                graphContentSubgraphStr = replaceAll(
                    graphContentSubgraphStr,
                    "{{GraphNumber}}",
                    std::to_string(graphContentSubgraphNumber));
                graphContentStr += graphContentSubgraphStr;
                dotSubgraphNameMap.insert(
                    {graphContent,
                     graphContentSubgraphName + "_"
                         + std::to_string(graphContentSubgraphNumber)});
                // Add node number
                auto nodeTemplates =
                    dotSubgraphNodeTemplateMap[graphContentSubgraphName];
                for (auto temp : nodeTemplates) {
                    auto nodeNumer = dotNodeNumMap[temp]++;
                    auto nodeTemplate = "{{" + temp + "}}";
                    graphContentStr = replaceAll(
                        graphContentStr,
                        nodeTemplate,
                        std::to_string(nodeNumer));
                }
                // Process the input/output map
                for (auto map : dotSubgraphInputMap) {
                    if (map.first
                        == graphContentSubgraphName + "_{{GraphNumber}}") {
                        auto newSubgraphNodeName =
                            graphContentSubgraphName + "_"
                            + std::to_string(graphContentSubgraphNumber);
                        SmallVector<std::string> newSubgraphNodeInputs;
                        for (auto input : map.second) {
                            newSubgraphNodeInputs.push_back(replaceAll(
                                input,
                                "{{GraphNumber}}",
                                std::to_string(graphContentSubgraphNumber)));
                        }
                        dotSubgraphInputMap.insert(
                            {newSubgraphNodeName, newSubgraphNodeInputs});
                    }
                }
                for (auto map : dotSubgraphOutputMap) {
                    if (map.first
                        == graphContentSubgraphName + "_{{GraphNumber}}") {
                        auto newSubgraphNodeName =
                            graphContentSubgraphName + "_"
                            + std::to_string(graphContentSubgraphNumber);
                        SmallVector<std::string> newSubgraphNodeOutputs;
                        for (auto output : map.second) {
                            newSubgraphNodeOutputs.push_back(replaceAll(
                                output,
                                "{{GraphNumber}}",
                                std::to_string(graphContentSubgraphNumber)));
                        }
                        dotSubgraphOutputMap.insert(
                            {newSubgraphNodeName, newSubgraphNodeOutputs});
                    }
                }
            }
            // Edges
            for (auto graphContent : graphOp.getGraphEdges()) {
                auto graphContentChannel = cast<ChannelOp>(graphContent);
                std::string graphContentEdgeStr = "{{INPUT}} -> {{OUTPUT}};";
                // Input
                auto inConnection = graphContentChannel.getInputConnection();
                std::string edgeInputStr;
                if (isa<GraphOpInterface>(inConnection)) {
                    auto subGraphNameStr = dotSubgraphNameMap[inConnection];
                    auto subGraphInPorts = dotSubgraphInputMap[subGraphNameStr];
                    auto inConnectionGraphAsNode =
                        dyn_cast<NodeOpInterface>(inConnection);
                    for (auto [inPort, inPortName] : llvm::zip(
                             inConnectionGraphAsNode.getOutputPorts(),
                             subGraphInPorts)) {
                        if (inPort == graphContentChannel.getInChan())
                            edgeInputStr = inPortName;
                    }
                } else if (
                    auto connectOp = dyn_cast<ConnectInputOp>(inConnection)) {
                    edgeInputStr = graphPortMap[connectOp.getRegionPort()];
                } else {
                    edgeInputStr = dotNodeOpMap[inConnection];
                }
                graphContentEdgeStr =
                    replaceAll(graphContentEdgeStr, "{{INPUT}}", edgeInputStr);
                // Output
                auto outConnection = graphContentChannel.getOutputConnection();
                std::string edgeOutputStr;
                if (isa<GraphOpInterface>(outConnection)) {
                    auto subGraphNameStr = dotSubgraphNameMap[outConnection];
                    auto subGraphOutPorts =
                        dotSubgraphOutputMap[subGraphNameStr];
                    auto outConnectionGraphAsNode =
                        dyn_cast<NodeOpInterface>(outConnection);
                    for (auto [outPort, outPortName] : llvm::zip(
                             outConnectionGraphAsNode.getInputPorts(),
                             subGraphOutPorts)) {
                        if (outPort == graphContentChannel.getOutChan())
                            edgeOutputStr = outPortName;
                    }
                } else if (
                    auto connectOp = dyn_cast<ConnectOutputOp>(outConnection)) {
                    edgeOutputStr = graphPortMap[connectOp.getRegionPort()];
                } else {
                    edgeOutputStr = dotNodeOpMap[outConnection];
                }
                graphContentEdgeStr = replaceAll(
                    graphContentEdgeStr,
                    "{{OUTPUT}}",
                    edgeOutputStr);
                graphContentStr += indentContent(graphContentEdgeStr);
            }
            // Add edges that point from region port to instance and vice versa
            // if this is subgraph
            if (isSubGraph) {
                for (auto inPort : graphOpAsNode.getInputPorts()) {
                    std::string graphContentSubgraphInEdge =
                        "{{INPUT}} -> {{OUTPUT}};";
                    auto inputStr = graphPortMap[inPort];
                    graphContentSubgraphInEdge = replaceAll(
                        graphContentSubgraphInEdge,
                        "{{INPUT}}",
                        inputStr);
                    auto inPortUser = *inPort.getUsers().begin();
                    std::string outputStr;
                    if (isa<GraphOpInterface>(inPortUser)) {
                        auto subGraphNameStr = dotSubgraphNameMap[inPortUser];
                        auto subGraphOutPorts =
                            dotSubgraphOutputMap[subGraphNameStr];
                        auto outConnectionGraphAsNode =
                            dyn_cast<NodeOpInterface>(inPortUser);
                        for (auto [outPort, outPortName] : llvm::zip(
                                 outConnectionGraphAsNode.getInputPorts(),
                                 subGraphOutPorts)) {
                            if (outPort == inPort) outputStr = outPortName;
                        }
                    } else {
                        outputStr = dotNodeOpMap[inPortUser];
                    }
                    graphContentSubgraphInEdge = replaceAll(
                        graphContentSubgraphInEdge,
                        "{{OUTPUT}}",
                        outputStr);
                    graphContentStr +=
                        indentContent(graphContentSubgraphInEdge);
                }
                for (auto outPort : graphOpAsNode.getOutputPorts()) {
                    std::string graphContentSubgraphInEdge =
                        "{{INPUT}} -> {{OUTPUT}};";
                    auto outputStr = graphPortMap[outPort];
                    graphContentSubgraphInEdge = replaceAll(
                        graphContentSubgraphInEdge,
                        "{{OUTPUT}}",
                        outputStr);
                    auto outPortUser = *outPort.getUsers().begin();
                    std::string inputStr;
                    if (isa<GraphOpInterface>(outPortUser)) {
                        auto subGraphNameStr = dotSubgraphNameMap[outPortUser];
                        auto subGraphInPorts =
                            dotSubgraphInputMap[subGraphNameStr];
                        auto inConnectionGraphAsNode =
                            dyn_cast<NodeOpInterface>(outPortUser);
                        for (auto [inPort, inPortName] : llvm::zip(
                                 inConnectionGraphAsNode.getOutputPorts(),
                                 subGraphInPorts)) {
                            if (inPort == outPort) inputStr = inPortName;
                        }
                    } else {
                        inputStr = dotNodeOpMap[outPortUser];
                    }
                    graphContentSubgraphInEdge = replaceAll(
                        graphContentSubgraphInEdge,
                        "{{INPUT}}",
                        inputStr);
                    graphContentStr +=
                        indentContent(graphContentSubgraphInEdge);
                }
            }
            // Save to map and if it's top graph, save it
            graphStr = indentContent(
                replaceAll(graphStr, "{{Content}}", graphContentStr));
            dotGraphMap.insert({graphName, graphStr});
            if (!isSubGraph)
                topGraphs.push_back(
                    replaceAll(graphStr, "{{GraphNumber}}", "0"));
        } else {
            // Each node (operator/process) is unique inside module
            auto nodeOp = cast<NodeOpInterface>(opi);
            auto nodeName = nodeOp.getNodeName();
            std::string nodeStr(kDotNode);
            nodeStr = replaceAll(nodeStr, "{{NodeName}}", nodeName);
            dotNodeMap.insert({nodeName, nodeStr});
            std::string nodeNameStr(kDotNodeName);
            nodeNameStr = replaceAll(nodeNameStr, "{{NodeName}}", nodeName);
            dotNodeNameMap.insert({nodeName, nodeNameStr});
        }
    }
}

void DfgPrintGraphPass::runOnOperation()
{
    // Get builtin.module
    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module) signalPassFailure();
    // Gets the nodes and subgraphs
    getGraph(module);
    // Print the graph
    for (auto [i, graph] : llvm::enumerate(topGraphs)) {
        // Create the file
        std::error_code ec;
        std::string dotFileName = "graph_" + std::to_string(i) + ".dot";
        llvm::raw_fd_ostream dotFile(dotFileName, ec, llvm::sys::fs::OF_Text);
        if (ec) signalPassFailure();
        // Print to dot file
        raw_indented_ostream os(dotFile);
        os << "digraph G {\n";
        os.indent() << "rankdir=LR;\n";
        dotFile << removeBlankLines(graph);
        os.unindent() << "}\n";
        dotFile.close();

        if (printToPdf) {
            std::string svgFileName = "graph_" + std::to_string(i) + ".svg";
            std::string pdfFileName = "graph_" + std::to_string(i) + ".pdf";
            std::string commandStr =
                "dot -Tsvg " + dotFileName + " -o " + svgFileName;
            commandStr += " && inkscape " + svgFileName
                          + " --export-filename=" + pdfFileName;
            commandStr += " && rm -f " + dotFileName + " " + svgFileName;
            auto result = std::system(commandStr.c_str());
            if (result != 0) {
                llvm::errs() << "Fail to generate pdf file(s). Make sure you "
                                "have dot and inkscape installed.\n";
                signalPassFailure();
            }
        }

        ++i;
    }
}

std::unique_ptr<Pass> mlir::dfg::createDfgPrintGraphPass()
{
    return std::make_unique<DfgPrintGraphPass>();
}
