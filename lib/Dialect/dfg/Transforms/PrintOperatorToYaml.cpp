/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/PrintOperatorToYaml.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "yaml-test"

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
    std::string type;
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

struct MappingInfo {
    std::string nodeName;
    GraphNodePort port;
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
        io.mapRequired("type", graphNode.type);
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
public:
    void runOnOperation() override;

private:
    void getGraphNodes(OperatorOp operatorOp);
    void getGraphNodesFromOp(
        Operation* op,
        std::vector<GraphNode> &nodes,
        size_t idxBias);
    void getGraphChannels();
    GraphNode findNode(std::string nodeName)
    {
        for (auto node : graphNodes)
            if (node.name == nodeName) return node;
    }
    void analyseChannelLoops()
    {
        for (auto channel : graphChannels) {
            adjList[channel.srcNode].push_back(channel.dstNode);
            LLVM_DEBUG(
                llvm::dbgs() << "\n Adding " << channel.srcNode << " <-> "
                             << channel.dstNode << " to adjacency list.\n");
        }

        for (auto channel : graphChannels) {
            std::unordered_set<std::string> visited;
            auto node = channel.srcNode;
            if (findNode(node).type == "src") {
                LLVM_DEBUG(
                    llvm::dbgs() << "\nNow checking from " << node << ".\n");
                findAndAddToken(node, visited);
            }
        }

        for (auto channel : graphChannels) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "\nChannel " << channel.srcNode << " <-> " << channel.dstNode
                << " has initToken " << channel.initToken << ".\n");
        }
    }
    void
    findAndAddToken(std::string node, std::unordered_set<std::string> &visited)
    {
        visited.insert(node);
        for (auto neighbor : adjList[node]) {
            LLVM_DEBUG(
                llvm::dbgs() << "\nChecking neighbor " << neighbor << " of "
                             << node << ".\n");
            if (visited.find(neighbor) == visited.end()) {
                LLVM_DEBUG(llvm::dbgs() << "\nContinue ...\n");
                findAndAddToken(neighbor, visited);
            } else {
                LLVM_DEBUG(
                    llvm::dbgs() << "\nFound backedge " << node << " <-> "
                                 << neighbor << "\n");
                for (auto &channel : graphChannels)
                    if (channel.srcNode == node
                        && channel.dstNode == neighbor) {
                        LLVM_DEBUG(
                            llvm::dbgs() << "\nFound the backedge channel!\n");
                        channel.initToken = 1;
                        LLVM_DEBUG(
                            llvm::dbgs()
                            << "\ninitToken = " << channel.initToken << "\n");
                    }
                return;
            }
        }
    }

    bool isConstant(Operation* op) { return (isa<arith::ConstantOp>(op)); }
    bool isArithNode(Operation* op)
    {
        return (
            isa<arith::AddIOp>(op) || isa<arith::SubIOp>(op)
            || isa<arith::MulIOp>(op) || isa<arith::CmpIOp>(op));
    }
    bool isMuxNode(Operation* op) { return (isa<scf::IfOp>(op)); }
    void addValueIntoMap(Value value, std::vector<MappingInfo> maps)
    {
        auto it = valueToPorts.find(value);
        if (it != valueToPorts.end()) {
            // Key exists, push the new port into the vector
            it->second.insert(it->second.end(), maps.begin(), maps.end());
        } else {
            // Key does not exist, create a new entry with the key and a vector
            // containing the port
            valueToPorts[value] = maps;
        }
    }

    std::vector<GraphNode> graphNodes;
    std::vector<GraphChannel> graphChannels;
    llvm::DenseMap<Value, std::vector<MappingInfo>> valueToPorts;
    SmallVector<Value> constants;
    llvm::DenseMap<Value, Value> yieldToOutput;
    int arithNodeIdx = 0;
    int muxNodeIdx = 0;
    SmallVector<Value> blkArgs, iterArgs;
    SmallVector<Value> debug_results;
    std::unordered_map<std::string, std::vector<std::string>> adjList;
};
} // namespace

void DfgPrintOperatorToYamlPass::getGraphNodes(OperatorOp operatorOp)
{
    std::vector<GraphNode> nodes;
    auto funcTy = operatorOp.getFunctionType();
    auto biasIterArg = funcTy.getNumInputs() + funcTy.getNumResults();
    blkArgs.append(
        operatorOp.getBody().getArguments().begin(),
        operatorOp.getBody().getArguments().begin() + biasIterArg);
    iterArgs.append(
        operatorOp.getBody().getArguments().begin() + biasIterArg,
        operatorOp.getBody().getArguments().end());

    // Each input will be a source node
    // If the input is used more than once, multiple output port will be created
    for (size_t i = 0; i < funcTy.getNumInputs(); i++) {
        auto nodeName = "arg" + std::to_string(i);
        auto blkArg = blkArgs[i];
        auto blkArgUses = blkArg.getUses();
        std::vector<GraphNodePort> ports;
        std::vector<MappingInfo> maps;
        for (auto j = 0;
             j < std::distance(blkArgUses.begin(), blkArgUses.end());
             j++) {
            auto portName = nodeName + '_' + std::to_string(j);
            GraphNodePort port = {portName, "out"};
            ports.push_back(port);
            maps.push_back({nodeName, port});
        }
        // The execution cycles for src nodes should be defined by hardware
        GraphNode node = {nodeName, "src", ports, 1};
        addValueIntoMap(blkArg, maps);
        nodes.push_back(node);
    }

    // Each operation will be handled
    auto idxBias = funcTy.getNumInputs();
    for (auto &op : operatorOp.getBody().getOps())
        getGraphNodesFromOp(&op, nodes, idxBias);

    // Each output will be a sink node
    for (size_t i = 0; i < funcTy.getNumResults(); i++) {
        auto idx = i + idxBias;
        auto nodeName = "arg" + std::to_string(idx);
        auto blkArg = blkArgs[idx];
        std::vector<GraphNodePort> ports;
        std::vector<MappingInfo> maps;
        // The execution cycles for src nodes should be defined by hardware
        GraphNodePort port = {nodeName, "in"};
        ports.push_back(port);
        maps.push_back({nodeName, port});
        GraphNode node = {nodeName, "sink", ports, 1};
        addValueIntoMap(yieldToOutput[blkArg], maps);
        nodes.push_back(node);
    }

    graphNodes = nodes;
}

void DfgPrintOperatorToYamlPass::getGraphNodesFromOp(
    Operation* op,
    std::vector<GraphNode> &nodes,
    size_t idxBias)
{
    if (isConstant(op)) {
        constants.push_back(op->getResult(0));
    } else if (isArithNode(op)) {
        auto nodeName = "arith_" + std::to_string(arithNodeIdx++) + '_'
                        + op->getName().getStringRef().str();
        std::vector<GraphNodePort> ports;

        auto lhs = op->getOperand(0);
        auto isLhsCst = isInSmallVector<Value>(lhs, constants);
        auto rhs = op->getOperand(1);
        auto isRhsCst = isInSmallVector<Value>(rhs, constants);
        assert(!(isLhsCst && isRhsCst) && "Please do constant folding first");
        GraphNodePort lhsPort, rhsPort;
        if (!isLhsCst) {
            lhsPort = {"lhs", "in"};
            addValueIntoMap(
                lhs,
                std::vector<MappingInfo>{
                    {nodeName, lhsPort}
            });
            ports.push_back(lhsPort);
        }
        if (!isRhsCst) {
            rhsPort = {"rhs", "in"};
            addValueIntoMap(
                rhs,
                std::vector<MappingInfo>{
                    {nodeName, rhsPort}
            });
            ports.push_back(rhsPort);
        }

        auto result = op->getResult(0);
        auto resultUses = result.getUses();
        std::vector<GraphNodePort> resultPorts;
        std::vector<MappingInfo> maps;
        for (auto j = 0;
             j < std::distance(resultUses.begin(), resultUses.end());
             j++) {
            auto portName = "result_" + std::to_string(j);
            GraphNodePort port = {portName, "out"};
            resultPorts.push_back(port);
            maps.push_back({nodeName, port});
        }
        addValueIntoMap(result, maps);
        ports.insert(ports.end(), resultPorts.begin(), resultPorts.end());
        GraphNode node = {nodeName, "arith", ports, 1};
        nodes.push_back(node);

        debug_results.push_back(result);
    } else if (isMuxNode(op)) {
        // TODO: deal with multiple results
        auto nodeName = "mux_" + std::to_string(muxNodeIdx++);
        std::vector<GraphNodePort> ports;
        auto select = op->getOperand(0);
        GraphNodePort selectPort = {"sel", "in"};
        addValueIntoMap(
            select,
            std::vector<MappingInfo>{
                {nodeName, selectPort}
        });
        ports.push_back(selectPort);

        int regionNum = 0;
        for (auto &region : op->getRegions())
            for (auto &op : region.getOps())
                if (isa<scf::YieldOp>(op)) {
                    auto yield = op.getOperand(0);
                    auto portName = "yield_" + std::to_string(regionNum++);
                    GraphNodePort yieldPort = {portName, "in"};
                    addValueIntoMap(
                        yield,
                        std::vector<MappingInfo>{
                            {nodeName, yieldPort}
                    });
                    ports.push_back(yieldPort);
                } else
                    getGraphNodesFromOp(&op, nodes, idxBias);

        auto result = op->getResult(0);
        auto resultUses = result.getUses();
        std::vector<GraphNodePort> resultPorts;
        std::vector<MappingInfo> maps;
        for (auto j = 0;
             j < std::distance(resultUses.begin(), resultUses.end());
             j++) {
            auto portName = "result_" + std::to_string(j);
            GraphNodePort port = {portName, "out"};
            resultPorts.push_back(port);
            maps.push_back({nodeName, port});
        }
        addValueIntoMap(result, maps);
        ports.insert(ports.end(), resultPorts.begin(), resultPorts.end());
        GraphNode node = {nodeName, "mux", ports, 1};
        nodes.push_back(node);

        debug_results.push_back(result);
    } else if (isa<scf::YieldOp>(op)) {
        return;
    } else if (isa<OutputOp>(op)) {
        for (size_t i = 0; i < op->getNumOperands(); i++)
            yieldToOutput[blkArgs[idxBias + i]] = op->getOperand(i);
    } else if (isa<YieldOp>(op)) {
        // Update the mapping info if there is value being iter args
        for (size_t i = 0; i < iterArgs.size(); i++) {
            auto it = valueToPorts.begin();
            while (it != valueToPorts.end()) {
                if (it->first == iterArgs[i]) {
                    auto currentPorts = std::move(it->second);
                    valueToPorts.erase(it);
                    addValueIntoMap(op->getOperand(i), currentPorts);
                    for (auto mapInfo : currentPorts)
                        LLVM_DEBUG(
                            llvm::dbgs() << "\nUpdating information of node "
                                         << mapInfo.nodeName << "'s port "
                                         << mapInfo.port.name << " of type "
                                         << mapInfo.port.type << ".\n");
                    LLVM_DEBUG(
                        llvm::dbgs()
                        << "  Now the value is " << op->getOperand(i)
                        << " instead of " << iterArgs[i] << ".\n");
                } else
                    ++it;
            }
        }
    }
}

void DfgPrintOperatorToYamlPass::getGraphChannels()
{
    LLVM_DEBUG(llvm::dbgs() << "\nHere is the mapping information:\n");
    for (auto map : valueToPorts) {
        // if (isInSmallVector(map.first, iterArgs))
        LLVM_DEBUG(
            llvm::dbgs()
            << "\nFor this value " << map.first << ", there is mappings:\n");
        for (auto mapInfo : map.second) {
            LLVM_DEBUG(
                llvm::dbgs() << "\n  Node " << mapInfo.nodeName << " and port "
                             << mapInfo.port.name << " of type "
                             << mapInfo.port.type << ".\n");
        }
    }
    std::vector<GraphChannel> channels;
    int channelNum = 0;
    for (auto map : valueToPorts) {
        std::vector<MappingInfo> producers, consumers;
        for (auto mapInfo : map.second)
            if (mapInfo.port.type == "in")
                consumers.push_back(mapInfo);
            else if (mapInfo.port.type == "out")
                producers.push_back(mapInfo);
        assert(producers.size() == consumers.size());
        for (size_t i = 0; i < producers.size(); i++) {
            channels.push_back(
                {"ch" + std::to_string(channelNum++),
                 producers[i].nodeName,
                 producers[i].port.name,
                 consumers[i].nodeName,
                 consumers[i].port.name,
                 0});
        }
    }
    graphChannels = channels;
}

void DfgPrintOperatorToYamlPass::runOnOperation()
{
    auto module = dyn_cast<ModuleOp>(getOperation());

    module.walk([&](OperatorOp operatorOp) {
        assert(
            !operatorOp.getBody().empty()
            && "Cannot print anything out of an empty operator.");
        for (auto &op : operatorOp.getBody().getOps())
            assert(
                (isa<arith::ConstantOp>(op) || isa<arith::AddIOp>(op)
                 || isa<arith::SubIOp>(op) || isa<arith::MulIOp>(op)
                 || isa<arith::CmpIOp>(op) || isa<scf::IfOp>(op)
                 || isa<scf::YieldOp>(op) || isa<YieldOp>(op)
                 || isa<OutputOp>(op))
                && "Unsupported ops in the region");

        auto graphName = operatorOp.getSymName().str();
        getGraphNodes(operatorOp);
        getGraphChannels();
        analyseChannelLoops();
        GraphYaml graph = {{graphName}, graphNodes, graphChannels};
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
