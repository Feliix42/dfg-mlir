/// Shared analysis for fusion-group passes (dfg-fuse-groups,
/// dfg-verify-fusion-groups). See the header for an overview.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/FusionGroups.h"

#include <set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace dfg {

ProcessOp getMemberProcess(InstantiateOp inst) {
    auto module = inst->getParentOfType<ModuleOp>();
    return module.lookupSymbol<ProcessOp>(
        cast<FlatSymbolRefAttr>(inst.getCalleeAttr()).getValue());
}

Block *procBodyBlock(ProcessOp proc) {
    return &*proc.getBody().begin();
}

bool processHasLoop(ProcessOp proc) {
    return llvm::any_of(procBodyBlock(proc)->getOperations(),
                        [](Operation &o) { return isa<LoopOp>(&o); });
}

SmallVector<FusionGroup, 0> collectFusionGroups(Operation *root) {
    DenseMap<std::pair<Block *, StringRef>, size_t> index;
    SmallVector<FusionGroup, 0> result;
    root->walk([&](InstantiateOp inst) {
        StringRef group = getFusionGroup(inst);
        if (group.empty())
            return;
        auto key = std::make_pair(inst->getBlock(), group);
        auto it = index.find(key);
        if (it == index.end()) {
            FusionGroup g;
            g.name = group;
            g.parent = inst->getBlock();
            it = index.insert({key, result.size()}).first;
            result.push_back(std::move(g));
        }
        result[it->second].members.push_back(inst);
    });
    // deterministic order: by first member position
    llvm::sort(result, [](const FusionGroup &a, const FusionGroup &b) {
        return a.members.front()->isBeforeInBlock(b.members.front());
    });
    return result;
}

LogicalResult analyzeFusionGroup(
    FusionGroup &g,
    const function_ref<void(Operation *, const Twine &)> &diag) {
    if (g.members.empty())
        return success();

    // all members must instantiate a process and not be offloaded
    SmallVector<ProcessOp> procs;
    for (auto inst : g.members) {
        if (inst.getOffloaded()) {
            diag(inst, "cannot fuse group '" + g.name +
                               "': member is marked offloaded");
            return failure();
        }
        ProcessOp proc = getMemberProcess(inst);
        if (!proc) {
            diag(inst, "cannot fuse group '" + g.name +
                               "': callee is not a dfg.process");
            return failure();
        }
        procs.push_back(proc);
    }

    // all members must agree on dfg.loop presence (all loops or no loops)
    bool wantLoop = processHasLoop(procs.front());
    for (auto proc : procs) {
        if (processHasLoop(proc) != wantLoop) {
            diag(g.members.front(),
                 "cannot fuse group '" + g.name +
                     "': members disagree on dfg.loop presence "
                     "(partially looped groups are not supported)");
            return failure();
        }
    }

    // classify channels of the parent block
    // internal channel: push-side end used only as a group member's output,
    // pull-side end used only as a group member's input, exactly once each.
    auto memberOutChan = [&](Value v) -> int {
        for (auto [i, inst] : llvm::enumerate(g.members))
            for (Value o : inst.getOutputs())
                if (o == v)
                    return (int)i;
        return -1;
    };
    auto memberInChan = [&](Value v) -> int {
        for (auto [i, inst] : llvm::enumerate(g.members))
            for (Value in : inst.getInputs())
                if (in == v)
                    return (int)i;
        return -1;
    };

    bool ok = true;
    for (Operation &op : g.parent->getOperations()) {
        auto chanOp = dyn_cast<ChannelOp>(&op);
        if (!chanOp)
            continue;
        Value inC = chanOp.getInChan();   // push side
        Value outC = chanOp.getOutChan(); // pull side
        // count uses within the parent block
        auto usesOf = [&](Value v) {
            int memberUses = 0, otherUses = 0;
            for (OpOperand &u : v.getUses()) {
                if (u.getOwner()->getBlock() != g.parent)
                    continue;
                if (isa<InstantiateOp>(u.getOwner()) &&
                    llvm::is_contained(g.members,
                                       cast<InstantiateOp>(u.getOwner())))
                    memberUses++;
                else
                    otherUses++;
            }
            return std::make_pair(memberUses, otherUses);
        };
        auto [inMember, inOther] = usesOf(inC);
        auto [outMember, outOther] = usesOf(outC);
        if (inOther == 0 && outOther == 0 && inMember > 0 && outMember > 0) {
            // internal channel candidate; must be single-use on both ends
            if (inMember != 1 || outMember != 1) {
                diag(chanOp, "cannot fuse group '" + g.name +
                                 "': internal channel used more than once");
                ok = false;
                continue;
            }
            g.internalChans.push_back({inC, outC});
            g.edges.push_back({(unsigned)memberOutChan(inC),
                               (unsigned)memberInChan(outC)});
        }
    }
    if (!ok)
        return failure();

    // every internal channel must be produced and consumed inside member
    // process bodies (via pushes/pulls, or as scf.for iter-arg/yield ends of
    // loop-carried state channels)
    for (auto [inC, outC] : g.internalChans) {
        bool hasPush = false, hasPull = false;
        for (auto [i, inst] : llvm::enumerate(g.members)) {
            Block *body = procBodyBlock(procs[i]);
            unsigned numIn = inst.getInputs().size();
            DenseMap<BlockArgument, Value> argChan;
            for (auto [k, in] : llvm::enumerate(inst.getInputs()))
                argChan[body->getArgument(k)] = in;
            for (auto [k, out] : llvm::enumerate(inst.getOutputs()))
                argChan[body->getArgument(numIn + k)] = out;
            body->walk([&](Operation *o) {
                auto resolve = [&](Value v) -> Value {
                    if (auto ba = dyn_cast<BlockArgument>(v))
                        return argChan.lookup(ba);
                    return v;
                };
                if (auto push = dyn_cast<PushOp>(o)) {
                    if (resolve(push.getChan()) == inC)
                        hasPush = true;
                } else if (auto pull = dyn_cast<PullOp>(o)) {
                    if (resolve(pull.getChan()) == outC)
                        hasPull = true;
                } else if (auto forOp = dyn_cast<scf::ForOp>(o)) {
                    // loop-carried state: for init operands and yield values
                    for (unsigned i = forOp.getNumControlOperands();
                         i < forOp->getNumOperands(); i++)
                        if (resolve(forOp->getOperand(i)) == outC)
                            hasPull = true;
                    if (auto yieldOp = dyn_cast<scf::YieldOp>(
                            forOp.getBody()->getTerminator()))
                        for (Value y : yieldOp->getOperands())
                            if (resolve(y) == outC)
                                hasPull = true;
                }
            });
        }
        if (!hasPush || !hasPull) {
            diag(g.members.front(),
                 "cannot fuse group '" + g.name +
                     "': internal channel not produced/consumed inside the "
                     "group's processes");
            return failure();
        }
    }

    // connectivity: the internal-edge graph must be weakly connected
    // (a singleton group is a no-op and legal)
    if (g.members.size() > 1) {
        unsigned n = g.members.size();
        SmallVector<SmallVector<unsigned, 4>> adj(n);
        for (auto [a, b] : g.edges) {
            adj[a].push_back(b);
            adj[b].push_back(a);
        }
        SmallVector<bool> seen(n, false);
        SmallVector<unsigned> stack{0};
        seen[0] = true;
        unsigned seenCount = 1;
        while (!stack.empty()) {
            unsigned u = stack.pop_back_val();
            for (unsigned v : adj[u])
                if (!seen[v]) {
                    seen[v] = true;
                    seenCount++;
                    stack.push_back(v);
                }
        }
        if (seenCount != n) {
            diag(g.members.front(), "cannot fuse group '" + g.name +
                                       "': members are not connected");
            return failure();
        }
    }

    // external roots: members fed from outside the group
    for (auto [i, inst] : llvm::enumerate(g.members)) {
        bool ext = false;
        for (Value in : inst.getInputs()) {
            auto chanOp = in.getDefiningOp<ChannelOp>();
            if (!chanOp ||
                llvm::is_contained(
                    llvm::map_range(g.internalChans,
                                    [](auto &p) { return p.second; }),
                    in))
                continue;
            ext = true;
        }
        if (ext)
            g.externalRoots.push_back(i);
    }
    return success();
}

unsigned computeHead(FusionGroup &g, ArrayRef<ProcessOp> procs) {
    unsigned n = g.members.size();
    if (n == 1)
        return 0;
    // predecessors in the internal graph
    SmallVector<SmallVector<unsigned>> preds(n);
    for (auto [a, b] : g.edges)
        preds[b].push_back(a);
    // roots: members with external inputs, else members without internal
    // predecessors, else all members
    SmallVector<unsigned> roots = g.externalRoots;
    if (roots.empty()) {
        for (unsigned i = 0; i < n; i++)
            if (preds[i].empty())
                roots.push_back(i);
    }
    if (roots.empty())
        for (unsigned i = 0; i < n; i++)
            roots.push_back(i);

    // iterative dominator computation; roots act as entries (a virtual entry
    // dominates them, which contributes nothing to the member sets)
    SmallVector<std::set<unsigned>> dom(n);
    for (unsigned i = 0; i < n; i++) {
        dom[i] = {};
        for (unsigned j = 0; j < n; j++)
            dom[i].insert(j); // initial over-approximation: all members
    }
    for (unsigned r : roots)
        dom[r] = {r};
    bool changed = true;
    while (changed) {
        changed = false;
        for (unsigned i = 0; i < n; i++) {
            std::set<unsigned> nd;
            bool first = true;
            for (unsigned p : preds[i]) {
                if (first) {
                    nd = dom[p];
                    first = false;
                } else {
                    std::set<unsigned> tmp;
                    for (unsigned x : nd)
                        if (dom[p].count(x))
                            tmp.insert(x);
                    nd = tmp;
                }
            }
            if (first)
                nd = {};
            nd.insert(i);
            if (nd != dom[i]) {
                dom[i] = nd;
                changed = true;
            }
        }
    }
    // head candidates: members dominating all members
    SmallVector<unsigned> cands;
    for (unsigned h = 0; h < n; h++) {
        bool all = true;
        for (unsigned i = 0; i < n; i++)
            if (!dom[i].count(h))
                all = false;
        if (all)
            cands.push_back(h);
    }
    if (cands.empty())
        cands = roots;
    // with back edges, dominance degenerates (cycles make every member
    // "dominate" every other), so prefer members with external inputs — they
    // own the group's input ports and are the natural head
    {
        SmallVector<unsigned> extCands;
        for (unsigned c : cands)
            if (llvm::is_contained(g.externalRoots, c))
                extCands.push_back(c);
        if (!extCands.empty())
            cands = extCands;
    }
    // deterministic tie-break: prefer scf.for in process, then callee name
    auto hasFor = [&](unsigned i) {
        ProcessOp proc = procs[i];
        bool found = false;
        proc.getBody().walk([&](Operation *o) {
            if (isa<scf::ForOp>(o))
                found = true;
        });
        return found;
    };
    llvm::sort(cands, [&](unsigned a, unsigned b) {
        if (hasFor(a) != hasFor(b))
            return hasFor(a);
        return cast<FlatSymbolRefAttr>(g.members[a]->getAttr("callee"))
                   .getValue() <
               cast<FlatSymbolRefAttr>(g.members[b]->getAttr("callee"))
                   .getValue();
    });
    return cands.front();
}

} // namespace dfg
} // namespace mlir
