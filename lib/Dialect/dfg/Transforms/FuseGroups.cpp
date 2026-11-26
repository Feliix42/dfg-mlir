/// Implementation of the dfg-fuse-groups transform pass.
///
/// Fuses all `dfg.instantiate` ops carrying the same `!dfg.fusion<"name">`
/// annotation (in one parent region) into a single node: the group head's
/// process absorbs the other members' bodies (only the regions *inside* their
/// dfg.loop, if any — a dfg.loop is never nested into another), internal
/// channels become direct SSA values, and the loop-carried back edge becomes
/// the scf.for iter arg value. The head is determined by dataflow dominance
/// over the internal edge graph (deterministic tie-break: prefer a member with
/// an scf.for, then smallest callee name).
///
/// Illegal groups are skipped with a warning.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/FusionGroups.h"

#include "dfg-mlir/Dialect/dfg/IR/Attrs.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"

#include <functional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGFUSEGROUPS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {
struct DfgFuseGroupsPass
        : public dfg::impl::DfgFuseGroupsBase<DfgFuseGroupsPass> {
    void runOnOperation() override;
};
} // namespace

namespace {

/// Recursively remaps operands of `op` (including nested regions) via `vm`.
static void remapOperands(Operation *op, const DenseMap<Value, Value> &vm) {
    op->walk([&](Operation *inner) {
        for (OpOperand &o : inner->getOpOperands())
            if (auto it = vm.find(o.get()); it != vm.end())
                o.set(it->second);
    });
}

/// Per-group working state for fuse(): the head (process + instantiate) that
/// absorbs the other members, the planned port args, and the internal-channel
/// bookkeeping (chanMap etc.) accumulated across the fusion phases.
struct Fusion {
    FusionGroup &g;
    SmallVector<ProcessOp> &procs;
    unsigned headIdx;
    InstantiateOp headInst;
    ProcessOp headProc;
    Block *headBody;
    unsigned numHeadIn;
    LoopOp headLoop; // may be null
    Region *content; // region whose block is the inline target base
    Block *target;   // block to inline into (scf.for body if present)

    DenseSet<Value> internalPush; // push-side channel ends
    DenseSet<Value> internalPull; // pull-side channel ends
    DenseMap<Value, Value> chanMap; // push-side end -> replacement value

    // port plan
    SmallVector<Value> planInChans;  // pull-side ends, input ports
    SmallVector<Value> planOutChans; // push-side ends, output ports
    SmallVector<BlockArgument> newInArgs, newOutArgs;
    DenseMap<BlockArgument, Value> keptArgMap;   // old head arg -> new arg
    DenseMap<BlockArgument, Value> removedInArg; // old head in-arg -> pull-side end
    DenseMap<BlockArgument, Value> removedOutArg;

    Fusion(FusionGroup &g, SmallVector<ProcessOp> &procs, unsigned headIdx)
        : g(g), procs(procs), headIdx(headIdx) {}

    Value pushSideOf(Value pullSide) {
        return pullSide.getDefiningOp<ChannelOp>().getInChan();
    }

    /// The new port arg for an external channel end, or null.
    BlockArgument portArgFor(Value chanEnd, bool input) {
        auto &plan = input ? planInChans : planOutChans;
        auto &args = input ? newInArgs : newOutArgs;
        for (auto [i, c] : llvm::enumerate(plan))
            if (c == chanEnd)
                return args[i];
        return nullptr;
    }
};

/// Fuses one group. Returns failure (group left untouched as far as possible)
/// if anything unexpected is found.
static LogicalResult fuse(Fusion &f) {
    MLIRContext *ctx = f.headInst->getContext();
    IRRewriter rewriter(ctx);

    // ---- phase 1: port plan + new args -----------------------------------
    for (auto [inC, outC] : f.g.internalChans) {
        f.internalPush.insert(inC);
        f.internalPull.insert(outC);
    }
    auto planAddIn = [&](Value outC) {
        if (llvm::is_contained(f.planInChans, outC))
            return;
        f.planInChans.push_back(outC);
    };
    auto planAddOut = [&](Value inC) {
        if (llvm::is_contained(f.planOutChans, inC))
            return;
        f.planOutChans.push_back(inC);
    };
    for (Value inC : f.headInst.getInputs())
        if (!f.internalPull.count(inC))
            planAddIn(inC);
    for (Value outC : f.headInst.getOutputs())
        if (!f.internalPush.count(outC))
            planAddOut(outC);
    for (auto [i, inst] : llvm::enumerate(f.g.members)) {
        if ((unsigned)i == f.headIdx)
            continue;
        for (Value inC : inst.getInputs())
            if (!f.internalPull.count(inC))
                planAddIn(inC);
        for (Value outC : inst.getOutputs())
            if (!f.internalPush.count(outC))
                planAddOut(outC);
    }

    Location loc = f.headInst.getLoc();
    for (Value c : f.planInChans)
        f.newInArgs.push_back(f.headBody->addArgument(c.getType(), loc));
    for (Value c : f.planOutChans)
        f.newOutArgs.push_back(f.headBody->addArgument(c.getType(), loc));

    // map old head args
    for (auto [i, inC] : llvm::enumerate(f.headInst.getInputs())) {
        BlockArgument oldArg = f.headBody->getArgument(i);
        if (f.internalPull.count(inC))
            f.removedInArg[oldArg] = inC;
        else
            f.keptArgMap[oldArg] = f.portArgFor(inC, true);
    }
    for (auto [j, outC] : llvm::enumerate(f.headInst.getOutputs())) {
        BlockArgument oldArg = f.headBody->getArgument(f.numHeadIn + j);
        if (f.internalPush.count(outC))
            f.removedOutArg[oldArg] = outC;
        else
            f.keptArgMap[oldArg] = f.portArgFor(outC, false);
    }
    for (auto [oldArg, newArg] : f.keptArgMap)
        oldArg.replaceAllUsesWith(newArg);

    // ---- phase 2: inline member bodies ------------------------------------
    // topological order over internal edges (Kahn); back edges don't matter
    // for the flat inline since values are resolved via chanMap afterwards.
    unsigned n = f.g.members.size();
    SmallVector<unsigned> indeg(n, 0);
    for (auto [a, b] : f.g.edges)
        indeg[b]++;
    SmallVector<unsigned> order;
    {
        SmallVector<unsigned> stack;
        for (unsigned i = 0; i < n; i++)
            if (indeg[i] == 0)
                stack.push_back(i);
        // deterministic: smallest index first
        while (!stack.empty()) {
            llvm::sort(stack);
            unsigned u = stack.front();
            stack.erase(stack.begin());
            order.push_back(u);
            for (auto [a, b] : f.g.edges)
                if (a == u && --indeg[b] == 0)
                    stack.push_back(b);
        }
        if (order.size() != n) // cycle without roots: append remaining
            for (unsigned i = 0; i < n; i++)
                if (llvm::is_contained(order, i) == false)
                    order.push_back(i);
    }

    // inline point: inside the head's scf.for body if present
    if (f.headLoop) {
        for (Operation &o : f.target->getOperations())
            if (auto forOp = dyn_cast<scf::ForOp>(&o)) {
                f.target = forOp.getBody();
                break;
            }
    }

    // resolve the push-side channel end of a push's chan operand
    auto pushSideOfChanOpd = [&](Value chanOpd) -> Value {
        if (auto co = chanOpd.getDefiningOp<ChannelOp>())
            return co.getInChan();
        if (auto ba = dyn_cast<BlockArgument>(chanOpd)) {
            if (f.removedOutArg.count(ba))
                return f.removedOutArg[ba];
        }
        return nullptr;
    };
    // locate the (current) push op for each internal push-side channel end
    DenseMap<Value, Operation *> pushOf;
    auto scanPushes = [&](Block *b,
                          const DenseMap<BlockArgument, Value> &argChan) {
        b->walk([&](PushOp push) {
            Value chan = push.getChan();
            Value pushSide = nullptr;
            if (auto co = chan.getDefiningOp<ChannelOp>())
                pushSide = co.getInChan();
            else if (auto ba = dyn_cast<BlockArgument>(chan))
                pushSide = argChan.lookup(ba);
            if (pushSide && f.internalPush.count(pushSide))
                pushOf[pushSide] = push.getOperation();
        });
    };
    scanPushes(f.headBody, f.removedOutArg);
    for (unsigned mi = 0; mi < n; mi++) {
        if (mi == f.headIdx)
            continue;
        // map the member's port block args to their channel ends
        DenseMap<BlockArgument, Value> argChan;
        InstantiateOp instM0 = f.g.members[mi];
        Block *bodyM0 = procBodyBlock(f.procs[mi]);
        unsigned numInM0 = instM0.getInputs().size();
        for (auto [k, outC] : llvm::enumerate(instM0.getOutputs()))
            argChan[bodyM0->getArgument(numInM0 + k)] = outC;
        scanPushes(bodyM0, argChan);
    }

    for (unsigned mi : order) {
        if (mi == f.headIdx)
            continue;
        InstantiateOp instM = f.g.members[mi];
        ProcessOp procM = f.procs[mi];
        Block *bodyM = procBodyBlock(procM);
        unsigned numInM = instM.getInputs().size();
        LoopOp loopM = nullptr;
        for (Operation &o : bodyM->getOperations())
            if (auto l = dyn_cast<LoopOp>(&o))
                loopM = l;
        Region &contentM = loopM ? loopM.getBody() : procM.getBody();
        Block &src = contentM.front();

        DenseMap<Value, Value> vm;
        for (auto [k, inC] : llvm::enumerate(instM.getInputs())) {
            Value dst = f.internalPull.count(inC)
                            ? inC // placeholder channel end; pulls fixed later
                            : (Value)f.portArgFor(inC, true);
            vm[bodyM->getArgument(k)] = dst;
        }
        for (auto [k, outC] : llvm::enumerate(instM.getOutputs())) {
            Value dst = f.internalPush.count(outC)
                            ? outC // placeholder; pushes fixed later
                            : (Value)f.portArgFor(outC, false);
            vm[bodyM->getArgument(numInM + k)] = dst;
        }

        // Values used inside the member's loop region but defined in the
        // member's process body outside of it (e.g. loop bound constants):
        // clone them into the head's process body so they dominate the
        // inline site.
        std::function<Value(Value)> materialize = [&](Value v) -> Value {
            if (auto it = vm.find(v); it != vm.end())
                return it->second;
            Operation *def = v.getDefiningOp();
            if (!def)
                return nullptr; // block arg: covered by initial vm entries
            Region *defReg = def->getParentRegion();
            if (defReg == &contentM || contentM.isAncestor(defReg))
                return nullptr; // inside the content; cloned with the body
            if (defReg != &procM.getBody())
                return nullptr;
            for (OpOperand &o : def->getOpOperands()) {
                Value mv = materialize(o.get());
                if (!mv)
                    return nullptr;
                vm[o.get()] = mv;
            }
            if (f.headLoop)
                rewriter.setInsertionPoint(f.headLoop);
            else
                rewriter.setInsertionPointToEnd(f.headBody);
            Operation *cloned = rewriter.clone(*def);
            remapOperands(cloned, vm);
            for (auto [rOld, rNew] :
                 llvm::zip(def->getResults(), cloned->getResults()))
                vm[rOld] = rNew;
            return vm[v];
        };

        // choose the inline block: the deepest block containing the pushes of
        // all internal input channels (members consume where producers push)
        Block *inlineBlock = f.target;
        for (Value inC : instM.getInputs()) {
            if (!f.internalPull.count(inC))
                continue;
            Value pushSide = f.pushSideOf(inC);
            Operation *push = pushOf.lookup(pushSide);
            if (!push) {
                emitWarning(instM.getLoc(),
                            "dfg-fuse-groups: no producer found for internal "
                            "channel; skipping group");
                return failure();
            }
            Block *pb = push->getBlock();
            // producers not yet inlined (back edges through a cycle) live in
            // other processes; ignore them for inline-block selection — their
            // values are resolved via chanMap in later phases
            if (!f.headProc->isProperAncestor(pb->getParentOp()))
                continue;
            Region *pr = pb->getParent();
            Region *ir = inlineBlock->getParent();
            if (pr == ir)
                continue;
            if (pr->isAncestor(ir))
                continue; // pb is an outer block; keep the deeper inlineBlock
            if (ir->isAncestor(pr))
                inlineBlock = pb; // pb is deeper
            else {
                emitWarning(instM.getLoc(),
                            "dfg-fuse-groups: internal inputs produced in "
                            "disjoint regions; skipping group");
                return failure();
            }
        }
        Operation *insertBefore = inlineBlock->getTerminator();

        for (Operation &op : src.getOperations()) {
            op.walk([&](Operation *inner) {
                for (OpOperand &o : inner->getOpOperands())
                    materialize(o.get());
            });
            if (insertBefore)
                rewriter.setInsertionPoint(insertBefore);
            else
                rewriter.setInsertionPointToEnd(inlineBlock);
            Operation *cloned = rewriter.clone(op);
            remapOperands(cloned, vm);
            for (auto [rOld, rNew] : llvm::zip(op.getResults(), cloned->getResults()))
                vm[rOld] = rNew;
            cloned->walk([&](PushOp push) {
                Value pushSide = pushSideOfChanOpd(push.getChan());
                if (pushSide && f.internalPush.count(pushSide))
                    pushOf[pushSide] = push.getOperation();
            });
        }
    }

    // ---- phase 3: rewrite internal channel pushes/pulls -------------------
    // resolve a chan operand to (channel end, channel op); null if not a channel
    auto resolveChan = [&](Value chanOpd) -> std::pair<Value, ChannelOp> {
        if (auto co = chanOpd.getDefiningOp<ChannelOp>())
            return {chanOpd, co};
        if (auto ba = dyn_cast<BlockArgument>(chanOpd)) {
            if (f.removedInArg.count(ba))
                return {f.removedInArg[ba], nullptr};
            if (f.removedOutArg.count(ba))
                return {f.removedOutArg[ba], nullptr};
        }
        return {nullptr, nullptr};
    };
    // first pass: record values pushed to internal channels. "Internal" here
    // means region-level internal channels (between group members) as well as
    // channels created locally inside the (already inlined) head process.
    auto isInternalPushSide = [&](Value chanEnd, ChannelOp co) {
        if (f.internalPush.count(chanEnd))
            return true;
        return co && f.headProc->isProperAncestor(co.getOperation());
    };
    auto isInternalPullSide = [&](Value chanEnd, ChannelOp co) {
        if (f.internalPull.count(chanEnd))
            return true;
        return co && f.headProc->isProperAncestor(co.getOperation());
    };
    SmallVector<Operation *> toErase;
    f.headBody->walk([&](Operation *op) {
        auto push = dyn_cast<PushOp>(op);
        if (!push)
            return;
        auto [chanEnd, co] = resolveChan(push.getChan());
        if (!chanEnd)
            return;
        if (isInternalPushSide(chanEnd, co))
            f.chanMap[chanEnd] = push.getInp();
    });
    // second pass: rewrite pulls of internal channels, erase internal pushes
    f.headBody->walk([&](Operation *op) {
        if (auto push = dyn_cast<PushOp>(op)) {
            auto [chanEnd, co] = resolveChan(push.getChan());
            if (chanEnd && isInternalPushSide(chanEnd, co))
                toErase.push_back(push);
        } else if (auto pull = dyn_cast<PullOp>(op)) {
            auto [chanEnd, co] = resolveChan(pull.getChan());
            if (!chanEnd || !isInternalPullSide(chanEnd, co))
                return;
            Value pushSide =
                co ? co.getInChan() : f.pushSideOf(chanEnd);
            auto it = f.chanMap.find(pushSide);
            if (it == f.chanMap.end()) {
                emitWarning(pull.getLoc(),
                            "dfg-fuse-groups: no producer for internal channel; "
                            "skipping pull rewrite");
                return;
            }
            // keep chanMap consistent: entries pointing at this pull's result
            // must follow the replacement
            for (auto &kv : f.chanMap)
                if (kv.second == pull.getResult())
                    kv.second = it->second;
            pull.getResult().replaceAllUsesWith(it->second);
            toErase.push_back(pull);
        }
    });
    for (Operation *op : toErase)
        rewriter.eraseOp(op);

    // ---- phase 4: convert channel-typed scf.for iter args whose channels
    // became internal to value-typed iter args ------------------------------
    SmallVector<Operation *> forOps;
    f.headBody->walk([&](Operation *op) {
        if (isa<scf::ForOp>(op))
            forOps.push_back(op);
    });
    for (Operation *forOp : forOps) {
        auto forOpT = cast<scf::ForOp>(forOp);
        Block *forBody = forOpT.getBody();
        auto yieldOp = cast<scf::YieldOp>(forBody->getTerminator());
        for (unsigned k = 0; k + 1 < forBody->getNumArguments(); k++) {
            BlockArgument arg = forBody->getArgument(1 + k);
            auto outTy = dyn_cast<OutputType>(arg.getType());
            if (!outTy)
                continue; // already value-typed
            Value init = forOpT->getOperand(3 + k);
            Value yieldV = yieldOp->getOperand(k);
            // Materialize a pull of an external channel end: before the loop
            // for the init value, at the end of the body for the yielded
            // value (per-iteration, matching channel pull semantics).
            auto materializePull = [&](Value chanEnd, bool inForBody) -> Value {
                Value end = chanEnd;
                if (auto ba = dyn_cast<BlockArgument>(chanEnd))
                    if (auto it = f.keptArgMap.find(ba); it != f.keptArgMap.end())
                        end = it->second;
                if (inForBody)
                    rewriter.setInsertionPoint(yieldOp.getOperation());
                else
                    rewriter.setInsertionPoint(forOp);
                auto pull = rewriter.create<PullOp>(forOp->getLoc(),
                                                    outTy.getElementType(), end);
                return pull.getResult();
            };
            // Resolve a channel-typed iter-arg end to a value: internal
            // channel -> the pushed value (chanMap); external port/channel ->
            // a fresh pull; anything else (e.g. an outer for's already
            // converted arg) -> the value itself.
            auto resolveState = [&](Value v, bool isInit) -> Value {
                if (auto ba = dyn_cast<BlockArgument>(v)) {
                    if (f.removedInArg.count(ba))
                        return f.chanMap.lookup(f.pushSideOf(f.removedInArg[ba]));
                    // external input port (old or new arg form): pull it
                    if (f.keptArgMap.count(ba) ||
                        llvm::is_contained(f.newInArgs, ba))
                        return materializePull(v, !isInit);
                    return v;
                }
                if (auto co = v.getDefiningOp<ChannelOp>()) {
                    if (Value r = f.chanMap.lookup(co.getInChan()))
                        return r;
                    return materializePull(v, !isInit);
                }
                return v;
            };
            Value newInit = resolveState(init, true);
            Value newYield = resolveState(yieldV, false);
            if (newInit == init && newYield == yieldV)
                continue; // no channel ends involved
            Type elemTy = outTy.getElementType();
            // replace body pulls of the iter arg with the arg itself
            SmallVector<Operation *> pullErase;
            for (OpOperand &u : arg.getUses()) {
                if (auto pull = dyn_cast<PullOp>(u.getOwner());
                    pull && pull->getOperand(0) == (Value)arg) {
                    for (auto &kv : f.chanMap)
                        if (kv.second == pull.getResult())
                            kv.second = arg;
                    pull.getResult().replaceAllUsesWith(arg);
                    pullErase.push_back(pull);
                }
            }
            for (Operation *p : pullErase)
                rewriter.eraseOp(p);
            arg.setType(elemTy);
            forOpT->getResult(k).setType(elemTy);
            forOpT->setOperand(3 + k, newInit);
            yieldOp->setOperand(k, newYield);
            // after-loop pulls of the for result become the result itself
            SmallVector<Operation *> afterPulls;
            f.headBody->walk([&](Operation *op) {
                if (auto pull = dyn_cast<PullOp>(op);
                    pull && pull->getOperand(0) ==
                                 (Value)forOpT->getResult(k)) {
                    for (auto &kv : f.chanMap)
                        if (kv.second == pull.getResult())
                            kv.second = forOpT->getResult(k);
                    pull.getResult().replaceAllUsesWith(forOpT->getResult(k));
                    afterPulls.push_back(pull);
                }
            });
            for (Operation *p : afterPulls)
                rewriter.eraseOp(p);
        }
    }
    // erase channel ops local to the head process that became dead
    {
        SmallVector<Operation *> deadChans;
        f.headBody->walk([&](ChannelOp co) {
            if (co->use_empty())
                deadChans.push_back(co.getOperation());
        });
        for (Operation *co : deadChans)
            rewriter.eraseOp(co);
    }

    // ---- phase 5: fix remaining uses of removed args, rebuild ports --------
    for (auto [oldArg, pullSide] : f.removedInArg) {
        Value v = f.chanMap.lookup(f.pushSideOf(pullSide));
        if (v)
            oldArg.replaceAllUsesWith(v);
    }
    for (auto [oldArg, pushSide] : f.removedOutArg)
        oldArg.dropAllUses();

    // ---- phase 6: capture outside-region uses of scf.for bodies -----------
    // Inlined member content may reference values defined in an enclosing
    // block (e.g. the head's bound pulls). scf.for regions are isolated from
    // above, so rebuild each for op with such values as passthrough iter args.
    {
        SmallVector<scf::ForOp> forOps;
        f.headBody->walk([&](scf::ForOp op) { forOps.push_back(op); });
        for (scf::ForOp forOp : forOps) {
            Region &region = forOp->getRegion(0);
            SmallVector<Value> captured;
            DenseSet<Value> seen;
            region.walk([&](Operation *op) {
                for (OpOperand &o : op->getOpOperands()) {
                    Value v = o.get();
                    if (auto ba = dyn_cast<BlockArgument>(v)) {
                        if (region.isAncestor(ba.getOwner()->getParent()))
                            continue; // own block arg
                    } else if (Operation *def = v.getDefiningOp()) {
                        if (region.isAncestor(def->getParentRegion()))
                            continue; // defined inside the region
                    } else {
                        continue;
                    }
                    if (seen.insert(v).second)
                        captured.push_back(v);
                }
            });
            if (captured.empty())
                continue;
            rewriter.setInsertionPoint(forOp);
            SmallVector<Value> newInits(forOp.getInitArgs().begin(),
                                        forOp.getInitArgs().end());
            newInits.append(captured.begin(), captured.end());
            auto newFor = rewriter.create<scf::ForOp>(
                forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
                forOp.getStep(), newInits);
            if (auto a = forOp->getAttr("unsigned_cmp"))
                newFor->setAttr("unsigned_cmp", a);
            Block *oldBody = forOp.getBody();
            Block *newBody = newFor.getBody();
            // old body args map onto the prefix of the new body args
            for (unsigned i = 0, e = oldBody->getNumArguments(); i < e; ++i)
                oldBody->getArgument(i).replaceAllUsesWith(
                    newBody->getArgument(i));
            // move all ops (including the old yield) into the new body
            newBody->getOperations().splice(newBody->end(),
                                            oldBody->getOperations());
            Region &newRegion = newFor->getRegion(0);
            unsigned base = 1 + (unsigned)forOp.getInitArgs().size();
            for (auto [k, v] : llvm::enumerate(captured)) {
                BlockArgument newArg = newBody->getArgument(base + k);
                v.replaceUsesWithIf(newArg, [&](OpOperand &u) {
                    return newRegion.isAncestor(u.getOwner()->getParentRegion());
                });
            }
            // the yield must produce one value per iter arg
            auto yieldOp = cast<scf::YieldOp>(newBody->getTerminator());
            for (auto [k, v] : llvm::enumerate(captured))
                yieldOp->insertOperands(yieldOp->getNumOperands(),
                                        {newBody->getArgument(base + k)});
            // redirect old results to the new op's results (prefix matches)
            for (auto [rOld, rNew] :
                 llvm::zip(forOp.getResults(), newFor.getResults()))
                rOld.replaceAllUsesWith(rNew);
            rewriter.eraseOp(forOp);
        }
    }

    // dfg.loop operands = new port args (+ unchanged iter args)
    if (f.headLoop) {
        SmallVector<Value> loopOps;
        for (BlockArgument a : f.newInArgs)
            loopOps.push_back(a);
        for (BlockArgument a : f.newOutArgs)
            loopOps.push_back(a);
        unsigned numIter = f.headLoop.getIterArgs().size();
        for (Value v : f.headLoop.getIterArgs())
            loopOps.push_back(v);
        f.headLoop->setOperands(loopOps);
        f.headLoop->setAttr("operand_segment_sizes",
                            DenseI32ArrayAttr::get(
                                ctx, {(int)f.newInArgs.size(),
                                      (int)f.newOutArgs.size(), (int)numIter}));
    }
    // process function type
    {
        SmallVector<Type> inTys, outTys;
        for (BlockArgument a : f.newInArgs)
            inTys.push_back(a.getType());
        for (BlockArgument a : f.newOutArgs)
            outTys.push_back(a.getType());
        f.headProc->setAttr(f.headProc.getFunctionTypeAttrName(),
                            TypeAttr::get(FunctionType::get(ctx, inTys, outTys)));
    }
    // head instantiate operands
    {
        SmallVector<Value> ops;
        for (Value c : f.planInChans)
            ops.push_back(c);
        for (Value c : f.planOutChans)
            ops.push_back(c);
        f.headInst->setOperands(ops);
        f.headInst->setAttr("operand_segment_sizes",
                            DenseI32ArrayAttr::get(
                                ctx, {(int)f.planInChans.size(),
                                      (int)f.planOutChans.size()}));
    }
    // erase old args (descending), internal channels, member instantiates
    for (unsigned i = f.headBody->getNumArguments(); i-- > 0;) {
        BlockArgument arg = f.headBody->getArgument(i);
        // new args were appended after the old ones; only erase old ones
        if (llvm::is_contained(f.newInArgs, arg) ||
            llvm::is_contained(f.newOutArgs, arg))
            continue;
        arg.dropAllUses();
        f.headBody->eraseArgument(i);
    }
    for (auto [i, inst] : llvm::enumerate(f.g.members)) {
        if ((unsigned)i == f.headIdx)
            continue;
        rewriter.eraseOp(inst);
    }
    for (auto [inC, outC] : f.g.internalChans) {
        if (auto co = inC.getDefiningOp<ChannelOp>())
            if (co->use_empty())
                rewriter.eraseOp(co);
    }
    // erase member processes that became dead
    for (auto [i, proc] : llvm::enumerate(f.procs)) {
        if ((unsigned)i == f.headIdx)
            continue;
        if (!proc->use_empty())
            continue;
        // make sure no op inside the member process is still referenced from
        // the fused head (would indicate a wiring bug)
        bool leaked = false;
        proc->walk([&](Operation *o) {
            for (Value r : o->getResults())
                for (OpOperand &u : r.getUses())
                    if (!proc->isProperAncestor(u.getOwner()))
                        leaked = true;
        });
        if (leaked) {
            emitWarning(proc.getLoc(),
                        "dfg-fuse-groups: member process still referenced "
                        "from outside; leaving it in place");
            continue;
        }
        proc->walk([](Operation *o) { o->dropAllUses(); });
        rewriter.eraseOp(proc);
    }
    f.headInst->removeAttr("fusion");
    f.headProc->removeAttr("fusion");
    return success();
}

} // namespace

void DfgFuseGroupsPass::runOnOperation()
{
    Operation *op = getOperation();
    auto groups = collectFusionGroups(op);
    for (FusionGroup &g : groups) {
        if (g.members.size() < 2)
            continue; // singleton: no-op
        auto warn = [&](Operation *site, const Twine &msg) {
            emitWarning(site->getLoc(), msg);
        };
        if (failed(analyzeFusionGroup(g, warn)))
            continue; // warned; skip group
        SmallVector<ProcessOp> procs;
        for (auto inst : g.members)
            procs.push_back(getMemberProcess(inst));
        unsigned headIdx = computeHead(g, procs);

        InstantiateOp headInst = g.members[headIdx];
        ProcessOp headProc = procs[headIdx];
        // if the head process is shared with other instantiates, clone it
        ModuleOp module = headInst->getParentOfType<ModuleOp>();
        auto symUses = SymbolTable::getSymbolUses(headProc, module);
        if (symUses && !llvm::hasSingleElement(*symUses)) {
            MLIRContext *ctx = headInst->getContext();
            Operation *cloned = headProc->clone();
            std::string newName = headProc.getSymName().str() + "_fused";
            while (module.lookupSymbol(StringRef(newName)))
                newName += "_";
            cast<ProcessOp>(cloned).setSymNameAttr(StringAttr::get(ctx, newName));
            cloned->setLoc(headInst.getLoc());
            module.push_back(cloned);
            headProc = cast<ProcessOp>(cloned);
            headInst.setCalleeAttr(FlatSymbolRefAttr::get(ctx, newName));
            procs[headIdx] = headProc;
        }

        Fusion f(g, procs, headIdx);
        f.headInst = headInst;
        f.headProc = headProc;
        f.headBody = procBodyBlock(headProc);
        f.numHeadIn = headInst.getInputs().size();
        for (Operation &o : f.headBody->getOperations())
            if (auto l = dyn_cast<LoopOp>(&o))
                f.headLoop = l;
        f.content = f.headLoop ? &f.headLoop.getBody() : &headProc.getBody();
        f.target = &f.content->front();

        if (failed(fuse(f))) {
            emitWarning(headInst.getLoc(),
                        "dfg-fuse-groups: failed to fuse group '" + g.name +
                            "'");
            continue;
        }
    }
}

std::unique_ptr<Pass> mlir::dfg::createDfgFuseGroupsPass()
{
    return std::make_unique<DfgFuseGroupsPass>();
}
