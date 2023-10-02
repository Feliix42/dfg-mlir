llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
llvm.func @malloc(i64) -> !llvm.ptr<i8>
llvm.func @free(!llvm.ptr<i8>) -> ()
llvm.mlir.global internal constant @str0("num recvd: %d\0A\00")

// TODO:
// - [x] Implement channel creation
// - [x] Implement channel closing
// - [x] Implement send
// - [x] Implement recv
// - [x] Signal channel closure via return
// - [x] Implement blocking in channels
// - [ ] test execution
// - [ ] Implement loop example
// - [ ] all lowerings covered?

// ============================================================================
// -- channel boilerplate
// ============================================================================

// Buffer item: i32
!bufferItem = i32
// buffer, head, tail, capacity, occupancy, connected (bool)
!channelTy = !llvm.struct<(!llvm.ptr<!bufferItem>, i64, i64, i64, i64, i8)>

llvm.func @make_channel() -> !llvm.ptr<!channelTy> {
    // useful constants
    %gep_hint = llvm.mlir.constant(1: index) : i64
    %buffer_hint = llvm.mlir.constant(1 : index) : !bufferItem
    %0 = llvm.mlir.null : !llvm.ptr
    %true = llvm.mlir.constant(1 : i8) : i8
    %cap = llvm.mlir.constant(64 : i64) : i64
    %zero_index = llvm.mlir.constant(0 : i64) : i64

    // allocate the channel construct on the heap
    %gep = llvm.getelementptr %0[%gep_hint] : (!llvm.ptr, i64) -> !llvm.ptr, !channelTy
    %size_bytes = llvm.ptrtoint %gep : !llvm.ptr to i64
    %raw_chan = llvm.call @malloc(%size_bytes) : (i64) -> !llvm.ptr<i8>
    %chan = llvm.bitcast %raw_chan : !llvm.ptr<i8> to !llvm.ptr<!channelTy>

    // allocate the buffer on the heap
    %gep_buffer = llvm.getelementptr %0[%buffer_hint] : (!llvm.ptr, !bufferItem) -> !llvm.ptr, !channelTy
    %buffer_bytes = llvm.ptrtoint %gep_buffer : !llvm.ptr to i64
    %raw_buf = llvm.call @malloc(%buffer_bytes) : (i64) -> !llvm.ptr<i8>
    %buffer = llvm.bitcast %raw_buf : !llvm.ptr<i8> to !llvm.ptr<!bufferItem>

    // create the channel type
    %1 = llvm.mlir.undef : !channelTy
    %2 = llvm.insertvalue %buffer, %1[0] : !channelTy
    %3 = llvm.insertvalue %zero_index, %2[1] : !channelTy
    %4 = llvm.insertvalue %zero_index, %3[2] : !channelTy
    %5 = llvm.insertvalue %cap, %4[3] : !channelTy
    %6 = llvm.insertvalue %zero_index, %5[4] : !channelTy
    %7 = llvm.insertvalue %true, %6[5] : !channelTy
    llvm.store %7, %chan : !channelTy, !llvm.ptr<!channelTy>

    llvm.return %chan : !llvm.ptr<!channelTy>
}

llvm.func @send_data(%sender: !llvm.ptr<!channelTy>, %to_send: !bufferItem) -> i1 {
    llvm.br ^validitycheck
^validitycheck:
    // check taint marker first to check channel availability
    %taint_ptr = llvm.getelementptr inbounds %sender[0, 5] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<i8>
    %conn = llvm.load %taint_ptr: !llvm.ptr<i8>
    %zero0 =llvm.mlir.constant(0: i1) : i8
    %conn1 = llvm.icmp "ne" %conn, %zero0 : i8
    %failed = llvm.mlir.constant(0: i1) : i1
    // on true, jump to next block, on false, jump to end
    llvm.cond_br %conn1, ^sizecheck, ^done(%failed : i1)
^sizecheck:
    %size_ptr = llvm.getelementptr inbounds %sender[0, 4] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<i64>
    %cap_ptr = llvm.getelementptr inbounds %sender[0, 3] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<i64>
    %initial_size = llvm.load %size_ptr : !llvm.ptr<i64>
    %cap = llvm.load %cap_ptr : !llvm.ptr<i64>

    // if size == capacity block until a spot becomes available
    %space_available = llvm.icmp "ult" %initial_size, %cap : i64
    llvm.cond_br %space_available, ^insert(%size_ptr, %cap: !llvm.ptr<i64>, i64), ^wait
^insert(%sizep: !llvm.ptr<i64>, %capacity: i64):
    // buf[tail] = item
    %tail_ptr = llvm.getelementptr inbounds %sender[0, 2] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<i64>
    %tail = llvm.load %tail_ptr : !llvm.ptr<i64>
    %innerptr = llvm.getelementptr inbounds %sender[0, 0] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<!llvm.ptr<!bufferItem>>
    %buffer = llvm.load %innerptr : !llvm.ptr<!llvm.ptr<!bufferItem>>
    %insertion_point = llvm.getelementptr %buffer[%tail] : (!llvm.ptr<!bufferItem>, i64) -> !llvm.ptr<!bufferItem>
    // the actual store op
    llvm.store %to_send, %insertion_point : !bufferItem, !llvm.ptr<!bufferItem>
    // tail = tail + 1 % cap
    %one = llvm.mlir.constant(1: i64) : i64
    %next_tail = llvm.add %tail, %one : i64
    %adjusted_tail = llvm.urem %next_tail, %capacity : i64
    llvm.store %adjusted_tail, %tail_ptr : i64, !llvm.ptr<i64>

    // size +1 (atomicrmw!)
    llvm.atomicrmw add %sizep, %one monotonic : !llvm.ptr<i64>, i64

    %success = llvm.mlir.constant(1 : i1) : i1
    llvm.br ^done(%success: i1)
^wait:
    // yield in favor of other tasks while the queue is blocked
    omp.taskyield
    llvm.br ^validitycheck
^done(%valid : i1):
    llvm.return %valid : i1
}

llvm.func @recv_data(%recv: !llvm.ptr<!channelTy>) -> !llvm.struct<(!bufferItem, i1)> {
    llvm.br ^sizecheck
^sizecheck:
    %size_ptr = llvm.getelementptr inbounds %recv[0, 4] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<i64>
    %initial_size = llvm.load %size_ptr : !llvm.ptr<i64>
    %zero = llvm.mlir.constant(0) : i64
    %empty = llvm.icmp "eq" %initial_size, %zero : i64
    // if size == 0 check if channel open & block until an item becomes available
    llvm.cond_br %empty, ^emptyqueue, ^pop_element(%size_ptr: !llvm.ptr<i64>)
    // llvm.cond_br %space_available, ^insert(%size_ptr, %cap: !llvm.ptr<i64>, i64), ^wait
^emptyqueue:
    // check taint marker first to check channel availability
    %taint_ptr = llvm.getelementptr inbounds %recv[0, 5] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<i8>
    %conn = llvm.load %taint_ptr: !llvm.ptr<i8>
    %zero0 =llvm.mlir.constant(0: i1) : i8
    %conn1 = llvm.icmp "ne" %conn, %zero0 : i8
    // on true, jump to next block, on false, jump to end
    llvm.cond_br %conn1, ^wait, ^closed
^wait:
    // yield in favor of other tasks while the queue is blocked
    omp.taskyield
    llvm.br ^sizecheck
^closed:
    %poison = llvm.mlir.poison : i32
    %valid = llvm.mlir.constant(0: i1) : i1
    llvm.br ^done(%poison, %valid: i32, i1)
^pop_element(%sizep: !llvm.ptr<i64>):
    %cap_ptr = llvm.getelementptr inbounds %recv[0, 3] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<i64>
    %cap = llvm.load %cap_ptr : !llvm.ptr<i64>
    // item = buf[head]
    %head_ptr = llvm.getelementptr inbounds %recv[0, 1] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<i64>
    %head = llvm.load %head_ptr : !llvm.ptr<i64>
    %innerptr = llvm.getelementptr inbounds %recv[0, 0] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<!llvm.ptr<!bufferItem>>
    %buffer = llvm.load %innerptr : !llvm.ptr<!llvm.ptr<!bufferItem>>
    %item_ptr = llvm.getelementptr %buffer[%head] : (!llvm.ptr<!bufferItem>, i64) -> !llvm.ptr<!bufferItem>
    // the actual loading op
    %item = llvm.load %item_ptr : !llvm.ptr<!bufferItem>
    // NOTE(feliix42): deletion not necessary?
    // head = head + 1 % cap
    %one = llvm.mlir.constant(1: i64) : i64
    %next_head = llvm.add %head, %one : i64
    %adjusted_head = llvm.urem %next_head, %cap : i64
    llvm.store %adjusted_head, %head_ptr : i64, !llvm.ptr<i64>

    // size +1 (atomicrmw!)
    llvm.atomicrmw sub %sizep, %one monotonic : !llvm.ptr<i64>, i64

    %success = llvm.mlir.constant(1 : i1) : i1
    llvm.br ^done(%item, %success: !bufferItem, i1)
^done(%res: !bufferItem, %validsig: i1):
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i1)>
    %1 = llvm.insertvalue %res, %0[0] : !llvm.struct<(i32, i1)>
    %2 = llvm.insertvalue %validsig, %1[1] : !llvm.struct<(i32, i1)>
    llvm.return %2 : !llvm.struct<(i32, i1)>
}

llvm.func @close(%chan: !llvm.ptr<!channelTy>) {
    %dead = llvm.mlir.constant(0: i8) : i8
    %live = llvm.mlir.constant(1: i8) : i8
    // This tries to mark the channel as closed. If this is already the case, deallocate the channel
    // 1. gep for the closed pointer
    // 2. cmpxchg 1 -> 0
    //     -> if that fails (i.e., the channel is closed on the other side), deallocate the channel and all items buffered
    %livenessptr = llvm.getelementptr inbounds %chan[0, 4] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<i8>
    %changed = llvm.cmpxchg %livenessptr, %live, %dead acq_rel monotonic: !llvm.ptr<i8>, i8
    %success = llvm.extractvalue %changed[1] : !llvm.struct<(i8, i1)>
    llvm.cond_br %success, ^done, ^deallocate
^deallocate:
    // deallocate the inner buffer first, then the outer pointer
    %innerptr = llvm.getelementptr inbounds %chan[0, 0] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<!llvm.ptr<!bufferItem>>
    %buffer = llvm.load %innerptr : !llvm.ptr<!llvm.ptr<!bufferItem>>
    %casted_buf = llvm.bitcast %buffer : !llvm.ptr<!bufferItem> to !llvm.ptr<i8>
    llvm.call @free(%casted_buf) : (!llvm.ptr<i8>) -> ()
    // dealloc channel
    %casted = llvm.bitcast %chan : !llvm.ptr<!channelTy> to !llvm.ptr<i8>
    llvm.call @free(%casted) : (!llvm.ptr<i8>) -> ()
    llvm.br ^done
^done:
    llvm.return
}


// ============================================================================
// -- actual "lowering"
// ============================================================================

func.func @produce_value(%val: !llvm.ptr<!channelTy>) {
    %val1 = arith.constant 323729 : i32

    %res = llvm.call @send_data(%val, %val1) : (!llvm.ptr<!channelTy>, i32) -> i1
    // NOTE(feliix42): If this were a loop, check the i1 result value for channel closure and branch off to the exit

    llvm.call @close(%val) : (!llvm.ptr<!channelTy>) -> ()
    return
}

func.func @consume(%number: !llvm.ptr<!channelTy>) {
    %recvd = llvm.call @recv_data(%number) : (!llvm.ptr<!channelTy>) -> !llvm.struct<(i32, i1)>
    %valid = llvm.extractvalue %recvd[1] : !llvm.struct<(i32, i1)>
    %value = llvm.extractvalue %recvd[0] : !llvm.struct<(i32, i1)>
    // NOTE(feliix42): If we receive multiple values, we unpack them all, AND combine all validity values and put them into the conditional branch
    llvm.cond_br %valid, ^inner(%value: i32), ^dead

^inner(%val: i32):
    %5 = llvm.mlir.addressof @str0 : !llvm.ptr<array<15 x i8>>
    %4 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.getelementptr %5[%4, %4] : (!llvm.ptr<array<15 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %21 = llvm.call @printf(%6, %val) : (!llvm.ptr<i8>, i32) -> i32
    // NOTE(feliix42): if this were a loop, we'd instread jump back to the receive block
    llvm.br ^dead
^dead:
    // TODO(feliix42): normally, we could have a loop that is wrapped around the block above. then, termination would drop us here, where cleanup needs to happen in any case:
    // 1. mark channels as closed. If closed already -> deallocate
    llvm.call @close(%number) : (!llvm.ptr<!channelTy>) -> ()
    return
}

func.func @algo() {
    %op_chan = llvm.call @make_channel() : () -> !llvm.ptr<!channelTy>

    %num_threads = arith.constant 4 : index
    omp.parallel num_threads(%num_threads : index) {
        func.call @produce_value(%op_chan) : (!llvm.ptr<!channelTy>) -> ()
        func.call @consume(%op_chan) : (!llvm.ptr<!channelTy>) -> ()
        omp.terminator
    }

    return
}