llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
llvm.func @malloc(i64) -> !llvm.ptr<i8>
llvm.func @free(!llvm.ptr<i8>) -> ()
llvm.mlir.global internal constant @str0("num recvd: %d\0A\00")

// ============================================================================
// -- channel boilerplate
// ============================================================================

// payload, pointer to next element
!bufferItem = !llvm.struct<"elem", (i32, ptr<struct<"elem">>)>

// first element, last element, capacity, occupancy, connected (bool)
!channelTy = !llvm.struct<(ptr<!bufferItem>, ptr<!bufferItem>, i64, i64, i8)>

llvm.func @make_channel() -> !llvm.ptr<!channelTy> {
    // useful constants
    %size = llvm.mlir.constant(1: index) : i64
    %0 = llvm.mlir.null : !llvm.ptr
    %nullptr = llvm.mlir.null : !llvm.ptr<!bufferItem>
    %true = llvm.mlir.constant(1 : i8) : i8
    %cap = llvm.mlir.constant(-1 : i64) : i64
    %used = llvm.mlir.constant(0 : i64) : i64

    // allocate the channel construct on the heap
    %gep = llvm.getelementptr %0[%size] : (!llvm.ptr, i64) -> !llvm.ptr, !channelTy
    %size_bytes = llvm.ptrtoint %gep : !llvm.ptr to i64
    %raw_chan = llvm.call @malloc(%size_bytes) : (i64) -> !llvm.ptr<i8>
    %chan = llvm.bitcast %raw_chan : !llvm.ptr<i8> to !llvm.ptr<!channelTy>

    // create the channel type
    %1 = llvm.mlir.undef : !channelTy
    %2 = llvm.insertvalue %nullptr, %1[0] : !channelTy
    %3 = llvm.insertvalue %nullptr, %2[1] : !channelTy
    %4 = llvm.insertvalue %cap, %3[2] : !channelTy
    %5 = llvm.insertvalue %used, %4[3] : !channelTy
    %6 = llvm.insertvalue %true, %5[4] : !channelTy
    llvm.store %6, %chan : !channelTy, !llvm.ptr<!channelTy>

    llvm.return %chan : !llvm.ptr<!channelTy>
}

llvm.func @send_data(%sender: !llvm.ptr<!channelTy>, %to_send: i32) {
// ^entry(%sender: !llvm.ptr<!channelTy>, %to_send: i64):
    // TODO:
    // check taint marker
    %sx = llvm.load %sender : !llvm.ptr<!channelTy>//  -> !channelTy
    %conn = llvm.extractvalue %sx[4] : !channelTy
    %zero0 =llvm.mlir.constant(0: i1) : i8
    %conn1 = llvm.icmp "ne" %conn, %zero0 : i8
    // on true, jump to next block, on false, jump to end
    llvm.cond_br %conn1, ^insert(%sx: !channelTy), ^done
^insert(%chan: !channelTy):
    // create item
    // useful constants
    %size = llvm.mlir.constant(1: index) : i64
    %0 = llvm.mlir.null : !llvm.ptr
    %nullptr = llvm.mlir.null : !llvm.ptr<!bufferItem>

    // allocate the new buffer item
    %gep = llvm.getelementptr %0[%size] : (!llvm.ptr, i64) -> !llvm.ptr, !bufferItem
    %size_bytes = llvm.ptrtoint %gep : !llvm.ptr to i64
    %raw_buffer= llvm.call @malloc(%size_bytes) : (i64) -> !llvm.ptr<i8>
    %buf = llvm.bitcast %raw_buffer : !llvm.ptr<i8> to !llvm.ptr<!bufferItem>

    // store input
    %1 = llvm.mlir.undef : !bufferItem
    %2 = llvm.insertvalue %to_send, %1[0] : !bufferItem
    %3 = llvm.insertvalue %nullptr, %1[1] : !bufferItem
    llvm.store %3, %buf : !bufferItem, !llvm.ptr<!bufferItem>

    // adjust pointers to next elements as necessary
    %occupancy = llvm.extractvalue %chan[3] : !channelTy
    %zero = llvm.mlir.constant(0) : i64
    %not_empty = llvm.icmp "ne" %occupancy, %zero : i64
    llvm.cond_br %not_empty, ^nonempty(%chan, %buf: !channelTy, !llvm.ptr<!bufferItem>), ^empty(%chan, %buf: !channelTy, !llvm.ptr<!bufferItem>)
^empty(%chan2: !channelTy, %buf2: !llvm.ptr<!bufferItem>):
    // if size is 0:
    // -> set start + end
    %chan21 = llvm.insertvalue %buf2, %chan2[0] : !channelTy
    %chan22 = llvm.insertvalue %buf2, %chan21[1] : !channelTy
    llvm.br ^setsize(%chan22: !channelTy)
^nonempty(%chan1: !channelTy, %buf1: !llvm.ptr<!bufferItem>):
    // else:
    // -> set end->next
    // -> set end
    %cur_next = llvm.extractvalue %chan[1] : !channelTy
    %cur_next_val = llvm.load %cur_next : !llvm.ptr<!bufferItem>
    %updated_cur = llvm.insertvalue %buf1, %cur_next_val[1] : !bufferItem
    llvm.store %updated_cur, %cur_next : !bufferItem, !llvm.ptr<!bufferItem>
    %chan11 = llvm.insertvalue %buf1, %chan1[1]: !channelTy
    llvm.br ^setsize(%chan11: !channelTy)
^setsize(%chan3: !channelTy):
    // size += 1
    %cur_size = llvm.extractvalue %chan3[3] : !channelTy
    %one = llvm.mlir.constant(1) : i64
    %new_size = llvm.add %cur_size, %one : i64
    %chan31 = llvm.insertvalue %new_size, %chan3[3] : !channelTy
    llvm.store %chan31, %sender : !channelTy, !llvm.ptr<!channelTy>
    llvm.br ^done
^done:
    llvm.return
}

llvm.func @recv_data(%recv: !llvm.ptr<!channelTy>) -> !llvm.struct<(i32, i1)> {
    // TODO:
    // 1. use gep for size
    // 2. use gep for open chan
    // 3. use gep for getting/setting start
    // 4. use atomicrmw sub for size adjustment
    // 5. check soundness -> size = 1??
    %rx = llvm.load %recv : !llvm.ptr<!channelTy>
    %size = llvm.extractvalue %rx[3] : !channelTy
    %zero = llvm.mlir.constant(0) : i64
    %empty = llvm.icmp "eq" %size, %zero : i64
    llvm.cond_br %empty, ^emptyqueue(%rx: !channelTy), ^nonemptyqueue(%rx, %size: !channelTy, i64)
^emptyqueue(%rx1: !channelTy):
    // if size == 0:
    // channel open? if yes: block
    %chan_open = llvm.extractvalue %rx1[4] : !channelTy
    %zero32 = llvm.mlir.constant(0: i8) : i8
    %should_block = llvm.icmp "ne" %chan_open, %zero32 : i8
    llvm.cond_br %should_block, ^block, ^closed
^block:
    // TODO(feliix42): block on empty queue -> omp.taskyield? => jump back to entry block
    //                 -> maybe use gep to have a ptr to the size variable
    llvm.br ^closed
^closed:
    %poison = llvm.mlir.poison : i32
    %valid = llvm.mlir.constant(0: i1) : i1
    llvm.br ^done(%poison, %valid: i32, i1)
^nonemptyqueue(%rx2: !channelTy, %old_size: i64):
    // read start
    %startp = llvm.extractvalue %rx2[0] : !channelTy
    %start = llvm.load %startp : !llvm.ptr<!bufferItem>
    %next_startptr = llvm.extractvalue %start[1] : !bufferItem
    // set start = start->next
    %rx21 = llvm.insertvalue %next_startptr, %rx2[0] : !channelTy
    // size -=1
    %one = llvm.mlir.constant(1) : i64
    %newsize = llvm.sub %old_size, %one : i64
    // if cur == end: end = null
    %null = llvm.mlir.null : !llvm.ptr<!bufferItem>
    // get pointer to `end` ptr
    %endptr = llvm.getelementptr inbounds %recv[0, 1] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<!llvm.ptr<!bufferItem>>
    llvm.cmpxchg %endptr, %next_startptr, %null acq_rel monotonic: !llvm.ptr<!llvm.ptr<!bufferItem>>, !llvm.ptr<!bufferItem>
    // dealloc item, return contained value
    %result = llvm.extractvalue %start[0] : !bufferItem
    %buf = llvm.bitcast %startp : !llvm.ptr<!bufferItem> to !llvm.ptr<i8>
    llvm.call @free(%buf) : (!llvm.ptr<i8>) -> ()
    %success = llvm.mlir.constant(1: i1) : i1
    llvm.br ^done(%result, %success: i32, i1)
^done(%res: i32, %validsig: i1):
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
    %innerptr = llvm.getelementptr inbounds %chan[0, 0] : (!llvm.ptr<!channelTy>) -> !llvm.ptr<!llvm.ptr<!bufferItem>>
    %head = llvm.load %innerptr : !llvm.ptr<!llvm.ptr<!bufferItem>>
    // dealloc channel
    %casted = llvm.bitcast %chan : !llvm.ptr<!channelTy> to !llvm.ptr<i8>
    llvm.call @free(%casted) : (!llvm.ptr<i8>) -> ()
    
    llvm.br ^loop(%head: !llvm.ptr<!bufferItem>)
^loop(%it: !llvm.ptr<!bufferItem>):
    // loop to iteratively deallocate the whole linked list of buffer items until we hit a null pointer
    %null = llvm.mlir.null : !llvm.ptr<!bufferItem>
    %empty = llvm.icmp "eq" %it, %null : !llvm.ptr<!bufferItem>
    llvm.cond_br %empty, ^done, ^body(%it: !llvm.ptr<!bufferItem>)
^body(%item: !llvm.ptr<!bufferItem>):
    %nextptr = llvm.getelementptr inbounds %item[0, 1] : (!llvm.ptr<!bufferItem>) -> !llvm.ptr<!llvm.ptr<!bufferItem>>
    %next = llvm.load %nextptr : !llvm.ptr<!llvm.ptr<!bufferItem>>
    %freeable = llvm.bitcast %item : !llvm.ptr<!bufferItem> to !llvm.ptr<i8>
    llvm.call @free(%freeable) : (!llvm.ptr<i8>) -> ()
    llvm.br ^loop(%next : !llvm.ptr<!bufferItem>)
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