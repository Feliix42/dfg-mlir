llvm.func @malloc(i64) -> !llvm.ptr<i8>
llvm.func @free(!llvm.ptr<i8>) -> ()

// payload, pointer to next element
!bufferItem = !llvm.struct<"elem", (i64, ptr<struct<"elem">>)>

// first element, last element, capacity, occupancy, connected (bool)
!channelTy = !llvm.struct<(ptr<!bufferItem>, ptr<!bufferItem>, i64, i64, i1)>

llvm.func @make_channel() -> !llvm.ptr<!channelTy> {
    // useful constants
    %size = llvm.mlir.constant(1: index) : i64
    %0 = llvm.mlir.null : !llvm.ptr
    %nullptr = llvm.mlir.null : !llvm.ptr<!bufferItem>
    %true = llvm.mlir.constant(1 : i1) : i1
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

// func.func @static_alloc() -> memref<32x18xf32> {
// // CHECK: %[[num_elems:.*]] = llvm.mlir.constant(576 : index) : i64
// // CHECK: %[[null:.*]] = llvm.mlir.null : !llvm.ptr
// // CHECK: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[num_elems]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// // CHECK: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr to i64
// // CHECK: llvm.call @malloc(%[[size_bytes]]) : (i64) -> !llvm.ptr
//  %0 = memref.alloc() : memref<32x18xf32>
//  return %0 : memref<32x18xf32>
// }

llvm.func @send_data(%sender: !llvm.ptr<!channelTy>, %to_send: i64) {
// ^entry(%sender: !llvm.ptr<!channelTy>, %to_send: i64):
    // TODO:
    // check taint marker
    %sx = llvm.load %sender : !llvm.ptr<!channelTy>//  -> !channelTy
    %conn = llvm.extractvalue %sx[4] : !channelTy
    // on true, jump to next block, on false, jump to end
    llvm.cond_br %conn, ^insert(%sx: !channelTy), ^done
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

llvm.func @recv_data(%recv: !llvm.ptr<!channelTy>) -> i64 {
    // TODO
    %rx = llvm.load %recv : !llvm.ptr<!channelTy>
    %size = llvm.extractvalue %rx[3] : !channelTy
    %zero = llvm.mlir.constant(0) : i64
    %empty = llvm.icmp "eq" %size, %zero : i64
    llvm.cond_br %empty, ^emptyqueue(%rx: !channelTy), ^nonemptyqueue(%rx, %size: !channelTy, i64)
^emptyqueue(%rx1: !channelTy):
    // if size == 0:
    // channel open? if yes: block
    %chan_open = llvm.extractvalue %rx1[4] : !channelTy
    // TODO: abort on close

    %111111 = llvm.mlir.constant(0) : i64
    llvm.br ^done(%111111: i64)
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
    llvm.br ^done(%result: i64)
^done(%res: i64):
    llvm.return %res : i64
}

llvm.func @close(%chan: !llvm.ptr<!channelTy>) {
    // mark the channel as closed
    %0 = llvm.load %chan : !llvm.ptr<!channelTy>
    %1 = llvm.mlir.constant(0: i1) : i1
    %2 = llvm.insertvalue %1, %0[4] : !channelTy
    llvm.store %2, %chan : !channelTy, !llvm.ptr<!channelTy>
    llvm.return
}
