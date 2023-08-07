//
// KFAF: I added some actual LLVM types for what I could pull from source.
//

!VectorBase = !llvm.struct<"VectorBase", (i64)>
!IVector = !llvm.struct<"IVector", (!VectorBase, ptr, ptr)>

!GpsPosition = !llvm.struct<"GpsPosition", (f64, f64, i32, i32)>
!GpsRectangle = !llvm.struct<"GpsRectangle", (f64, f64, f64, f64)>
!GpsVector = !llvm.struct<"GpsVector", (!IVector, array<10x!GpsPosition>)>

!RoadSegment = !llvm.struct<"RoadSegment", (i32, i32, i32)>
!RoadPath = !llvm.struct<"RoadPath", (!IVector, array<5x!RoadSegment>)>

!CandiItem = !llvm.struct<"CandiItem", (f64, f64, i32, i32, f64, i32, i32, i32, i32, i32, i32, i32)>
!CandiList = !llvm.struct<"CandiList", (!IVector, array<5x!CandiItem>)>
!CandiVector = !llvm.struct<"CandiVector", (!IVector, array<10x!CandiList>)>

!RoadSpeed = !llvm.struct<"RoadSpeed", (i32, i32, i32, i32, i32, f64, f64, f64)>
!RoadSpeedVector = !llvm.struct<"RoadSpeedVector", (!IVector, array<30x!RoadSpeed>)>

!TrellisItem = !llvm.struct<"TrellisItem", (i32, f64, f64, i32, i1)>
!Trellis = !llvm.struct<"Trellis", (i32, array<50x!TrellisItem>)>

!Vertex = !llvm.struct<"Vertex", (i32, i64, f64, f64, array<16 x i32>)>
!Edge = !llvm.struct<"Edge", (i32, i64, i32, i32, i32, i32, i32, i32)>
!Geom = !llvm.struct<"Geom", (i32, i32, f64, f64, f64, f64, i32, i32)>
!MapCell = !llvm.struct<"MapCell", (!IVector, array<2048x!Vertex>, !IVector, array<1024x!Edge>, !IVector, array<9128x!Geom>)>

// KFAF: WTF are these?

!FCDPROC = i128 //i5600000323
!VecMapCell = i128 //i1513472000 // FS: I assumed 1000 elements.
!Dijkstra = i128 //i1513536

// placeholders
func.func private @read_map() -> !VecMapCell
func.func private @vclone(%mapcells_0_0_2_0: !VecMapCell) -> !VecMapCell
func.func private @get_next_vector(%fcdproc_0_0_1_0: !FCDPROC) -> !GpsVector
func.func private @inc(%ctrl: i32, %var_1: i8) -> ()
func.func private @convenient_placeholder(%size: i32) -> tuple<i1, i32>
func.func private @write_vector(%var_0: !RoadSpeedVector) -> i8
func.func private @cclone(%var_0: !CandiVector) -> !CandiVector
func.func private @gclone(%var_0: !GpsVector) -> !GpsVector
func.func private @clone(%value: !MapCell) -> !MapCell
func.func private @Dijkstra(%var_0: !MapCell) -> !Dijkstra
func.func private @vecat(%var_0: !VecMapCell, %var_1: i64) -> !MapCell
func.func private @findcell(%var_0: !VecMapCell, %var_1: !GpsPosition) -> i64
func.func private @gpsat(%var_0: !GpsVector, %idx: index) -> !GpsPosition
func.func private @Magic() -> i32


//
// BEGIN OHUA OUTPUT
//

dfg.operator @get_map outputs(%mapcells_0_0_2_tx: !VecMapCell) {
  %mapcells_0_0_2 = func.call @read_map() : () -> !VecMapCell
  dfg.push(%mapcells_0_0_2) %mapcells_0_0_2_tx : !VecMapCell
}


dfg.operator @mapcells_ctrl inputs(%mapcells_0_0_2_rx: !VecMapCell, %ctrl_0_6_rx: tuple<i1, i32>)
                            outputs(%mapcells0_0_0_0_tx: !VecMapCell) {
  dfg.loop inputs(%mapcells_0_0_2_rx: !VecMapCell, %ctrl_0_6_rx: tuple<i1, i32>) outputs(%mapcells0_0_0_0_tx: !VecMapCell) {
    %init_renew = arith.constant 0 : i1
    %mapcells_0_0_2_0 = dfg.pull %mapcells_0_0_2_rx : !VecMapCell

    scf.while (%renew = %init_renew) : (i1) -> () {
      // Forward the argument (as result or "after" region argument).
      %true = arith.constant 1 : i1
      %renew_cond = llvm.xor %renew, %true : i1
      scf.condition(%renew_cond)
    } do {
    ^bb0():
      // "After" region.
      %sig = dfg.pull %ctrl_0_6_rx : tuple<i1, i32>
      %idx = arith.constant 2 : index
      %count = "tuple.nth"(%sig, %idx) : (tuple<i1, i32>, index) -> i32

      %lb = arith.constant 0 : i32
      %step = arith.constant 1 : i32
      scf.for %iv = %lb to %count step %step : i32 {
        %mapcells0_0_0_0 = func.call @vclone(%mapcells_0_0_2_0) : (!VecMapCell) -> !VecMapCell
        dfg.push(%mapcells0_0_0_0) %mapcells0_0_0_0_tx : !VecMapCell
      }

      %idx1 = arith.constant 1 : index
      %renew_next_time = "tuple.nth"(%sig, %idx1) : (tuple<i1, i32>, index) -> i1

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %renew_next_time : i1
    }
  }
}

dfg.operator @fcdproc_ctrl inputs(%fcdproc_0_0_1_rx: !FCDPROC, %ctrl_0_1_rx: tuple<i1, i32>)
                           outputs(%gv_0_0_2_tx: !GpsVector) {
  dfg.loop inputs(%fcdproc_0_0_1_rx: !FCDPROC, %ctrl_0_1_rx: tuple<i1, i32>) outputs(%gv_0_0_2_tx: !GpsVector) {
    %init_renew = arith.constant 0 : i1
    %fcdproc_0_0_1_0 = dfg.pull %fcdproc_0_0_1_rx : !FCDPROC

    scf.while (%renew = %init_renew) : (i1) -> () {
      // Forward the argument (as result or "after" region argument).
      %true = arith.constant 1 : i1
      %renew_cond = llvm.xor %renew, %true : i1
      scf.condition(%renew_cond)
    } do {
    ^bb0():
      // "After" region.
      %sig = dfg.pull %ctrl_0_1_rx : tuple<i1, i32>
      %idx = arith.constant 2 : index
      %count = "tuple.nth"(%sig, %idx) : (tuple<i1, i32>, index) -> i32

      %lb = arith.constant 0 : i32
      %step = arith.constant 1 : i32
      scf.for %iv = %lb to %count step %step : i32 {
        %gv_0_0_2 = func.call @get_next_vector(%fcdproc_0_0_1_0) : (!FCDPROC) -> !GpsVector
        dfg.push(%gv_0_0_2) %gv_0_0_2_tx : !GpsVector
      }

      %idx1 = arith.constant 1 : index
      %renew_next_time = "tuple.nth"(%sig, %idx1) : (tuple<i1, i32>, index) -> i1

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %renew_next_time : i1
    }
  }
}

dfg.operator @counter_incr inputs(%counter_0_0_1_rx: i32, %ctrl_0_0_rx: tuple<i1, i32>, %something_0_0_0_rx: i8)
                           outputs(%counter_0_1_0_tx: i32) {
  dfg.loop inputs(%counter_0_0_1_rx: i32, %ctrl_0_0_rx: tuple<i1, i32>, %something_0_0_0_rx: i8) outputs(%counter_0_1_0_tx: i32) {
    %init_renew = arith.constant 0 : i1
    %counter_0_0_1_0 = dfg.pull %counter_0_0_1_rx : i32

    %res = scf.while (%renew = %init_renew, %counter = %counter_0_0_1_0) : (i1, i32) -> i32 {
      // Forward the argument (as result or "after" region argument).
      %true = arith.constant 1 : i1
      %renew_cond = llvm.xor %renew, %true : i1
      scf.condition(%renew_cond) %counter : i32
    } do {
    ^bb0(%ctr: i32):
      // "After" region.
      %sig = dfg.pull %ctrl_0_0_rx : tuple<i1, i32>
      %idx = arith.constant 2 : index
      %count = "tuple.nth"(%sig, %idx) : (tuple<i1, i32>, index) -> i32

      %lb = arith.constant 0 : i32
      %step = arith.constant 1 : i32
      scf.for %iv = %lb to %count step %step : i32 {
        %var_1 = dfg.pull %something_0_0_0_rx : i8
        // TODO: UPDATE TO THE COUNTER VALUE?!
        func.call @inc(%ctr, %var_1) : (i32, i8) -> ()
      }

      %idx1 = arith.constant 1 : index
      %renew_next_time = "tuple.nth"(%sig, %idx1) : (tuple<i1, i32>, index) -> i1

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %renew_next_time, %ctr : i1, i32
    }

    dfg.push(%counter_0_0_1_0) %counter_0_1_0_tx : i32
  }
}

dfg.operator @ctrl outputs(%ctrl_0_0_tx: tuple<i1, i32>, %ctrl_0_1_tx: tuple<i1, i32>, %ctrl_0_6_tx: tuple<i1, i32>) {
  %size = arith.constant 1000 : i32
  %ctrl = func.call @convenient_placeholder(%size) : (i32) -> tuple<i1, i32>

  dfg.push(%ctrl) %ctrl_0_0_tx : tuple<i1, i32>
  dfg.push(%ctrl) %ctrl_0_1_tx : tuple<i1, i32>
  dfg.push(%ctrl) %ctrl_0_6_tx : tuple<i1, i32>
}

dfg.operator @writevec inputs(%rsv_0_0_0_rx: !RoadSpeedVector)
                       outputs(%something_0_0_0_tx: i8) {
  dfg.loop inputs(%rsv_0_0_0_rx: !RoadSpeedVector) outputs(%something_0_0_0_tx: i8) {
    %var_0 = dfg.pull %rsv_0_0_0_rx : !RoadSpeedVector
    %something_0_0_0 = func.call @write_vector(%var_0) : (!RoadSpeedVector) -> i8
    dfg.push(%something_0_0_0) %something_0_0_0_tx : i8
  }
}

dfg.operator @cell_clone3 inputs(%mapcell_0_0_1_0_rx: !MapCell)
                          outputs(%mapcell_0_0_0_0_tx: !MapCell, %mapcell_0_0_0_0_2_tx: !MapCell) {
  dfg.loop inputs(%mapcell_0_0_1_0_rx: !MapCell) outputs(%mapcell_0_0_0_0_tx: !MapCell, %mapcell_0_0_0_0_2_tx: !MapCell) {
    %var_0 = dfg.pull %mapcell_0_0_1_0_rx : !MapCell
    %var_1 = func.call @clone(%var_0) : (!MapCell) -> !MapCell
    dfg.push(%var_0) %mapcell_0_0_0_0_tx : !MapCell
    dfg.push(%var_1) %mapcell_0_0_0_0_2_tx : !MapCell
  }
}

dfg.operator @cv_clone inputs(%cv_0_0_1_rx: !CandiVector)
                       outputs(%cv_0_0_0_0_tx: !CandiVector, %cv_0_0_0_0_2_tx: !CandiVector) {
  dfg.loop inputs(%cv_0_0_1_rx: !CandiVector) outputs(%cv_0_0_0_0_tx: !CandiVector, %cv_0_0_0_0_2_tx: !CandiVector) {
    %var_0 = dfg.pull %cv_0_0_1_rx : !CandiVector
    %var_1 = func.call @cclone(%var_0) : (!CandiVector) -> !CandiVector
    dfg.push(%var_0) %cv_0_0_0_0_tx : !CandiVector
    dfg.push(%var_1) %cv_0_0_0_0_2_tx : !CandiVector
  }
}

dfg.operator @gv_clone inputs(%gv_0_0_1_0_rx: !GpsVector)
                       outputs(%gv_0_0_0_0_tx: !GpsVector, %gv0_0_0_0_tx: !GpsVector) {
  dfg.loop inputs(%gv_0_0_1_0_rx: !GpsVector) outputs(%gv_0_0_0_0_tx: !GpsVector, %gv0_0_0_0_tx: !GpsVector) {
    %var_0 = dfg.pull %gv_0_0_1_0_rx : !GpsVector
    %gv_0_0_0_0 = func.call @gclone(%var_0) : (!GpsVector) -> !GpsVector
    dfg.push(%var_0) %gv_0_0_0_0_tx : !GpsVector
    dfg.push(%gv_0_0_0_0) %gv0_0_0_0_tx : !GpsVector
  }
}

dfg.operator @cell_clone2 inputs(%cin2: !MapCell)
                          outputs(%mapcell_0_0_1_0_tx: !MapCell, %mapcell0_0_0_0_tx: !MapCell) {
  dfg.loop inputs(%cin2: !MapCell) outputs(%mapcell_0_0_1_0_tx: !MapCell, %mapcell0_0_0_0_tx: !MapCell) {
    %var_0 = dfg.pull %cin2 : !MapCell
    %mapcell0_0_0_0 = func.call @clone(%var_0) : (!MapCell) -> !MapCell
    dfg.push(%mapcell0_0_0_0) %mapcell0_0_0_0_tx : !MapCell
    dfg.push(%var_0) %mapcell_0_0_1_0_tx : !MapCell
  }
}

dfg.operator @cell_clone inputs(%mapcell_0_0_2_rx: !MapCell)
                         outputs(%cout1: !MapCell, %cout2: !MapCell) {
  dfg.loop inputs(%mapcell_0_0_2_rx: !MapCell) outputs(%cout1: !MapCell, %cout2: !MapCell) {
    %value = dfg.pull %mapcell_0_0_2_rx : !MapCell
    %val2 = func.call @clone(%value) : (!MapCell) -> !MapCell
    dfg.push(%value) %cout1 : !MapCell
    dfg.push(%val2) %cout2 : !MapCell
  }
}

dfg.operator @make_dijkstra inputs(%cin1: !MapCell)
                            outputs(%dijkstra_0_0_0_tx: !Dijkstra) {
  dfg.loop inputs(%cin1: !MapCell) outputs(%dijkstra_0_0_0_tx: !Dijkstra) {
    %var_0 = dfg.pull %cin1 : !MapCell
    %dijkstra_0_0_0 = func.call @Dijkstra(%var_0) : (!MapCell) -> !Dijkstra
    dfg.push(%dijkstra_0_0_0) %dijkstra_0_0_0_tx : !Dijkstra
  }
}

dfg.operator @var_0_at inputs(%mapcells_0_0_1_0_rx: !VecMapCell, %index_0_0_0_rx: i64)
                       outputs(%mapcell_0_0_2_tx: !MapCell) {
  dfg.loop inputs(%mapcells_0_0_1_0_rx: !VecMapCell, %index_0_0_0_rx: i64) outputs(%mapcell_0_0_2_tx: !MapCell) {
    %var_0 = dfg.pull %mapcells_0_0_1_0_rx : !VecMapCell
    %var_1 = dfg.pull %index_0_0_0_rx : i64
    %mapcell_0_0_2 = func.call @vecat(%var_0, %var_1) : (!VecMapCell, i64) -> !MapCell
    dfg.push(%mapcell_0_0_2) %mapcell_0_0_2_tx : !MapCell
  }
}

dfg.operator @findcells inputs(%mapcells0_0_0_0_rx: !VecMapCell, %cell_0_0_0_0_rx: !GpsPosition)
                        outputs(%index_0_0_0_tx: i64) {
  dfg.loop inputs(%mapcells0_0_0_0_rx: !VecMapCell, %cell_0_0_0_0_rx: !GpsPosition) outputs(%index_0_0_0_tx: i64) {
    %var_0 = dfg.pull %mapcells0_0_0_0_rx : !VecMapCell
    %var_1 = dfg.pull %cell_0_0_0_0_rx : !GpsPosition
    %index_0_0_0 = func.call @findcell(%var_0, %var_1) : (!VecMapCell, !GpsPosition) -> i64
    dfg.push(%index_0_0_0) %index_0_0_0_tx : i64
  }
}

dfg.operator @gvat inputs(%gv_0_0_2_rx: !GpsVector)
                    outputs(%cell_0_0_0_0_tx: !GpsPosition, %gv_0_0_1_0_tx: !GpsVector) {
  dfg.loop inputs(%gv_0_0_2_rx: !GpsVector) outputs(%cell_0_0_0_0_tx: !GpsPosition, %gv_0_0_1_0_tx: !GpsVector) {
    %var_0 = dfg.pull %gv_0_0_2_rx : !GpsVector
    %idx = arith.constant 0 : index
    %cell_0_0_0_0 = func.call @gpsat(%var_0, %idx) : (!GpsVector, index) -> !GpsPosition
    dfg.push(%cell_0_0_0_0) %cell_0_0_0_0_tx : !GpsPosition
    dfg.push(%var_0) %gv_0_0_1_0_tx : !GpsVector
  }
}

dfg.operator @make_ctr outputs(%counter_0_0_1_tx: i32) {
  %counter_0_0_1 = func.call @Magic() : () -> i32
  dfg.push(%counter_0_0_1) %counter_0_0_1_tx : i32
}


dfg.operator @kernel_interpolate inputs(%rsvbb_0_0_0_rx: !RoadSpeedVector, %mapcell_0_0_0_0_1_rx: !MapCell)
                                 outputs(%rsv_0_0_0_tx: !RoadSpeedVector)
                                 attributes { dfg.path = "mma/component_interpolate.cpp" }

dfg.operator @kernel_viterbi inputs(%t_0_0_0_rx: !Trellis, %cv_0_0_0_0_1_rx: !CandiVector)
                             outputs(%rsvbb_0_0_0_tx: !RoadSpeedVector)
                             attributes { dfg.path = "mma/viterbi.cpp" }

dfg.operator @kernel_build_trellis inputs(%gv_0_0_0_0_rx: !GpsVector, %cv_0_0_0_0_2_rx: !CandiVector, %mapcell_0_0_0_0_2_rx: !MapCell, %dijkstra_0_0_0_rx: !Dijkstra)
                                   outputs(%t_0_0_0_tx: !Trellis)
                                   attributes { dfg.path = "mma/component_trellis.cpp" }

dfg.operator @kernel_projection inputs(%gv0_0_0_0_rx: !GpsVector, %mapcell0_0_0_0_rx: !MapCell)
                                outputs(%cv_0_0_1_tx: !CandiVector)
                                attributes { dfg.path = "mma/component_projection.cpp" }



func.func @run_dfg(%fcdproc_0_0_1: !FCDPROC) -> i32 {
  %counter_0_1_0_tx, %counter_0_1_0_rx = dfg.channel(i32)
  %counter_0_0_1_tx, %counter_0_0_1_rx = dfg.channel(i32)
  %ctrl_0_0_tx, %ctrl_0_0_rx = dfg.channel(tuple<i1, i32>)
  %fcdproc_0_0_1_tx, %fcdproc_0_0_1_rx = dfg.channel(!FCDPROC)
  %ctrl_0_1_tx, %ctrl_0_1_rx = dfg.channel(tuple<i1, i32>)
  %mapcells_0_0_2_tx, %mapcells_0_0_2_rx = dfg.channel(!VecMapCell)
  %ctrl_0_6_tx, %ctrl_0_6_rx = dfg.channel(tuple<i1, i32>)
  %gv_0_0_2_tx, %gv_0_0_2_rx = dfg.channel(!GpsVector)
  %cell_0_0_0_0_tx, %cell_0_0_0_0_rx = dfg.channel(!GpsPosition)
  %mapcells0_0_0_0_tx, %mapcells0_0_0_0_rx = dfg.channel(!VecMapCell)
  %mapcells_0_0_1_0_tx, %mapcells_0_0_1_0_rx = dfg.channel(!VecMapCell)
  %index_0_0_0_tx, %index_0_0_0_rx = dfg.channel(i64)
  %mapcell_0_0_2_tx, %mapcell_0_0_2_rx = dfg.channel(!MapCell)
  %gv_0_0_1_0_tx, %gv_0_0_1_0_rx = dfg.channel(!GpsVector)
  %mapcell0_0_0_0_tx, %mapcell0_0_0_0_rx = dfg.channel(!MapCell)
  %gv0_0_0_0_tx, %gv0_0_0_0_rx = dfg.channel(!GpsVector)
  %cv_0_0_1_tx, %cv_0_0_1_rx = dfg.channel(!CandiVector)
  %mapcell_0_0_1_0_tx, %mapcell_0_0_1_0_rx = dfg.channel(!MapCell)
  %dijkstra_0_0_0_tx, %dijkstra_0_0_0_rx = dfg.channel(!Dijkstra)
  %gv_0_0_0_0_tx, %gv_0_0_0_0_rx = dfg.channel(!GpsVector)
  %cv_0_0_0_0_tx, %cv_0_0_0_0_1_rx = dfg.channel(!CandiVector)
  %cv_0_0_0_0_2_tx, %cv_0_0_0_0_2_rx = dfg.channel(!CandiVector)
  %t_0_0_0_tx, %t_0_0_0_rx = dfg.channel(!Trellis)
  %mapcell_0_0_0_0_tx, %mapcell_0_0_0_0_1_rx = dfg.channel(!MapCell)
  %mapcell_0_0_0_0_2_tx, %mapcell_0_0_0_0_2_rx = dfg.channel(!MapCell)
  %rsvbb_0_0_0_tx, %rsvbb_0_0_0_rx = dfg.channel(!RoadSpeedVector)
  %rsv_0_0_0_tx, %rsv_0_0_0_rx = dfg.channel(!RoadSpeedVector)
  %something_0_0_0_tx, %something_0_0_0_rx = dfg.channel(i8)

  %cout1, %cin1 = dfg.channel(!MapCell)
  %cout2, %cin2 = dfg.channel(!MapCell)

  // inputs
  dfg.push(%fcdproc_0_0_1) %fcdproc_0_0_1_tx : !FCDPROC

  // run DFG
  dfg.instantiate @get_map outputs(%mapcells_0_0_2_tx) : () -> !VecMapCell
  dfg.instantiate @mapcells_ctrl inputs(%mapcells_0_0_2_rx, %ctrl_0_6_rx) outputs(%mapcells0_0_0_0_tx) : (!VecMapCell, tuple<i1, i32>) -> !VecMapCell
  dfg.instantiate @fcdproc_ctrl inputs(%fcdproc_0_0_1_rx, %ctrl_0_1_rx) outputs(%gv_0_0_2_tx) : (!FCDPROC, tuple<i1, i32>) -> !GpsVector
  dfg.instantiate @counter_incr inputs(%counter_0_0_1_rx, %ctrl_0_0_rx, %something_0_0_0_rx) outputs(%counter_0_1_0_tx) : (i32, tuple<i1, i32>, i8) -> i32
  dfg.instantiate @ctrl outputs(%ctrl_0_0_tx, %ctrl_0_1_tx, %ctrl_0_6_tx) : () -> (tuple<i1, i32>, tuple<i1, i32>, tuple<i1, i32>)
  dfg.instantiate @cell_clone3 inputs(%mapcell_0_0_1_0_rx) outputs(%mapcell_0_0_0_0_tx, %mapcell_0_0_0_0_2_tx) : (!MapCell) -> (!MapCell, !MapCell)
  dfg.instantiate @cv_clone inputs(%cv_0_0_1_rx) outputs(%cv_0_0_0_0_tx, %cv_0_0_0_0_2_tx) : (!CandiVector) -> (!CandiVector, !CandiVector)
  dfg.instantiate @gv_clone inputs(%gv_0_0_1_0_rx) outputs(%gv_0_0_0_0_tx, %gv0_0_0_0_tx) : (!GpsVector) -> (!GpsVector, !GpsVector)
  dfg.instantiate @cell_clone2 inputs(%cin2) outputs(%mapcell_0_0_1_0_tx, %mapcell0_0_0_0_tx) : (!MapCell) -> (!MapCell, !MapCell)
  dfg.instantiate @cell_clone inputs(%mapcell_0_0_2_rx) outputs(%cout1, %cout2) : (!MapCell) -> (!MapCell, !MapCell)
  dfg.instantiate @make_dijkstra inputs(%cin1) outputs(%dijkstra_0_0_0_tx) : (!MapCell) -> !Dijkstra
  dfg.instantiate @var_0_at inputs(%mapcells_0_0_1_0_rx, %index_0_0_0_rx) outputs(%mapcell_0_0_2_tx) : (!VecMapCell, i64) -> !MapCell
  dfg.instantiate @findcells inputs(%mapcells0_0_0_0_rx, %cell_0_0_0_0_rx) outputs(%index_0_0_0_tx) : (!VecMapCell, !GpsPosition) -> i64
  dfg.instantiate @gvat inputs(%gv_0_0_2_rx) outputs(%cell_0_0_0_0_tx, %gv_0_0_1_0_tx) : (!GpsVector) -> (!GpsPosition, !GpsVector)
  dfg.instantiate @make_ctr outputs(%counter_0_0_1_tx) : () -> i32
  dfg.instantiate offloaded @kernel_interpolate inputs(%rsvbb_0_0_0_rx, %mapcell_0_0_0_0_1_rx) outputs(%rsv_0_0_0_tx) : (!RoadSpeedVector, !MapCell) -> !RoadSpeedVector
  dfg.instantiate offloaded @kernel_viterbi inputs(%t_0_0_0_rx, %cv_0_0_0_0_1_rx) outputs(%rsvbb_0_0_0_tx) : (!Trellis, !CandiVector) -> !RoadSpeedVector
  dfg.instantiate offloaded @kernel_build_trellis inputs(%gv_0_0_0_0_rx, %cv_0_0_0_0_2_rx, %mapcell_0_0_0_0_2_rx, %dijkstra_0_0_0_rx) outputs(%t_0_0_0_tx) : (!GpsVector, !CandiVector, !MapCell, !Dijkstra) -> !Trellis
  dfg.instantiate offloaded @kernel_projection inputs(%gv0_0_0_0_rx, %mapcell0_0_0_0_rx) outputs(%cv_0_0_1_tx) : (!GpsVector, !MapCell) -> !CandiVector

  // outputs
  %res = dfg.pull %counter_0_1_0_rx : i32

  return %res : i32
}
