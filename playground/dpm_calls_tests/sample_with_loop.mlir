func.func private @source() -> (i64, i64, i64)
func.func private @mul(i64, i64) -> i64
func.func private @sink(i64)
dfg.process @multiply inputs(%arg0 : i64, %arg1 : i64)  outputs(%arg2 : i64)  attributes {multiplicity = array<i64>} {
  dfg.loop inputs (%arg0 : i64, %arg1 : i64) outputs(%arg2 : i64) {
    %0 = dfg.pull %arg0 : i64
    %1 = dfg.pull %arg1 : i64
    %2 = arith.muli %0, %1 : i64
    dfg.push (%2) %arg2 : i64
  }
}
dfg.process @source_wrap outputs(%arg0 : i64, %arg1 : i64, %arg2 : i64)  attributes {multiplicity = array<i64>} {
  %0:3 = func.call @source() : () -> (i64, i64, i64)
  dfg.push (%0#0) %arg0 : i64
  dfg.push (%0#1) %arg1 : i64
  dfg.push (%0#2) %arg2 : i64
}
dfg.process @sum_wrap inputs(%arg0 : i64, %arg1 : i64)  outputs(%arg2 : i64)  attributes {multiplicity = array<i64>} {
  %0 = dfg.pull %arg0 : i64
  %1 = dfg.pull %arg1 : i64
  %2 = arith.addi %0, %1 : i64
  dfg.push (%2) %arg2 : i64
}
dfg.process @sink_wrap inputs(%arg0 : i64)  attributes {multiplicity = array<i64>} {
  %0 = dfg.pull %arg0 : i64
  func.call @sink(%0) : (i64) -> ()
}
dfg.region @multiplyRegion inputs(%arg0 : i64, %arg1 : i64)  outputs(%arg2 : i64)  {
  dfg.instantiate @multiply inputs(%arg0, %arg1) outputs(%arg2) : (i64, i64) -> i64
}
dfg.region @mainRegion {
  %in_chan, %out_chan = dfg.channel() : i64
  %in_chan_0, %out_chan_1 = dfg.channel() : i64
  %in_chan_2, %out_chan_3 = dfg.channel() : i64
  %in_chan_4, %out_chan_5 = dfg.channel() : i64
  %in_chan_6, %out_chan_7 = dfg.channel() : i64
  dfg.instantiate @source_wrap outputs(%in_chan, %in_chan_0, %in_chan_4) : () -> (i64, i64, i64)
  dfg.instantiate @sum_wrap inputs(%out_chan, %out_chan_1) outputs(%in_chan_2) : (i64, i64) -> i64
  dfg.embed @multiplyRegion inputs(%out_chan_3, %out_chan_5) outputs(%in_chan_6) : (i64, i64) -> i64
  dfg.instantiate @sink_wrap inputs(%out_chan_7) : (i64) -> ()
}

