dfg.operator @lshifter inputs(%in: i32) outputs(%out: i32) {
  
    %0 = arith.constant 2 : i32
    %1 = arith.muli %in, %0 : i32   
    dfg.output %1: i32  
}
dfg.operator @accumulator inputs(%in: i32) outputs(%out: i32) {
  
    %0 = arith.constant 1 : i32
    %1 = arith.addi %in, %0 : i32      
    dfg.output %1: i32 
}

dfg.region @top inputs(%arg0: i32) outputs(%arg1: i32) {

	%0:2 = dfg.channel(1) : i32
    	dfg.instantiate @accumulator inputs(%arg0) outputs(%0#0) : (i32) -> i32
    	dfg.instantiate @lshifter inputs(%0#1) outputs(%arg1) : (i32) -> i32
}

