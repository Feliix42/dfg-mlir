dfg.operator @accumulator inputs(%in: i32) outputs(%out: i32) {
        
    %0 = arith.constant 1 : i32
    %1 = arith.addi %in, %0 : i32      
    dfg.output %1: i32
    
}

dfg.region @top inputs(%arg0: i32, %arg1: i32) outputs(%arg3: i32, %arg4: i32) {

    	dfg.instantiate @accumulator inputs(%arg0) outputs(%arg3) : (i32) -> i32
    	dfg.instantiate @accumulator inputs(%arg1) outputs(%arg4) : (i32) -> i32
}

