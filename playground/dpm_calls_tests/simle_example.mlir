llvm.func @print(i32) -> ()

dfg.process @source inputs() outputs(%a_out : i32, %b_out : i32) {
	%lb = arith.constant 0 : i32
	%ub = arith.constant 10 : i32
	%step = arith.constant 1 : i32
	scf.for %index = %lb to %ub step %step : i32 {
		dfg.push(%index) %a_out : i32
		dfg.push(%index) %b_out : i32
	}
}

dfg.operator @multiply inputs(%a_in : i32, %b_in : i32) outputs(%a_out : i32) {
	%res = arith.muli %a_in, %b_in : i32
	dfg.output %res : i32
}

dfg.process @sink inputs(%a_in : i32) {
	%lb = arith.constant 0 : i32
	%ub = arith.constant 10 : i32
	%step = arith.constant 1 : i32
	scf.for %index = %lb to %ub step %step : i32 {
		%res = dfg.pull %a_in : i32
		llvm.call @print(%res) : (i32) -> ()
	}
}

dfg.region @parallelRegion inputs(%a_in : i32, %b_in : i32) outputs(%a_out : i32) is_parallel {
	dfg.instantiate @multiply inputs(%a_in, %b_in) outputs(%a_out) : (i32, i32) -> (i32)	
}

dfg.region @mainRegion {
	%source_1_in, %source_1_out = dfg.channel() : i32
	%source_2_in, %source_2_out = dfg.channel() : i32
	%sink_in, %sink_out = dfg.channel() : i32

	dfg.instantiate @source inputs() outputs(%source_1_in, %source_2_in) : () -> (i32, i32)
	dfg.embed @parallelRegion inputs(%source_1_out, %source_2_out) outputs(%sink_in) : (i32, i32) -> (i32)
	dfg.instantiate @sink inputs(%sink_out) outputs() : (i32) -> ()
}
