!audio_type = !llvm.array<65535 x f32>

llvm.func @fill_data(!llvm.ptr) -> ()
llvm.func @filter_data(!llvm.ptr, !llvm.ptr) -> ()
llvm.func @push_done() -> ()

dfg.process @source outputs(%out_a : !audio_type) {

	%lb = arith.constant 0 : i32
	%ub = arith.constant 30000 : i32
	%step = arith.constant 1 : i32

	%len = arith.constant 1 : i32
	%data_ptr = llvm.alloca %len x !audio_type : (i32) -> (!llvm.ptr)

	scf.for %index = %lb to %ub step %step : i32 {
		llvm.call @fill_data(%data_ptr) : (!llvm.ptr) -> ()
		%object = llvm.load %data_ptr : !llvm.ptr -> !audio_type
		dfg.push(%object) %out_a : !audio_type
	}
	llvm.call @push_done() : () -> ()
}


dfg.operator @filter inputs(%a_in : !audio_type) outputs(%a_out : !audio_type) {
	%len = arith.constant 1 : i32

	%input_data_ptr = llvm.alloca %len x !audio_type : (i32) -> (!llvm.ptr)
	llvm.store %a_in, %input_data_ptr : !audio_type, !llvm.ptr

	%output_data_ptr = llvm.alloca %len x !audio_type : (i32) -> (!llvm.ptr)

	llvm.call @filter_data(%input_data_ptr, %output_data_ptr) : (!llvm.ptr, !llvm.ptr) -> ()
	%object = llvm.load %output_data_ptr : !llvm.ptr -> !audio_type
	dfg.output %object : !audio_type
}


dfg.process @sink inputs(%a_in : !audio_type) outputs() {

	%lb = arith.constant 0 : i32
	%ub = arith.constant 30000 : i32
	%step = arith.constant 1 : i32

	scf.for %index = %lb to %ub step %step : i32 {
		dfg.pull %a_in : !audio_type
	}

}

dfg.region @filterPr inputs(%a : !audio_type) outputs(%b : !audio_type) is_parallel {
	dfg.instantiate @filter inputs(%a) outputs(%b) : (!audio_type) -> (!audio_type)
}

dfg.region @mainRegion inputs() outputs() {

	%sourcein, %sourceout = dfg.channel() : !audio_type
	%sinkin, %sinkout = dfg.channel() : !audio_type

	dfg.instantiate @source inputs() outputs(%sourcein) : () -> (!audio_type)
	dfg.embed @filterPr inputs(%sourceout) outputs(%sinkin) : (!audio_type) -> (!audio_type)
	dfg.instantiate @sink inputs(%sinkout) outputs() : (!audio_type) -> ()
}
