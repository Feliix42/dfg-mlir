!msg_type = !llvm.array<5 x i8>
!signature_type = !llvm.array<2176 x i8>

llvm.func @lms_fill_message(i32, !llvm.ptr) -> ()
llvm.func @lms_sign(!llvm.ptr, !llvm.ptr) -> ()
llvm.func @lms_verify(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @lms_sink(i32) -> ()
llvm.func @lms_get_iterations() -> i32
llvm.func @lms_done(i32) -> ()

dfg.process @source inputs() outputs(%out_a: !msg_type) {

	%lb = arith.constant 0 : i32
	%ub = llvm.call @lms_get_iterations() : () -> (i32)
	%step = arith.constant 1 : i32

	scf.for %index = %lb to %ub step %step : i32 {
		%len = arith.constant 1 : i32
		%array_data = llvm.alloca %len x !msg_type : (i32) -> !llvm.ptr
		llvm.call @lms_fill_message(%index, %array_data) : (i32, !llvm.ptr) -> ()
		%result = llvm.load %array_data : !llvm.ptr -> !msg_type
		dfg.push(%result) %out_a : !msg_type
	}
}

dfg.operator @sign inputs(%msg_in: !msg_type) outputs(%msg_out: !msg_type, %out_a: !signature_type) {
	%len = arith.constant 1 : i32
	%msg_address = llvm.alloca %len x !msg_type : (i32) -> !llvm.ptr
	llvm.store %msg_in, %msg_address : !msg_type, !llvm.ptr

	%sig_address = llvm.alloca %len x !signature_type : (i32) -> !llvm.ptr

	llvm.call @lms_sign(%msg_address, %sig_address) : (!llvm.ptr, !llvm.ptr) -> ()

	%sig_object = llvm.load %sig_address : !llvm.ptr -> !signature_type

	dfg.output %msg_in, %sig_object : !msg_type, !signature_type
}

dfg.operator @send inputs(%msg_in : !msg_type, %sig_in: !signature_type) outputs(%d1 : !msg_type, %d2 : !signature_type) {
	dfg.output %msg_in, %sig_in : !msg_type, !signature_type
}

dfg.operator @verify inputs(%msg_in: !msg_type, %sig_in: !signature_type) outputs(%d : i32) {
	%len = arith.constant 1 : i32
	%msg_address = llvm.alloca %len x !msg_type : (i32) -> !llvm.ptr
	llvm.store %msg_in, %msg_address : !msg_type, !llvm.ptr

	%sig_address = llvm.alloca %len x !signature_type : (i32) -> !llvm.ptr
	llvm.store %sig_in, %sig_address : !signature_type, !llvm.ptr

	%res = llvm.call @lms_verify(%msg_address, %sig_address) : (!llvm.ptr, !llvm.ptr) -> i32

	dfg.output %res : i32

}

dfg.process @sink inputs(%is_valid: i32) outputs() {
	%lb = arith.constant 0 : i32
	%ub = llvm.call @lms_get_iterations() : () -> (i32)
	%step = arith.constant 1 : i32
	scf.for %index = %lb to %ub step %step : i32 {
		%bool = dfg.pull %is_valid : i32
		llvm.call @lms_sink(%bool) : (i32) -> ()
	}
	llvm.call @lms_done(%ub) : (i32) -> ()
}

dfg.region @signParallelRegion inputs(%msg_in : !msg_type) outputs(%msg_out: !msg_type, %sig_out: !signature_type) is_parallel {
	dfg.instantiate @sign inputs(%msg_in) outputs(%msg_out, %sig_out) : (!msg_type) -> (!msg_type, !signature_type)
}

dfg.region @verifyParallelRegion inputs(%msg_in : !msg_type, %sig_in: !signature_type) outputs(%is_valid: i32) is_parallel {
	dfg.instantiate @verify inputs(%msg_in, %sig_in) outputs(%is_valid) : (!msg_type, !signature_type) -> i32	
}

dfg.region @mainRegion inputs() outputs() {

	%sign_msg_in, %sign_msg_out = dfg.channel() : !msg_type

	%send_sig_in, %send_sig_out = dfg.channel() : !signature_type
	%send_msg_in, %send_msg_out = dfg.channel() : !msg_type

	%rec_sig_in, %rec_sig_out = dfg.channel() : !signature_type
	%rec_msg_in, %rec_msg_out = dfg.channel() : !msg_type

	%is_valid_in, %is_valid_out = dfg.channel() : i32
		
	dfg.instantiate @source inputs() outputs(%sign_msg_in) : () -> (!msg_type)

	dfg.embed @signParallelRegion inputs(%sign_msg_out) outputs(%send_msg_in, %send_sig_in) : (!msg_type) -> (!msg_type, !signature_type)

	dfg.instantiate @send inputs(%send_msg_out, %send_sig_out) outputs(%rec_msg_in, %rec_sig_in) : (!msg_type, !signature_type) -> (!msg_type, !signature_type)

	dfg.embed @verifyParallelRegion inputs(%rec_msg_out, %rec_sig_out) outputs(%is_valid_in) : (!msg_type, !signature_type) -> i32

	dfg.instantiate @sink inputs(%is_valid_out) outputs() : (i32) -> ()

}
