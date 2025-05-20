// ----------------------------------------------------------------------------
//
// Multi-Dataflow Composer tool - Platform Composer
// Sbox 2x1 module 
// Date: 2025/05/20 12:47:44
//
// ----------------------------------------------------------------------------

module sbox2x1 #(
	parameter SIZE = 32
)(
	output [SIZE-1 : 0] out1_data,
	input out1_full_n,
	output out1_write,
	input [SIZE-1 : 0] in1_data,
	input [SIZE-1 : 0] in2_data,
	output in1_full_n,
	output in2_full_n,
	input in1_write,
	input in2_write,
	input sel
);


assign out1_data = sel ? in2_data : in1_data;
assign out1_write = sel ? in2_write : in1_write;
assign in1_full_n = sel ? {1{1'b0}} : out1_full_n;
assign in2_full_n = sel ? out1_full_n : {1{1'b0}};

endmodule
