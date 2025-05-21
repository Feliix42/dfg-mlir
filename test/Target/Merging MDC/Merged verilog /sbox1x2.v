// ----------------------------------------------------------------------------
//
// Multi-Dataflow Composer tool - Platform Composer
// Sbox 1x2 module 
// Date: 2025/05/20 12:47:44
//
// ----------------------------------------------------------------------------

module sbox1x2 #(
	parameter SIZE = 32
)(
	output [SIZE-1 : 0] out1_data,
	output [SIZE-1 : 0] out2_data,
	input out1_full_n,
	input out2_full_n,
	output out1_write,
	output out2_write,
	input [SIZE-1 : 0] in1_data,
	output in1_full_n,
	input in1_write,
	input sel
);


assign out1_data = sel ? {SIZE{1'b0}} : in1_data;
assign out2_data = sel ? in1_data : {SIZE{1'b0}};
assign out1_write = sel ? {1{1'b0}} : in1_write;
assign out2_write = sel ? in1_write : {1{1'b0}};
assign in1_full_n = sel ? out2_full_n : out1_full_n;

endmodule
