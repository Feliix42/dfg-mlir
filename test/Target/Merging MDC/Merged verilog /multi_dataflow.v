// ----------------------------------------------------------------------------
//
// Multi-Dataflow Composer tool - Platform Composer
// Multi-Dataflow Network module 
// Date: 2025/05/20 12:47:44
//
// ----------------------------------------------------------------------------

module multi_dataflow (
	// Input(s)
	input [31 : 0] in0_data,
	output in0_full_n,
	input in0_write,
	
	// Output(s)
	input [31 : 0] in1_data,
	output in1_full_n,
	input in1_write,
	
	// Output(s)
	output [31 : 0] out0_data,
	input out0_full_n,
	output out0_write,
	output [31 : 0] out1_data,
	input out1_full_n,
	output out1_write,
	
	// Dynamic Parameter(s)
	
	// Monitoring
	
	// Configuration ID
	input [7:0] ID,
	
	
	// System Signal(s)		
	input ap_clk,
	input ap_rst
);	

// internal signals
// ----------------------------------------------------------------------------
// Sboxes Config Wire(s)
wire [1 : 0] sel;
		


// Actors Wire(s)
	
// actor accumulator_0
wire [31 : 0] fifo_gen_accumulator_0_arg0_data;
wire fifo_gen_accumulator_0_arg0_full_n;
wire fifo_gen_accumulator_0_arg0_write;
wire [31 : 0] accumulator_0_arg0_data;
wire accumulator_0_arg0_empty_n;
wire accumulator_0_arg0_read;
wire [31 : 0] accumulator_0_arg1_data;
wire accumulator_0_arg1_full_n;
wire accumulator_0_arg1_write;
	
// actor accumulator_1
wire [31 : 0] fifo_gen_accumulator_1_arg0_data;
wire fifo_gen_accumulator_1_arg0_full_n;
wire fifo_gen_accumulator_1_arg0_write;
wire [31 : 0] accumulator_1_arg0_data;
wire accumulator_1_arg0_empty_n;
wire accumulator_1_arg0_read;
wire [31 : 0] accumulator_1_arg1_data;
wire accumulator_1_arg1_full_n;
wire accumulator_1_arg1_write;
	
// actor lshifter_0
wire [31 : 0] fifo_gen_lshifter_0_arg0_data;
wire fifo_gen_lshifter_0_arg0_full_n;
wire fifo_gen_lshifter_0_arg0_write;
wire [31 : 0] lshifter_0_arg0_data;
wire lshifter_0_arg0_empty_n;
wire lshifter_0_arg0_read;
wire [31 : 0] lshifter_0_arg1_data;
wire lshifter_0_arg1_full_n;
wire lshifter_0_arg1_write;
	
// actor sbox_0
wire [31 : 0] sbox_0_in1_data;
wire sbox_0_in1_full_n;
wire sbox_0_in1_write;
wire [31 : 0] sbox_0_out1_data;
wire sbox_0_out1_full_n;
wire sbox_0_out1_write;
wire [31 : 0] sbox_0_out2_data;
wire sbox_0_out2_full_n;
wire sbox_0_out2_write;
	
// actor sbox_1
wire [31 : 0] sbox_1_in1_data;
wire sbox_1_in1_full_n;
wire sbox_1_in1_write;
wire [31 : 0] sbox_1_in2_data;
wire sbox_1_in2_full_n;
wire sbox_1_in2_write;
wire [31 : 0] sbox_1_out1_data;
wire sbox_1_out1_full_n;
wire sbox_1_out1_write;
// ----------------------------------------------------------------------------

// body
// ----------------------------------------------------------------------------
// Network Configurator
configurator config_0 (
	.sel(sel),
	.ID(ID)
);



// fifo_gen_accumulator_0_arg0
fifo_gen #(
	.width(32),
	.depth(64)
) fifo_gen_accumulator_0_arg0(
	.din(fifo_gen_accumulator_0_arg0_data),
	.empty_n(accumulator_0_arg0_empty_n),
	.read(accumulator_0_arg0_read),
	.dout(accumulator_0_arg0_data),
	.full_n(fifo_gen_accumulator_0_arg0_full_n),
	.write(fifo_gen_accumulator_0_arg0_write),
	
	// System Signal(s)
	.clk(ap_clk),
	.ap_rst(ap_rst)
);

// actor accumulator_0
accumulator actor_accumulator_0 (
	// Input Signal(s)
	.arg0_dout(accumulator_0_arg0_data),
	.arg0_empty_n(accumulator_0_arg0_empty_n),
	.arg0_read(accumulator_0_arg0_read)
	,
	
	// Output Signal(s)
	.arg1_din(accumulator_0_arg1_data),
	.arg1_full_n(accumulator_0_arg1_full_n),
	.arg1_write(accumulator_0_arg1_write)
		,
	
	// System Signal(s)
	.ap_clk(ap_clk),
	.ap_rst(ap_rst)
);


// fifo_gen_accumulator_1_arg0
fifo_gen #(
	.width(32),
	.depth(64)
) fifo_gen_accumulator_1_arg0(
	.din(fifo_gen_accumulator_1_arg0_data),
	.empty_n(accumulator_1_arg0_empty_n),
	.read(accumulator_1_arg0_read),
	.dout(accumulator_1_arg0_data),
	.full_n(fifo_gen_accumulator_1_arg0_full_n),
	.write(fifo_gen_accumulator_1_arg0_write),
	
	// System Signal(s)
	.clk(ap_clk),
	.ap_rst(ap_rst)
);

// actor accumulator_1
accumulator actor_accumulator_1 (
	// Input Signal(s)
	.arg0_dout(accumulator_1_arg0_data),
	.arg0_empty_n(accumulator_1_arg0_empty_n),
	.arg0_read(accumulator_1_arg0_read)
	,
	
	// Output Signal(s)
	.arg1_din(accumulator_1_arg1_data),
	.arg1_full_n(accumulator_1_arg1_full_n),
	.arg1_write(accumulator_1_arg1_write)
		,
	
	// System Signal(s)
	.ap_clk(ap_clk),
	.ap_rst(ap_rst)
);


// fifo_gen_lshifter_0_arg0
fifo_gen #(
	.width(32),
	.depth(64)
) fifo_gen_lshifter_0_arg0(
	.din(fifo_gen_lshifter_0_arg0_data),
	.empty_n(lshifter_0_arg0_empty_n),
	.read(lshifter_0_arg0_read),
	.dout(lshifter_0_arg0_data),
	.full_n(fifo_gen_lshifter_0_arg0_full_n),
	.write(fifo_gen_lshifter_0_arg0_write),
	
	// System Signal(s)
	.clk(ap_clk),
	.ap_rst(ap_rst)
);

// actor lshifter_0
lshifter actor_lshifter_0 (
	// Input Signal(s)
	.arg0_dout(lshifter_0_arg0_data),
	.arg0_empty_n(lshifter_0_arg0_empty_n),
	.arg0_read(lshifter_0_arg0_read)
	,
	
	// Output Signal(s)
	.arg1_din(lshifter_0_arg1_data),
	.arg1_full_n(lshifter_0_arg1_full_n),
	.arg1_write(lshifter_0_arg1_write)
		,
	
	// System Signal(s)
	.ap_clk(ap_clk),
	.ap_rst(ap_rst)
);



// actor sbox_0
sbox1x2 #(
	.SIZE(32)
)
sbox_0 (
	// Input Signal(s)
	.in1_data(sbox_0_in1_data),
	.in1_full_n(sbox_0_in1_full_n),
	.in1_write(sbox_0_in1_write),
	
	// Output Signal(s)
	.out1_data(sbox_0_out1_data),
	.out1_full_n(sbox_0_out1_full_n),
	.out1_write(sbox_0_out1_write),
	.out2_data(sbox_0_out2_data),
	.out2_full_n(sbox_0_out2_full_n),
	.out2_write(sbox_0_out2_write),
	
	// Selector
	.sel(sel[0])	
);


// actor sbox_1
sbox2x1 #(
	.SIZE(32)
)
sbox_1 (
	// Input Signal(s)
	.in1_data(sbox_1_in1_data),
	.in1_full_n(sbox_1_in1_full_n),
	.in1_write(sbox_1_in1_write),
	.in2_data(sbox_1_in2_data),
	.in2_full_n(sbox_1_in2_full_n),
	.in2_write(sbox_1_in2_write),
	
	// Output Signal(s)
	.out1_data(sbox_1_out1_data),
	.out1_full_n(sbox_1_out1_full_n),
	.out1_write(sbox_1_out1_write),
	
	// Selector
	.sel(sel[1])	
);

// Module(s) Assignments
assign fifo_gen_accumulator_0_arg0_data = in0_data;
assign in0_full_n = fifo_gen_accumulator_0_arg0_full_n;
assign fifo_gen_accumulator_0_arg0_write = in0_write;

assign fifo_gen_accumulator_1_arg0_data = in1_data;
assign in1_full_n = fifo_gen_accumulator_1_arg0_full_n;
assign fifo_gen_accumulator_1_arg0_write = in1_write;

assign sbox_1_in1_data = sbox_0_out1_data;
assign sbox_0_out1_full_n = sbox_1_in1_full_n;
assign sbox_1_in1_write = sbox_0_out1_write;

assign out1_data = accumulator_1_arg1_data;
assign accumulator_1_arg1_full_n = out1_full_n;
assign out1_write = accumulator_1_arg1_write;

assign sbox_0_in1_data = accumulator_0_arg1_data;
assign accumulator_0_arg1_full_n = sbox_0_in1_full_n;
assign sbox_0_in1_write = accumulator_0_arg1_write;

assign fifo_gen_lshifter_0_arg0_data = sbox_0_out2_data;
assign sbox_0_out2_full_n = fifo_gen_lshifter_0_arg0_full_n;
assign fifo_gen_lshifter_0_arg0_write = sbox_0_out2_write;

assign out0_data = sbox_1_out1_data;
assign sbox_1_out1_full_n = out0_full_n;
assign out0_write = sbox_1_out1_write;

assign sbox_1_in2_data = lshifter_0_arg1_data;
assign lshifter_0_arg1_full_n = sbox_1_in2_full_n;
assign sbox_1_in2_write = lshifter_0_arg1_write;

endmodule
