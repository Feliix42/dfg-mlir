`timescale 1 ns / 1 ps
// ----------------------------------------------------------------------------
//
// Multi-Dataflow Composer tool - Platform Composer
// Multi-Dataflow Test Bench module 
// Date: 2025/05/20 12:47:44
//
// Please note that the testbench manages only common signals to dataflows
// - clock system signals
// - reset system signals
// - dataflow communication signals
//
// ----------------------------------------------------------------------------

module tb_multi_dataflow;

	// test bench parameters
	// ----------------------------------------------------------------------------
	parameter AP_CLK_PERIOD = 10;
	
	parameter IN0_TOP_FILE = "in0_top_file.mem";
	parameter IN0_TOP_SIZE = 64;
	parameter IN0_TOP1_FILE = "in0_top1_file.mem";
	parameter IN0_TOP1_SIZE = 64;
	parameter IN1_TOP_FILE = "in1_top_file.mem";
	parameter IN1_TOP_SIZE = 64;
	parameter IN1_TOP1_FILE = "in1_top1_file.mem";
	parameter IN1_TOP1_SIZE = 64;
	
	parameter OUT0_TOP_FILE = "out0_top_file.mem";
	parameter OUT0_TOP_SIZE = 64;
	parameter OUT0_TOP1_FILE = "out0_top1_file.mem";
	parameter OUT0_TOP1_SIZE = 64;
	parameter OUT1_TOP_FILE = "out1_top_file.mem";
	parameter OUT1_TOP_SIZE = 64;
	parameter OUT1_TOP1_FILE = "out1_top1_file.mem";
	parameter OUT1_TOP1_SIZE = 64;
	
	// ----------------------------------------------------------------------------
	
	// multi_dataflow signals
	// ----------------------------------------------------------------------------
	reg start_feeding;
	reg [31 : 0] in0_data;
	wire in0_full_n;
	reg in0_write;
	reg [31:0] in0_top_file_data [IN0_TOP_SIZE-1:0];
	reg [31:0] in0_top1_file_data [IN0_TOP1_SIZE-1:0];
	integer in0_i = 0;
	reg [31 : 0] in1_data;
	wire in1_full_n;
	reg in1_write;
	reg [31:0] in1_top_file_data [IN1_TOP_SIZE-1:0];
	reg [31:0] in1_top1_file_data [IN1_TOP1_SIZE-1:0];
	integer in1_i = 0;
	
	wire [31 : 0] out0_data;
	reg out0_full_n;
	wire out0_write;
	reg [31:0] out0_top_file_data [OUT0_TOP_SIZE-1:0];
	reg [31:0] out0_top1_file_data [OUT0_TOP1_SIZE-1:0];
	integer out0_i = 0;
	wire [31 : 0] out1_data;
	reg out1_full_n;
	wire out1_write;
	reg [31:0] out1_top_file_data [OUT1_TOP_SIZE-1:0];
	reg [31:0] out1_top1_file_data [OUT1_TOP1_SIZE-1:0];
	integer out1_i = 0;
	
	
	reg [7:0] ID;
	
	reg ap_clk;
	reg ap_rst;
	// ----------------------------------------------------------------------------

	// network input and output files
	// ----------------------------------------------------------------------------
	initial
	 	$readmemh(IN0_TOP_FILE, in0_top_file_data);
	initial
	 	$readmemh(IN1_TOP_FILE, in1_top_file_data);
	initial
		$readmemh(OUT0_TOP_FILE, out0_top_file_data);
	initial
		$readmemh(OUT1_TOP_FILE, out1_top_file_data);
	initial
	 	$readmemh(IN0_TOP1_FILE, in0_top1_file_data);
	initial
	 	$readmemh(IN1_TOP1_FILE, in1_top1_file_data);
	initial
		$readmemh(OUT0_TOP1_FILE, out0_top1_file_data);
	initial
		$readmemh(OUT1_TOP1_FILE, out1_top1_file_data);
	// ----------------------------------------------------------------------------

	// dut
	// ----------------------------------------------------------------------------
	multi_dataflow dut (
		.in0_data(in0_data),
		.in0_full_n(in0_full_n),
		.in0_write(in0_write),
		
		.in1_data(in1_data),
		.in1_full_n(in1_full_n),
		.in1_write(in1_write),
		
		.out0_data(out0_data),
		.out0_full_n(out0_full_n),
		.out0_write(out0_write),
		.out1_data(out1_data),
		.out1_full_n(out1_full_n),
		.out1_write(out1_write),
		
		
		.ID(ID),
				
		.ap_clk(ap_clk),
		.ap_rst(ap_rst)
	);	
	// ----------------------------------------------------------------------------

	// clocks
	// ----------------------------------------------------------------------------
	always #(AP_CLK_PERIOD/2)
		ap_clk = ~ap_clk;
	// ----------------------------------------------------------------------------

	// signals evolution
	// ----------------------------------------------------------------------------
	initial
	begin
		// feeding flag initialization
		start_feeding = 0;
		
		// network configuration
		ID = 8'd0;
		
	
		// clocks initialization
			ap_clk = 0;
	
		// network signals initialization
				in0_data = 0;
							in0_write  = 1'b0;
				in1_data = 0;
							in1_write  = 1'b0;
				out0_full_n = 1'b1;
				out1_full_n = 1'b1;
	
		// initial reset
				ap_rst = 1;
		#2
				ap_rst = 0;
		#100
				ap_rst = 1;
		#100
	
		// network inputs (output side)
				out0_full_n = 1'b1;
				out1_full_n = 1'b1;
				
		 		// executing top
		 		ID = 8'd1;
	start_feeding = 1;
	while(in0_i != IN0_TOP_SIZE)
		#10;
	while(in1_i != IN1_TOP_SIZE)
		#10;
	start_feeding = 0;
	in0_data = 0;
	in0_write  = 1'b0;
	in0_i = 0;
	in1_data = 0;
	in1_write  = 1'b0;
	in1_i = 0;
	#1000
		 		// executing top1
		 		ID = 8'd2;
	start_feeding = 1;
	while(in0_i != IN0_TOP1_SIZE)
		#10;
	while(in1_i != IN1_TOP1_SIZE)
		#10;
	start_feeding = 0;
	in0_data = 0;
	in0_write  = 1'b0;
	in0_i = 0;
	in1_data = 0;
	in1_write  = 1'b0;
	in1_i = 0;
	#1000
	
		$stop;
	end
	// ----------------------------------------------------------------------------

	// input feeding
	// ----------------------------------------------------------------------------
	always@(*)
		if(start_feeding && ID == 1)
	 			begin
			while(in0_i < IN0_TOP_SIZE)
			begin
				#10
			 			if(in0_full_n == 1)
			 			begin
							in0_data = in0_top_file_data[in0_i];
							in0_write  = 1'b1;
					in0_i = in0_i + 1;
				end
				else
				begin
							in0_data = 0;
							in0_write  = 1'b0;
				end
			end
			#10
					in0_data = 0;
					in0_write  = 1'b0;
				end
	always@(*)
		if(start_feeding && ID == 1)
	 			begin
			while(in1_i < IN1_TOP_SIZE)
			begin
				#10
			 			if(in1_full_n == 1)
			 			begin
							in1_data = in1_top_file_data[in1_i];
							in1_write  = 1'b1;
					in1_i = in1_i + 1;
				end
				else
				begin
							in1_data = 0;
							in1_write  = 1'b0;
				end
			end
			#10
					in1_data = 0;
					in1_write  = 1'b0;
				end
	always@(*)
		if(start_feeding && ID == 2)
	 			begin
			while(in0_i < IN0_TOP1_SIZE)
			begin
				#10
			 			if(in0_full_n == 1)
			 			begin
							in0_data = in0_top1_file_data[in0_i];
							in0_write  = 1'b1;
					in0_i = in0_i + 1;
				end
				else
				begin
							in0_data = 0;
							in0_write  = 1'b0;
				end
			end
			#10
					in0_data = 0;
					in0_write  = 1'b0;
				end
	always@(*)
		if(start_feeding && ID == 2)
	 			begin
			while(in1_i < IN1_TOP1_SIZE)
			begin
				#10
			 			if(in1_full_n == 1)
			 			begin
							in1_data = in1_top1_file_data[in1_i];
							in1_write  = 1'b1;
					in1_i = in1_i + 1;
				end
				else
				begin
							in1_data = 0;
							in1_write  = 1'b0;
				end
			end
			#10
					in1_data = 0;
					in1_write  = 1'b0;
				end
	// ----------------------------------------------------------------------------

	// output check
	// ----------------------------------------------------------------------------
	always@(posedge ap_clk)
				if(ID == 1)
					begin
					if(out0_write == 1)
						begin	
						if(out0_data != out0_top_file_data[out0_i])
							$display("Error for config %d on output %d: obtained %d, expected %d", 1, out0_i, out0_data, out0_top_file_data[out0_i]);
						out0_i = out0_i + 1;
						end
									if(out0_i == OUT0_TOP_SIZE)
						out0_i = 0;
					end
	always@(posedge ap_clk)
				if(ID == 1)
					begin
					if(out1_write == 1)
						begin	
						if(out1_data != out1_top_file_data[out1_i])
							$display("Error for config %d on output %d: obtained %d, expected %d", 1, out1_i, out1_data, out1_top_file_data[out1_i]);
						out1_i = out1_i + 1;
						end
									if(out1_i == OUT1_TOP_SIZE)
						out1_i = 0;
					end
	always@(posedge ap_clk)
				if(ID == 2)
					begin
					if(out0_write == 1)
						begin	
						if(out0_data != out0_top1_file_data[out0_i])
							$display("Error for config %d on output %d: obtained %d, expected %d", 2, out0_i, out0_data, out0_top1_file_data[out0_i]);
						out0_i = out0_i + 1;
						end
									if(out0_i == OUT0_TOP1_SIZE)
						out0_i = 0;
					end
	always@(posedge ap_clk)
				if(ID == 2)
					begin
					if(out1_write == 1)
						begin	
						if(out1_data != out1_top1_file_data[out1_i])
							$display("Error for config %d on output %d: obtained %d, expected %d", 2, out1_i, out1_data, out1_top1_file_data[out1_i]);
						out1_i = out1_i + 1;
						end
									if(out1_i == OUT1_TOP1_SIZE)
						out1_i = 0;
					end
	// ----------------------------------------------------------------------------

endmodule
