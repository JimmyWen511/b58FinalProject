module pong(CLOCK_50, VGA_CLK, VGA_HS, VGA_VS, VGA_BLANK_N, VGA_SYNC_N, VGA_R, VGA_G, VGA_B, KEY, SW);
input [3:0] KEY;
input [9:0] SW;
input CLOCK_50;
output VGA_CLK, VGA_HS, VGA_VS, VGA_BLANK_N, VGA_SYNC_N;
output [9:0] VGA_R;
output [9:0] VGA_G;
output [9:0] VGA_B;

reg [7:0] x = 8'b00000010;
reg [6:0] y = 7'b0000010;
reg [2:0] colour = 3'b111;
reg [27:0] q = 28'd0;

vga_adapter VGA(
					 .resetn(SW[9]),
					 .clock(CLOCK_50),
					 .colour(colour),
					 .x(x),
					 .y(y),
					 .plot(1),
					 .VGA_R(VGA_R),
					 .VGA_G(VGA_G),
					 .VGA_B(VGA_B),
					 .VGA_HS(VGA_HS),
					 .VGA_VS(VGA_VS),
					 .VGA_BLANK(VGA_BLANK_N),
					 .VGA_SYNC(VGA_SYNC_N),
					 .VGA_CLK(VGA_CLK));
		defparam VGA.RESOLUTION = "160x120";
		defparam VGA.MONOCHROME = "FALSE";
		defparam VGA.BITS_PER_COLOUR_CHANNEL = 1;
		defparam VGA.BACKGROUND_IMAGE = "black.mif";
			wire [2:0] stateNum;

always @ (posedge CLOCK_50)
begin
	if (SW[0])
	begin
		colour <= 3'b111;
	end
	else if (SW[1])
	begin
		colour <= 3'b100;
	end
	else if (SW[2])
	begin
		colour <= 3'b010;
	end
	else if (SW[3])
	begin
		colour <= 3'b001;
	end
	else if (SW[4])
	begin
		colour <= 3'b000;
	end
end

always @ (posedge CLOCK_50)
begin
	q <= q + 1'd1;
	if (q == 28'd09999999)
	begin
		q <= 0;
		if (~KEY[3] & y > 7'b1)
		begin
			y <= y - 1'b1;
		end
		else if (~KEY[2] & y < 7'b1110111)
		begin
			y <= y + 1'b1;
		end
		else if (~KEY[1] & x > 8'b1)
		begin
			x <= x - 1'b1;
		end
		else if (~KEY[0] & x < 8'b10011111)
		begin
			x <= x + 1'b1;
		end
	end
end
endmodule
