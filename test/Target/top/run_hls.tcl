set TARGET_DEVICE "xck26-sfvc784-2LV-c"
set CLOCK_PERIOD 10
set PROJECT_DIR "./hls_project"
set SOURCE_FILE "main.cpp"
if {![file exists $SOURCE_FILE]} {
    puts "ERROR: Source file $SOURCE_FILE does not exist!"
    exit 1
}
puts "INFO: Using source file: $SOURCE_FILE"
open_project $PROJECT_DIR
add_files $SOURCE_FILE
puts "INFO: Creating HLS solution"
open_solution "solution_top"
set_part $TARGET_DEVICE
create_clock -period $CLOCK_PERIOD -name default
set_top top
puts "INFO: Set top function to 'top'"
puts "INFO: Running C synthesis..."
if {[catch {csynth_design} result]} {
    puts "ERROR: C synthesis failed: $result"
    exit 1
}
puts "INFO: C synthesis completed successfully"
puts "INFO: Exporting RTL design..."
if {[catch {export_design -rtl verilog} result]} {
    puts "ERROR: Failed to export design: $result"
    exit 1
}
puts "INFO: Design exported successfully"
close_solution
close_project
puts "INFO: HLS completed successfully"
exit
