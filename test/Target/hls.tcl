set TARGET_DEVICE "xck26-sfvc784-2LV-c"
set CLOCK_PERIOD 10
set PROJECT_DIR "./hls_project"
set cpp_files [glob -nocomplain "./*.cpp"]
if {[llength $cpp_files] == 0} {
  puts "ERROR: No .cpp file found in the directory!"
  exit 1
}
set SOURCE_FILE [lindex $cpp_files 0]
puts "INFO: Using source file: $SOURCE_FILE"
open_project $PROJECT_DIR
add_files $SOURCE_FILE
set FUNCTION_NAMES {foo mac}
foreach func $FUNCTION_NAMES {
  open_solution "solution_$func"
  set_part $TARGET_DEVICE
  create_clock -period $CLOCK_PERIOD -name default
  set_top $func
  csynth_design
  close_solution
}
close_project
exit
