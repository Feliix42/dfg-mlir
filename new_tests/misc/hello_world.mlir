module {
	memref.global "private" constant @string : memref<14xi8> = dense<[0x48,0x65,0x6c,0x6c,0x6f,0x2c,0x20,0x57,0x6f,0x72,0x6c,0x64,0x21,0]>
	llvm.func external @puts(!llvm.ptr<i8>) -> ()
	func.func @main() -> i64 {
		%c0_i64 = arith.constant 0 : i64
		%0 = memref.get_global @string : memref<14xi8>
		%1 = memref.extract_aligned_pointer_as_index %0 : memref<14xi8> -> index
		%2 = arith.index_cast %1 : index to i64
		%3 = llvm.inttoptr %2 : i64 to !llvm.ptr<i8>
		llvm.call @puts(%3) : (!llvm.ptr<i8>) -> ()
		return %c0_i64 : i64
	}
}
