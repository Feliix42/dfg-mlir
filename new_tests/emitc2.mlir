module {
    
    func.func @main() {
        %cst = "emitc.constant"() {value = dense<[1,2,3]> : tensor<3xi32>} : () -> tensor<3xi32> 
        return
    }

}
