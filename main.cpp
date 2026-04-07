#include "tensor.h"
#include <iostream>
#include <ctime>

using namespace std;

int main() {
    srand(time(NULL));
    cout << "INICIANDO PROCESAMIENTO TENSOR++" << endl << endl;

    // 1. Tensor de entrada (datos crudos) 1000x20x20
    Tensor entrada = Tensor::random({1000, 20, 20}, -1.0, 1.0);
    entrada.mostrar("Paso 1: Entrada original");

    // 2. view 1000x400
    Tensor reestructurado = entrada.view({1000, 400});
    reestructurado.mostrar("Paso 2: Despues de view");

    // 3. matmul con pesos W1
    Tensor Pesos1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor z1_matmul = matmul(reestructurado, Pesos1);
    z1_matmul.mostrar("Paso 3: Resultado matmul W1");

    // 4. Suma con bias b1 (1x100)
    Tensor sesgo1 = Tensor::ones({1, 100});
    Tensor z1 = z1_matmul + sesgo1;
    z1.mostrar("Paso 4: Suma con bias b1");

    // 5. Activacion ReLU
    ReLU func_relu;
    Tensor a1 = z1.apply(func_relu);
    a1.mostrar("Paso 5: Activacion ReLU");

    // 6. matmul con pesos W2
    Tensor Pesos2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor z2_matmul = matmul(a1, Pesos2);
    z2_matmul.mostrar("Paso 6: Resultado matmul W2");

    // 7. Suma con bias b2 (1x10)
    Tensor sesgo2 = Tensor::ones({1, 10});
    Tensor z2 = z2_matmul + sesgo2;
    z2.mostrar("Paso 7: Suma con bias b2");

    // 8. Activacion Sigmoid
    Sigmoid func_sigm;
    Tensor salida = z2.apply(func_sigm);
    salida.mostrar("Paso 8: Salida final (Sigmoid)");

    cout << "PROCESO COMPLETADO EXITOSAMENTE" << endl;

    return 0;
}