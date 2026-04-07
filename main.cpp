#include "tensor.h"
#include <iostream>
#include <ctime>

using namespace std;

int main() {
    srand(time(NULL));
    cout << "--- Red Neuronal Tensor++ (Thiago Frias) ---" << endl;

    // Pasos de la Tabla 1 del PDF
    Tensor entrada = Tensor::random({1000, 20, 20}, -1.0, 1.0);
    Tensor reestructurado = entrada.view({1000, 400});

    Tensor Pesos1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor sesgo1 = Tensor::ones({1, 100});
    ReLU func_relu;
    Tensor capa1 = (matmul(reestructurado, Pesos1) + sesgo1).apply(func_relu);

    Tensor Pesos2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor sesgo2 = Tensor::ones({1, 10});
    Sigmoid func_sigm;
    Tensor salida = (matmul(capa1, Pesos2) + sesgo2).apply(func_sigm);

    cout << "Resultado final: " << salida.obtener_forma()[0] << "x" << salida.obtener_forma()[1] << endl;
    cout << "Proceso completado correctamente." << endl;

    return 0;
}