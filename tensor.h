#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>

using namespace std;

class Tensor;

// Interfaz para transformaciones
class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

class Tensor {
private:
    vector<size_t> dimensiones_;
    double* puntero_datos_;
    size_t total_elementos_;

public:
    Tensor(const vector<size_t>& forma, const vector<double>& valores);
    Tensor();

    // Regla de los 5 (Gestion de memoria)
    Tensor(const Tensor& otro);
    Tensor(Tensor&& otro) noexcept;
    Tensor& operator=(const Tensor& otro);
    Tensor& operator=(Tensor&& otro) noexcept;
    ~Tensor();

    static Tensor zeros(const vector<size_t>& forma);
    static Tensor ones(const vector<size_t>& forma);
    static Tensor random(const vector<size_t>& forma, double minimo, double maximo);
    static Tensor arange(double inicio, double fin);

    Tensor apply(const TensorTransform& transformacion) const;

    Tensor operator+(const Tensor& otro) const;
    Tensor operator-(const Tensor& otro) const;
    Tensor operator*(const Tensor& otro) const;
    Tensor operator*(double escalar) const;

    Tensor view(const vector<size_t>& nueva_forma);
    Tensor unsqueeze(size_t dim_pos) const;

    static Tensor concat(const vector<Tensor>& tensores, size_t eje);

    friend Tensor dot(const Tensor& t1, const Tensor& t2);
    friend Tensor matmul(const Tensor& m1, const Tensor& m2);

    vector<size_t> obtener_forma() const { return dimensiones_; }
    size_t obtener_tamano() const { return total_elementos_; }
    double* obtener_datos() const { return puntero_datos_; }
};

#endif