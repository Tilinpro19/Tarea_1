#include "tensor.h"
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

Tensor::Tensor(const vector<size_t>& forma, const vector<double>& valores) {
    dimensiones_ = forma;
    total_elementos_ = 1;
    for(size_t i = 0; i < dimensiones_.size(); i++) total_elementos_ *= dimensiones_[i];

    puntero_datos_ = new double[total_elementos_];
    for(size_t i = 0; i < total_elementos_; i++) puntero_datos_[i] = valores[i];
}

Tensor::Tensor() : puntero_datos_(nullptr), total_elementos_(0) {
    dimensiones_ = {0};
}

// Implementacion de copia y movimiento
Tensor::Tensor(const Tensor& otro) {
    dimensiones_ = otro.dimensiones_;
    total_elementos_ = otro.total_elementos_;
    puntero_datos_ = new double[total_elementos_];
    for(size_t i=0; i<total_elementos_; i++) puntero_datos_[i] = otro.puntero_datos_[i];
}

Tensor::Tensor(Tensor&& otro) noexcept {
    dimensiones_ = otro.dimensiones_;
    total_elementos_ = otro.total_elementos_;
    puntero_datos_ = otro.puntero_datos_;
    otro.puntero_datos_ = nullptr;
    otro.total_elementos_ = 0;
}

Tensor& Tensor::operator=(const Tensor& otro) {
    if (this != &otro) {
        delete[] puntero_datos_;
        dimensiones_ = otro.dimensiones_;
        total_elementos_ = otro.total_elementos_;
        puntero_datos_ = new double[total_elementos_];
        for(size_t i=0; i<total_elementos_; i++) puntero_datos_[i] = otro.puntero_datos_[i];
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& otro) noexcept {
    if (this != &otro) {
        delete[] puntero_datos_;
        dimensiones_ = otro.dimensiones_;
        total_elementos_ = otro.total_elementos_;
        puntero_datos_ = otro.puntero_datos_;
        otro.puntero_datos_ = nullptr;
        otro.total_elementos_ = 0;
    }
    return *this;
}

Tensor::~Tensor() {
    delete[] puntero_datos_;
}

// Metodos de creacion
Tensor Tensor::zeros(const vector<size_t>& forma) {
    size_t t = 1;
    for(auto d : forma) t *= d;
    return Tensor(forma, vector<double>(t, 0.0));
}

Tensor Tensor::ones(const vector<size_t>& forma) {
    size_t t = 1;
    for(auto d : forma) t *= d;
    return Tensor(forma, vector<double>(t, 1.0));
}

Tensor Tensor::random(const vector<size_t>& forma, double minimo, double maximo) {
    size_t t = 1;
    for(auto d : forma) t *= d;
    vector<double> v;
    for(size_t i=0; i<t; i++) {
        v.push_back(minimo + ((double)rand() / RAND_MAX) * (maximo - minimo));
    }
    return Tensor(forma, v);
}

Tensor Tensor::arange(double inicio, double fin) {
    vector<double> v;
    for (double i = inicio; i < fin; i++) v.push_back(i);
    return Tensor({v.size()}, v);
}

// Activaciones
Tensor ReLU::apply(const Tensor& t) const {
    Tensor res = t;
    for(size_t i=0; i<res.obtener_tamano(); i++) {
        if(res.obtener_datos()[i] < 0) res.obtener_datos()[i] = 0;
    }
    return res;
}

Tensor Sigmoid::apply(const Tensor& t) const {
    Tensor res = t;
    for(size_t i=0; i<res.obtener_tamano(); i++) {
        res.obtener_datos()[i] = 1.0 / (1.0 + exp(-res.obtener_datos()[i]));
    }
    return res;
}

// Algebra
Tensor Tensor::operator+(const Tensor& otro) const {
    if (this->dimensiones_ == otro.dimensiones_) {
        Tensor res = *this;
        for(size_t i=0; i<total_elementos_; i++) res.puntero_datos_[i] += otro.puntero_datos_[i];
        return res;
    }
    // Suma de bias (especial para la red del PDF)
    if (this->dimensiones_.size() == 2 && otro.dimensiones_.size() == 2 && otro.dimensiones_[0] == 1) {
        Tensor res = *this;
        for(size_t i=0; i<dimensiones_[0]; i++) {
            for(size_t j=0; j<dimensiones_[1]; j++)
                res.puntero_datos_[i*dimensiones_[1]+j] += otro.puntero_datos_[j];
        }
        return res;
    }
    return *this;
}

Tensor matmul(const Tensor& m1, const Tensor& m2) {
    size_t f = m1.obtener_forma()[0], c = m2.obtener_forma()[1], k_lim = m1.obtener_forma()[1];
    vector<double> v(f * c, 0.0);
    for(size_t i=0; i<f; i++) {
        for(size_t j=0; j<c; j++) {
            for(size_t k=0; k<k_lim; k++)
                v[i*c+j] += m1.obtener_datos()[i*k_lim+k] * m2.obtener_datos()[k*c+j];
        }
    }
    return move(Tensor({f, c}, v));
}

Tensor Tensor::view(const vector<size_t>& nueva_forma) {
    Tensor res;
    res.dimensiones_ = nueva_forma;
    res.total_elementos_ = this->total_elementos_;
    res.puntero_datos_ = this->puntero_datos_;
    this->puntero_datos_ = nullptr; // Move semantic exigido
    return move(res);
}

Tensor Tensor::apply(const TensorTransform& transformacion) const {
    return transformacion.apply(*this);
}
// ... (manten todo el codigo anterior igual y añade esto al final)

void Tensor::mostrar(string nombre_paso) const {
    cout << "=== " << nombre_paso << " ===" << endl;
    cout << "Forma: [";
    for(size_t i = 0; i < dimensiones_.size(); i++) {
        cout << dimensiones_[i] << (i == dimensiones_.size() - 1 ? "" : ", ");
    }
    cout << "]" << endl;

    cout << "Primeros valores: ";
    size_t limite = (total_elementos_ < 5) ? total_elementos_ : 5;
    for(size_t i = 0; i < limite; i++) {
        cout << puntero_datos_[i] << "  ";
    }
    cout << "..." << endl << endl;
}