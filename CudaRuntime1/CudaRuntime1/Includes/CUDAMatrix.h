#pragma once

#include "Core.h"

template<typename T>
class Matrix;

typedef Matrix<bool> BoolMatrix;

bool MatIsEqual(const BoolMatrix& A, const BoolMatrix& B);
void MatNegate(BoolMatrix& A);
void MatOr(const BoolMatrix& A, const BoolMatrix& B, BoolMatrix& C);
BoolMatrix MatAnd(const BoolMatrix& A, const BoolMatrix& B);
void MatAndProduct(const BoolMatrix& A, const BoolMatrix& B, BoolMatrix& C);
void MatOrProduct(const BoolMatrix& A, const BoolMatrix& B, BoolMatrix& C);

template<typename T>
class Matrix
{
public:
    Matrix(uint32 rows, uint32 columns)
    {
        this->rows = rows;
        this->columns = columns;
        cudaMalloc(&m, rows * columns * sizeof(*m));
    }

    Matrix(uint32 rows, uint32 columns, const T* const f) : Matrix(rows, columns)
    {
        cudaMemcpy(m, f, rows * columns * sizeof(*m), cudaMemcpyHostToDevice);
    }

    Matrix(uint32 rows, uint32 columns, const T f) : Matrix(rows, columns)
    {
        cudaMemset(m, f, rows * columns * sizeof(*m));
    }

    Matrix(const Matrix& other) : rows(other.rows), columns(other.columns), m(new T[other.rows * other.columns])
    {
        cudaMalloc(&m, rows * columns * sizeof(*m));
        cudaMemcpy(m, other.m, rows * columns * sizeof(*m), cudaMemcpyDeviceToDevice);
    }

    ~Matrix()
    {
        cudaFree(m);
    }

    Matrix& operator=(const Matrix& other)
    {
        rows = other.rows;
        columns = other.columns;
        cudaMemcpy(m, other.m, rows * columns * sizeof(*m), cudaMemcpyDeviceToDevice);
        return *this;
    }

    Matrix transpose()
    {
        Matrix a(rows, columns);
        for (ufast16 i = 0; i < rows; i++)
        {
            for (ufast16 j = 0; j < columns; j++)
            {
                a(j, i) = (*this)(i, j);
            }
        }
        return a;
    }

    void orProduct(const Matrix& b, Matrix& c)const
    {
        MatOrProduct(*this, b, c);
    }

    void andProduct(const Matrix& b, Matrix& c)const
    {
        MatAndProduct(*this, b, c);
    }

    void copyAsync(const Matrix& other, cudaStream_t& stream)
    {
        rows = other.rows;
        columns = other.columns;
        cudaMemcpyAsync(m, other.m, rows * columns * sizeof(*m), cudaMemcpyDeviceToDevice, stream);
    }

    T* getElements()const { return m; }
    ufast16 getRows()const { return rows; }
    ufast16 getColumns()const { return columns; }

    const Matrix operator&(Matrix& b)const
    {
        return MatAnd(*this, b);
    }

    void or(const Matrix& a, const Matrix& b)
    {
        MatOr(a, b, *this);
    }

    Matrix& operator~()
    {
        MatNegate(*this);
        return *this;
    }

    bool operator==(const Matrix& b)const
    {
        if (rows != b.rows) return false;
        if (columns != b.columns) return false;
        return MatIsEqual(*this, b);
    }

    T const operator[](ufast16 r)const
    {
        T value;
        cudaMemcpy(&value, &m[r], sizeof(*m), cudaMemcpyDeviceToHost);
        return value;
    }

    T const operator()(uint32 r, uint32 c)const
    {
        T value;
        cudaMemcpy(&value, &m[r * columns + c], sizeof(*m), cudaMemcpyDeviceToHost);
        return value;
    }

    T operator()(uint32 r, uint32 c)
    {
        T value;
        cudaMemcpy(&value, &m[r * columns + c], sizeof(*m), cudaMemcpyDeviceToHost);
        return value;

    }

    string show()const
    {
        string s = "";
        for (ufast16 i = 0; i < rows; i++)
        {
            for (ufast16 j = 0; j < columns; j++)
            {
                s += to_string((*this)(i, j)) + " ";
            }
            s += '\n';
        }
        return s;
    }

protected:
    T* m;
    ufast16 rows;
    ufast16 columns;
};

string parseRules(BoolMatrix& rhs, BoolMatrix& lhs);
string parseRules(bool* rhs, bool* lhs, uint32 r, uint32 c);
string parseRHS(BoolMatrix& m);
string parseLHS(BoolMatrix& m);
string parseConclusions(BoolMatrix& m);
void computeAllSetsOfConclusions(BoolMatrix* rhs, BoolMatrix* lhs, FILE* out = stdout);
void computeLastSetOfConclusions(BoolMatrix* rhs, BoolMatrix* lhs, FILE* out = stdout);
void generateRulesMatrixWithMaxSteps(ufast16 i);