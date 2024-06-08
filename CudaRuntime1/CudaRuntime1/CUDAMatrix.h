#pragma once

#include "Core.h"

template<typename T>
class Matrix;

typedef Matrix<bool> BoolMatrix;

void MatNegate(BoolMatrix& A);
BoolMatrix MatOr(const BoolMatrix& A, const BoolMatrix& B);
BoolMatrix MatAnd(const BoolMatrix& A, const BoolMatrix& B);
BoolMatrix MatAndProduct(const BoolMatrix& A, const BoolMatrix& B);
BoolMatrix MatOrProduct(const BoolMatrix A, const BoolMatrix B);

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
        cudaFree(m);
        cudaMalloc(&m, rows * columns * sizeof(*m));
        cudaMemcpy(m, other.m, rows * columns * sizeof(*m), cudaMemcpyDeviceToDevice);
        return *this;
    }

    Matrix transpose()
    {
        Matrix a(rows, columns);
        for (uint32 i = 0; i < rows; i++)
        {
            for (uint32 j = 0; j < columns; j++)
            {
                a(j, i) = (*this)(i, j);
            }
        }
        return a;
    }

    const Matrix orProduct(Matrix& b)const
    {
        return MatOrProduct(*this, b);
    }

    const Matrix andProduct(Matrix& b)const
    {
        return MatAndProduct(*this, b);
    }

    T* getElements()const { return m; }
    uint32 getRows()const { return rows; }
    uint32 getColumns()const { return columns; }

    const Matrix operator&(Matrix& b)const
    {
        return MatAnd(*this, b);
    }

    const Matrix operator|(Matrix& b)const
    {
        return MatOr(*this, b);
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
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                if ((*this)(i, j) != b(i, j))
                    return false;
            }
        }
        return true;
    }

    T const operator[](uint32 r)const
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
        for (uint32 i = 0; i < rows; i++)
        {
            for (uint32 j = 0; j < columns; j++)
            {
                s += to_string((*this)(i, j)) + " ";
            }
            s += '\n';
        }
        return s;
    }

protected:
    T* m;
    uint32 rows;
    uint32 columns;
};

string parseRules(BoolMatrix& rhs, BoolMatrix& lhs);
string parseRHS(BoolMatrix& m);
string parseLHS(BoolMatrix& m);
string parseConclusions(BoolMatrix& m);
void computeAllSetsOfConclusions(BoolMatrix* rhs, BoolMatrix* lhs, FILE* out = stdout);
void generateRulesMatrixWithMaxSteps(uint32 i);