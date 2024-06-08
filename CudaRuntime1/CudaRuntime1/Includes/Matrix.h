#pragma once

#include "Core.h"

template<typename T>
class Matrix;

typedef Matrix<bool> BoolMatrix;

template<typename T>
class Matrix
{
public:
    Matrix(uint32 rows, uint32 columns)
    { 
        this->rows = rows;
        this->columns = columns;
        m = new T[rows * columns];
    }

    Matrix(uint32 rows, uint32 columns, const T* const f): Matrix(rows, columns)
    {
        memcpy(m, f, columns * rows * sizeof(T));
    }

    Matrix(uint32 rows, uint32 columns, const T f): Matrix(rows, columns)
    {
        memset(m, f, columns * rows * sizeof(T));
    }

    Matrix(const Matrix& other): rows(other.rows), columns(other.columns), m(new T[other.rows * other.columns])
    {
        std::copy(other.m, other.m + (other.rows * other.columns), m);
    }

    ~Matrix()
    {
        delete[] m;
    }

    Matrix& operator=(const Matrix& other)
    {
        rows = other.rows;
        columns = other.columns;

        delete[] m;

        m = new T[rows * columns];
        std::copy(other.m, other.m + (rows * columns), m);

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
        Matrix a(min(rows, b.rows), min(columns, b.columns), T(0));
        for (uint32 i = 0; i < b.rows; i++)
        {
            bool value = false;
            for (uint32 j = 0; j < columns; j++)
            {
                value |= (*this)(i, j) & b(j, 0);
            }
            a(i, 0) = value;
        }
        return a;
    }

    const Matrix andProduct(Matrix& b)const
    {      
        Matrix a(min(rows, b.rows), min(columns, b.columns));
        for (uint32 i = 0; i < rows; i++)
        {
            bool value = true;
            for (uint32 j = 0; j < columns; j++)
            {
                value &= (*this)(i, j) | b(j, 0);
            }
            a(i, 0) = value;
        }
        return a;
    }

    T* getElements()const { return m; }
    uint32 getRows()const { return rows; }
    uint32 getColumns()const { return columns; }

    const Matrix operator&(Matrix& b)const
    {
        uint32 r = min(rows, b.rows);
        uint32 c = min(columns, b.columns);
        Matrix a(r, c);
        for (uint32 i = 0; i < r; i++)
        {
            for (uint32 j = 0; j < c; j++)
            {
                a(i, j) = (*this)(i, j) & b(i, j);
            }
        }
        return a;
    }

    const Matrix operator|(Matrix& b)const
    {
        uint32 r = min(rows, b.rows);
        uint32 c = min(columns, b.columns);
        Matrix a(r, c);
        for (uint32 i = 0; i < r; i++)
        {
            for (uint32 j = 0; j < c; j++)
            {
                a(i, j) = (*this)(i, j) | b(i, j);
            }
        }
        return a;
    }

    Matrix& operator~()
    {
        for (uint32 i = 0; i < rows; i++)
        {
            for (uint32 j = 0; j < columns; j++)
            {
                (*this)(i, j) = !(*this)(i, j);
            }
        }
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
        return m[r];
    }

    T& const operator()(uint32 r, uint32 c)const
    {
        return m[r * columns + c];
    }

    T& operator()(uint32 r, uint32 c)
    {
        return m[r * columns + c];

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
string parseRules(bool* rhs, bool* lhs, uint32 r, uint32 c);
string parseRHS(BoolMatrix& m);
string parseLHS(BoolMatrix& m);
string parseConclusions(BoolMatrix& m);
void computeAllSetsOfConclusions(BoolMatrix* rhs, BoolMatrix* lhs, FILE* out = stdout);
void computeLastSetOfConclusions(BoolMatrix* rhs, BoolMatrix* lhs, FILE* out = stdout);
void generateRulesMatrixWithMaxSteps(uint32 i);