#pragma once

#include "Core.h"

template<typename T, uint32 R, uint32 C>
class Matrix
{
private:
    union {
        T m[R][C];
        T v[R * C];
    };

public:
    Matrix() {}
    Matrix(const T* const f)
    {
        memcpy(&v, f, C * R * sizeof(T));
    }

    Matrix(const T f)
    {
        memset(&v, f, C * R * sizeof(T));
    }

    Matrix transpose()
    {
        Matrix a;
        for (uint32 i = 0; i < R; i++)
        {
            for (uint32 j = 0; j < C; j++)
            {
                a[j][i] = m[i][j];
            }
        }
        return a;
    }

    template<uint32 R2, uint32 C2 = 0>
    const Matrix<T, min(R, R2), min(C, C2)> operator&(Matrix<T, R2, C2>& b)const
    {
        Matrix<T, min(R, R2), min(C, C2)> a;
        for (uint32 i = 0; i < R; i++)
        {
            bool value = true;
            for (uint32 j = 0; j < C; j++)
            {
                value &= m[i][j] | b[i][0];
            }
            a[i][0] = value;
        }
        return a;
    }

    template<uint32 R2, uint32 C2 = 0>
    const Matrix<T, min(R, R2), min(C, C2)> operator|(Matrix<T, R2, C2>& b)const
    {
        Matrix<T, min(R, R2), min(C, C2)> a = T(0);
        for (uint32 i = 0; i < R2; i++)
        {
            if (b[i][0] == false)
                continue;
            for (uint32 j = 0; j < C; j++)
            {
                a[j][0] |= m[i][j];
            }
        }
        return a;
    }

    Matrix& operator~()
    {
        for (uint32 i = 0; i < R; i++)
        {
            for (uint32 j = 0; j < C; j++)
            {
                m[i][j] = !m[i][j];
            }
        }
        return *this;
    }

    T* const operator[](uint32 r)
    {
        return m[r];
    }

    void show()
    {
        for (uint32 i = 0; i < R; i++)
        {
            for (uint32 j = 0; j < C; j++)
            {
                cout << m[i][j] << " ";
            }
            cout << endl;
        }
    }

};