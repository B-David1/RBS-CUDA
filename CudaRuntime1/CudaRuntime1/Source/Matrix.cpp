#include "Core.h"

#ifdef USE_CUDA
	#include "..\Includes\CUDAMatrix.h"
#else
	#include "..\Includes\Matrix.h"
#endif

inline string getVar(ufast16 index)
{
	static const char ans[] = "abcdefghijklmnopqrstuvwxyz";
	string s = "";
	ufast16 total = index + 1;
	do
	{
		total--;
		s.insert(0, 1, ans[total % (sizeof(ans)-1)]);
	} while (total /= (sizeof(ans)-1));
	return s;
}

string parseRules(BoolMatrix& rhs, BoolMatrix& lhs)
{
	ufast16 r = rhs.getRows();
	ufast16 c = rhs.getColumns();
	string s = "";
	for (ufast16 i = 0; i < r; i++)
	{
		s += "r" + to_string(i + 1) + ":";
		string srhs = "";
		string slhs = " ";
		bool isNull = true;
		for (ufast16 j = 0; j < c; j++)
		{
			if (rhs(i, j) == true)
			{
				srhs = srhs + getVar(j) + " & ";
				isNull = false;
			}
			if (lhs(j, i) == true)
				slhs = slhs + getVar(j) + ",";
		}
		if (!isNull)
		{
			srhs = " <- " + srhs;
			srhs.pop_back();
			srhs.pop_back();
		}
		slhs.pop_back();
		s += slhs + srhs + '\n';
	}
	return s;
}

string parseRules(bool* rhs, bool* lhs, uint32 r, uint32 c)
{
	string s = "";
	for (ufast16 i = 0; i < r; i++)
	{
		s += "r" + to_string(i + 1) + ":";
		string srhs = "";
		string slhs = " ";
		bool isNull = true;
		for (ufast16 j = 0; j < c; j++)
		{
			if (rhs[i * r + j] == true)
			{
				srhs = srhs + getVar(j) + " & ";
				isNull = false;
			}
			if (lhs[j * r + i] == true)
				slhs = slhs + getVar(j) + ",";
		}
		if (!isNull)
		{
			srhs = " <- " + srhs;
			srhs.pop_back();
			srhs.pop_back();
		}
		slhs.pop_back();
		s += slhs + srhs + '\n';
	}
	return s;
}

string parseRHS(const BoolMatrix& m)
{
	ufast16 r = m.getRows();
	ufast16 c = m.getColumns();
	string s = "";
	for (ufast16 i = 0; i < r; i++)
	{
		s += "rhs(r" + to_string(i + 1) + "): {";
		bool isNull = true;
		for (ufast16 j = 0; j < c; j++)
		{
			if (m(i, j) == true)
			{
				s = s + getVar(j) + ",";
				isNull = false;
			}
		}
		if (!isNull)
			s.pop_back();
		s += "}\n";
	}
	return s;
}

string parseLHS(const BoolMatrix& m)
{
	ufast16 r = m.getRows();
	ufast16 c = m.getColumns();
	string s = "";
	for (ufast16 i = 0; i < r; i++)
	{
		s += "lhs(r" + to_string(i + 1) + "): ";
		for (ufast16 j = 0; j < c; j++)
		{
			if (m(j, i) == true)
			{
				s = s + getVar(j) + " ";
			}
		}
		s += '\n';
	}
	return s;
}

string parseConclusions(BoolMatrix& m)
{
	ufast16 r = m.getRows();
	string s = "{";
	bool isNull = true;
	for (ufast16 i = 0; i < r; i++)
	{
		if (m(i, 0) == true)
		{
			s = s + getVar(i) + ',';
			isNull = false;
		}
	}
	if (!isNull)
		s.pop_back();
	return s += "}";
}

bool* getElementsAsync(const BoolMatrix& m, cudaStream_t& stream)
{
	ufast16 r = m.getRows();
	bool* v;
	cudaHostAlloc(&v, r * sizeof(bool), cudaHostAllocDefault);
	cudaMemcpyAsync(v, m.getElements(), r * sizeof(bool), cudaMemcpyDeviceToHost);
}

string parseConclusionsCUDA(const BoolMatrix& m)
{
	ufast16 r = m.getRows();
	string s = "{";
	bool* v = (bool*)malloc(r * sizeof(bool));
	cudaMemcpy(v, m.getElements(), r * sizeof(bool), cudaMemcpyDeviceToHost);
	bool isNull = true;
	for (ufast16 i = 0; i < r; i++)
	{
		if (v[i] == true)
		{
			s = s + getVar(i) + ',';
			isNull = false;
		}
	}
	if (!isNull)
		s.pop_back();
	free(v);
	return s += "}";
}

string parseConclusionsCUDAAsync(bool* v, ufast16 r)
{
	string s = "{";
	bool isNull = true;
	for (ufast16 i = 0; i < r; i++)
	{
		if (v[i] == true)
		{
			s = s + getVar(i) + ',';
			isNull = false;
		}
	}
	if (!isNull)
		s.pop_back();
	return s += "}";
}

#ifdef USE_CUDA
void computeAllSetsOfConclusions(BoolMatrix* rhs, BoolMatrix* lhs, FILE* out)
{
	ufast16 rows = rhs->getRows();

	BoolMatrix cP(rows, 1, false);

	string conclusions = "Step 0: " + parseConclusionsCUDA(cP) + '\n';
	fwrite(conclusions.c_str(), sizeof(char), conclusions.size(), out);

	bool* cArr;
	cudaHostAlloc(&cArr, rows * sizeof(bool), cudaHostAllocDefault);
	cudaStream_t printStream;
	cudaStreamCreate(&printStream);

	BoolMatrix r(rows, 1);
	BoolMatrix p(rows, 1);
	
	rhs->andProduct(cP, r);
	lhs->orProduct(r, p);

	BoolMatrix c(rows, 1);
	cudaStream_t conclusionsStream;
	cudaStreamCreate(&conclusionsStream);

	ufast16 i = 1;
	while (i > 0)
	{
		cudaError_t err = cudaStreamSynchronize(printStream);
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		c.or(cP, p);

		conclusions = "Step " + to_string(i) + ": " + parseConclusionsCUDAAsync(cArr, rows) + '\n';
		fwrite(conclusions.c_str(), sizeof(char), conclusions.size(), out);

		cudaMemcpyAsync(cArr, c.getElements(), rows * sizeof(bool), cudaMemcpyDeviceToHost, printStream);

		if (c == cP)
			break;

		cP.copyAsync(c, conclusionsStream);

		++i;

		rhs->andProduct(c, r);
		lhs->orProduct(r, p);

		err = cudaStreamSynchronize(conclusionsStream);
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));
	}
	cudaStreamDestroy(conclusionsStream);
	cudaStreamDestroy(printStream);

	cudaFreeHost(cArr);
}

void computeLastSetOfConclusions(BoolMatrix* rhs, BoolMatrix* lhs, FILE* out)
{
	ufast16 rows = rhs->getRows();

	BoolMatrix cP(rows, 1, false);

	BoolMatrix r(rows, 1);
	BoolMatrix p(rows, 1);

	rhs->andProduct(cP, r);
	lhs->orProduct(r, p);

	BoolMatrix c(rows, 1);
	cudaStream_t conclusionsStream;
	cudaStreamCreate(&conclusionsStream);

	ufast16 i = 1;
	while (i > 0)
	{
		c.or(cP, p);
		
		if (c == cP)
			break;

		cP.copyAsync(c, conclusionsStream);

		++i;

		rhs->andProduct(c, r);
		lhs->orProduct(r, p);

		cudaError_t err = cudaStreamSynchronize(conclusionsStream);
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));
	}
	cudaStreamDestroy(conclusionsStream);

	string conclusions = "Step " + to_string(i) + ": " + parseConclusionsCUDA(c) + '\n';
	fwrite(conclusions.c_str(), sizeof(char), conclusions.size(), out);
}
#else
void computeAllSetsOfConclusions(BoolMatrix* rhs, BoolMatrix* lhs, FILE* out)
{
	ufast16 rows = rhs->getRows();

	BoolMatrix cP(rows, 1, false);

	string conclusions = "Step 0: " + parseConclusions(cP) + '\n';
	fwrite(conclusions.c_str(), sizeof(char), conclusions.size(), out);

	BoolMatrix r = rhs->andProduct(cP);
	BoolMatrix p = lhs->orProduct(r);

	BoolMatrix c(rows, 1);

	ufast16 i = 1;
	while (i > 0)
	{
		c = cP | p;

		conclusions = "Step " + to_string(i++) + ": " + parseConclusions(c) + '\n';
		fwrite(conclusions.c_str(), sizeof(char), conclusions.size(), out);

		if (c == cP)
			break;

		r = rhs->andProduct(c);
		p = lhs->orProduct(r);

		cP = c;
	}
}

void computeLastSetOfConclusions(BoolMatrix* rhs, BoolMatrix* lhs, FILE* out)
{
	ufast16 rows = rhs->getRows();

	BoolMatrix cP(rows, 1, false);
	BoolMatrix r = rhs->andProduct(cP);
	BoolMatrix p = lhs->orProduct(r);

	BoolMatrix c(rows, 1);
	cudaStream_t stream;

	ufast16 i = 1;
	while (i > 0)
	{
		c = cP | p;
		if (c == cP)
			break;

		++i;

		r = rhs->andProduct(c);
		p = lhs->orProduct(r);

		cP = c;
	}
	string conclusions = "Step " + to_string(i) + ": " + parseConclusions(c) + '\n';
	fwrite(conclusions.c_str(), sizeof(char), conclusions.size(), out);
}
#endif

void generateRulesMatrixWithMaxSteps(ufast16 rulesCount)
{
	FILE* flhs = fopen("lhs.txt", "w+");

	string s = "";
	for (ufast16 i = 0; i < rulesCount; i++)
	{
		for (ufast16 j = 0; j < rulesCount; j++)
		{
			if (i == j)
				s += '1';
			else
				s += '0';
		}
		s += '\n';
	}
	s.pop_back();
	fwrite(s.c_str(), sizeof(char), s.size(), flhs);

	FILE* frhs = fopen("rhs.txt", "w+");

	s = "";
	for (ufast16 i = 0; i < rulesCount; i++)
	{
		for (ufast16 j = 0; j < rulesCount; j++)
		{
			if (i == j + 1)
				s += '1';
			else
				s += '0';
		}
		s += '\n';
	}
	s.pop_back();
	fwrite(s.c_str(), sizeof(char), s.size(), frhs);

	fclose(flhs);
	fclose(frhs);
}