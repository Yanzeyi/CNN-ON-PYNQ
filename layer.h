#ifndef LAYER_H
#define LAYER_H

#include"stdio.h"
#include"math.h"
#include"iostream"

#define ReLU(x) x > 0 ? x:0

template<int SIZE>
float Conv(float input[SIZE * SIZE], float kernel[SIZE * SIZE])
{
	float temp = 0;
	for (int i = 0; i < SIZE * SIZE; ++i)
	{
		temp += input[i] * kernel[i];
	}
	return temp;
}

template<int SIZE>
float Avg(float input[SIZE * SIZE])
{
	float result = 0;
	for (int i = 0; i < SIZE * SIZE; ++i)
	{
		result += input[i];
	}
	result = result / 4;
	return result;
}

template<int NUM>
void Softmax(float input[], float* output)
{
	float temp = 0;
	for (int i = 0; i < NUM; i++)
	{
		temp += expf(input[i]);
	}
	for (int i = 0; i < NUM; i++)
	{
		output[i] = expf(input[i]) / temp;
	}
}

template<int INFMWID, int INFMHEI, int CHANNEL, int KERNELSIZE, int FILTERNUM, int OUTFMHEI, int OUTFMWID>
void Conv_Layer(float input[], float* output, float weight[], float bias[])
{
	int cnt = 0;
	int KERNELSqrt = KERNELSIZE * KERNELSIZE;
	for (int num = 0; num < FILTERNUM; ++num)
	{
		for (int rows = 0; rows < OUTFMHEI; ++rows)
		{
			for (int cols = 0; cols < OUTFMWID; ++cols)
			{
				float temp[KERNELSIZE * KERNELSIZE] = { 0 };
				float kernel[KERNELSIZE * KERNELSIZE]{ 0 };
				float result = 0;
				for (int channel = 0; channel < CHANNEL; ++channel)
				{
					for (int i = 0; i < KERNELSIZE; ++i)
						for (int j = 0; j < KERNELSIZE; ++j)
						{
							kernel[i * KERNELSIZE+ j] = weight[num * CHANNEL * KERNELSqrt + KERNELSqrt * channel + j + i * KERNELSIZE];
							std::cout << kernel[i * KERNELSIZE + j] << " ";
						}
					printf("\n");
					for (int winy = 0; winy < KERNELSIZE; ++winy)
					{
						for (int winx = 0; winx < KERNELSIZE; ++winx){

							temp[winy * KERNELSIZE + winx] = input[INFMHEI * INFMWID * channel + (rows + winy) * INFMWID + winx + cols];
							std::cout << temp[winy * KERNELSIZE + winx] << " ";
						}
					}
					
					result += Conv<KERNELSIZE>(kernel, temp);
					std::cout << "½á¹ûÊÇ" << result << " ";
					printf("\n");
				}
				output[cnt] = result + bias[num];
				cnt++;
			}
		}
	}
}

template<int INFMWID, int INFMHEI, int CHANNEL, int POOLSIZE>
void AvgPool_Layer(float input[], float* output)
{
	int cnt = 0;
	for (int channel = 0; channel < CHANNEL; ++channel)
	{
		for (int rows = 0; rows < INFMHEI; rows += POOLSIZE)
		{
			for (int cols = 0; cols < INFMWID; cols += POOLSIZE)
			{
				float temp[POOLSIZE * POOLSIZE] = { 0 };
				for (int i = 0; i < POOLSIZE; ++i)
				{
					for (int j = 0; j < POOLSIZE; ++j)
					{
						temp[i * POOLSIZE + j] = input[(rows + i) * INFMHEI + j + cols + channel * INFMWID * INFMHEI];
					}
				}
				output[cnt] = Avg<2>(temp);
				cnt++;
			}
		}
	}
}

template<int IN, int OUT>
void Fullconnect_Layer(float input[], float* output, float weight[], float bias[])
{
	for (int i = 0; i < OUT; ++i)
	{
		float result = 0;
		for (int j = 0; j < IN; ++j)
		{
			result += input[j] * weight[j + i * IN];
		}
		output[i] = ReLU(result + bias[i]);
	}
}
template<int IN, int OUT>
void Out_Layer(float input[], float* output, float weight[], float bias[])
{
	for (int i = 0; i < OUT; ++i)
	{
		float result = 0;
		for (int j = 0; j < IN; ++j)
		{
			result += input[j] * weight[j + i * IN];
		}
		output[i] = result + bias[i];
	}
}

template<int IFMWID, int IFMHEI, int CHANNEL>
void Flatten(float input[], float output[])
{
	int cnt = 0;
	for (int i = 0; i < IFMHEI; i++)
	{
		for (int j = 0; j < IFMWID; j++)
		{
			for (int channel = 0; channel < CHANNEL; channel++)
			{
				output[cnt] = input[IFMHEI * IFMWID * channel + i * IFMWID + j];
				cnt++;
			}
		}
	}
}
#endif 