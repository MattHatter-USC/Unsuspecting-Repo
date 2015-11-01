/*


  Copyright (c) 2015, Newcastle University (United Kingdom)
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/

//typedef float* __restrict__  __attribute__((align_value(64))) fptr;

#define parallels false


#include <cilk/cilk.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
//#include <cmath>
#include <getopt.h>
#include "util.hpp"
//#include <tbb/tbb.h>
//#include <algorithm>
//#include <mkl.h> 
#include <omp.h>
#include <cmath>
#include <immintrin.h>
#include <mkl.h>
//#include <cmath>
//used floating point mode relaxed
using namespace std;
//using namespace tbb;
static int quiet = 0;
static bool halfway = false;
static int finalnum;
static int finalnum2;
static int lv1;
static int lv2;
static int lv3;

static float RandomFloatPos() {
    // returns a random number between a given minimum and maximum
    float random = ((float) rand()) / (float) RAND_MAX;
    float a = 0;
    float r = random;
    return a + r;
}

static float getNorm(float currArray[]) {
    // computes L2 norm of input array
    int c;
    float arraySum=0;
    //for (c=0; c<3; c++) {
	arraySum = __sec_reduce_add(currArray[0:3] * currArray[0:3]);
    //arraySum += currArray[c]*currArray[c];
    //}
    float res = sqrt(arraySum);
	
    return res;
}



static float getrNorm(float currArray[]) {
	// computes L2 norm of input array
	int c;
	float arraySum = 0;
	//for (c = 0; c<3; c++) {
	arraySum = __sec_reduce_add(currArray[0:3] * currArray[0:3]);
	//}
	float res = 1/sqrt(arraySum);

	return res;
}


static float getL2Distance(float pos1x, float pos1y, float pos1z, float pos2x, float pos2y, float pos2z) {
    // returns distance (L2 norm) between two positions in 3D
	__attribute__((aligned(64))) float distArray[3];
    distArray[0] = pos2x-pos1x;
    distArray[1] = pos2y-pos1y;
    distArray[2] = pos2z-pos1z;
    float l2Norm = getNorm(distArray);
    return l2Norm;
}


//couldn't find parallel library defines for these :(

//#define min(x,y) (((x) < (y)) ? (x) : (y))
#pragma omp declare simd
__attribute__((vector)) inline float min(float x, float y) {
	return (((x) < (y)) ? (x) : (y));
}

#pragma omp declare simd
__attribute__((vector)) inline int min(int x, int y) {
	return (((x) < (y)) ? (x) : (y));
}

#pragma omp declare simd
__attribute__((vector)) inline float max(float x, float y) {
	return (((x) > (y)) ? (x) : (y));
}

#pragma omp declare simd
__attribute__((vector)) inline int max(int x, int y) {
	return (((x) > (y)) ? (x) : (y));
}


__attribute__((vector)) inline float  getL2DistanceSq(float dx,float dy,float dz) {
	// returns distance (L2 norm) between two positions in 3D
	return dx*dx+dy*dy+dz*dz;
}

static stopwatch produceSubstances_sw;
static stopwatch runDiffusionStep_sw;
static stopwatch runDecayStep_sw;
static stopwatch cellMovementAndDuplication_sw;
static stopwatch runDiffusionClusterStep_sw;
static stopwatch getEnergy_sw;
static stopwatch getCriterion_sw;
static stopwatch phase1_sw;
static stopwatch phase2_sw;
static stopwatch compute_sw;
static stopwatch init_sw;

static void produceSubstances(float* Conc, float* posAll, int* typesAll, int L, int n) {
	produceSubstances_sw.reset();

	// increases the concentration of substances at the location of the cells
	//float sideLength = 1 / (float)L; // length of a side of a diffusion voxel
	float rsideLength = (float)L;
	int div = min(n / 20, 500); //figure this out later
	#pragma omp parallel default(shared) if(parallels)
	{
		int c, c2, e; //i1, i2, i3;
		e = 16;
		__attribute__((aligned(64))) int i1[16];
		__attribute__((aligned(64))) int i2[16];
		__attribute__((aligned(64))) int i3[16];
		#pragma omp for
		for (c = 0; c < n; c += 16) {
			if (n - c < 16) { e = (int)n - c; }
			//for (c = 0; c < n; ++c) {
			#pragma vector nontemporal
			#pragma vector aligned

			i1[0:e] = min((int)(posAll[c:e] * rsideLength), (L - 1));
			i2[0:e] = min((int)(posAll[finalnum+c:e] * rsideLength), (L - 1));
			i3[0:e] = min((int)(posAll[finalnum2+c:e] * rsideLength), (L - 1));
			for (c2 = 0; c2 < e; ++c2) {
				if (typesAll[c+c2] == 1) {
					Conc[i1[c2]*lv2+i2[c2]*lv1+i3[c2]] += 0.1;
					if (Conc[i1[c2]*lv2+i2[c2]*lv1+i3[c2]]> 1) {
						Conc[i1[c2]*lv2+i2[c2]*lv1+i3[c2]] = 1;
					}
				}
				else {
					Conc[lv3+i1[c2]*lv2+i2[c2]*lv1+i3[c2]] += 0.1;
					if (Conc[lv3+i1[c2]*lv2+i2[c2]*lv1+i3[c2]]> 1) {
						Conc[lv3+i1[c2]*lv2+i2[c2]*lv1+i3[c2]] = 1;
					}
				}
			}
		}
	}
    produceSubstances_sw.mark();
}

static void runDiffusionStep(float * Conc, int L, float D) {
    runDiffusionStep_sw.reset();
    // computes the changes in substance concentrations due to diffusion
	#pragma omp parallel default(shared) if (parallels)
	{
		float con = D / 6.0;
		int i1, i2, i3, subInd, e;
		int LM = L - 1;
		int LMM = L - 2;
		__attribute__((aligned(64))) float temp[L];
		#pragma omp for     //                  TRY TO PUT PARALLEL FO{
		for (i1 =0; i1 < L; ++i1) {
			for (i2 = 0; i2 < L; ++i2) {
				int added = 2;
				for (subInd = 0; subInd < 2; subInd++) {
					#pragma vector aligned
					#pragma vector nontemporal
					#pragma ivdep
					temp[0:L] = 0.0;
					if ((i1 + 1) < L) {
						#pragma vector aligned
						#pragma vector nontemporal
						#pragma ivdep
						temp[0:L] += Conc[subInd*lv3+(i1+1)*lv2+i2*lv1:L];
						++added;
					}
					if ((i1 - 1) >= 0) {
						#pragma vector aligned
						#pragma vector nontemporal
						#pragma ivdep
						temp[0:L] += Conc[subInd*lv3+(i1 - 1)*lv2+i2*lv1:L];
						++added;
					}
					if ((i2 + 1) < L) {
						#pragma vector aligned
						#pragma vector nontemporal
						#pragma ivdep
						temp[0:L] += Conc[subInd*lv3+i1*lv2+(i2 + 1)*lv1:L];
						++added;
					}
					if ((i2 - 1) >= 0) {
						#pragma vector aligned
						#pragma vector nontemporal
						#pragma ivdep
						temp[0:L] += Conc[subInd*lv3+i1*lv2+(i2 - 1)*lv1:L];
						++added;
					}
					#pragma vector aligned//just put them everywhere psh
					#pragma vector nontemporal
					#pragma ivdep
					temp[0:LM] += Conc[subInd*lv3+i1*lv2+i2*lv1+1:LM];
					#pragma vector aligned
					#pragma vector nontemporal
					#pragma ivdep
					temp[1:LM] += Conc[subInd*lv3+i1*lv2+i2*lv1:LM];
					#pragma vector aligned
					#pragma vector nontemporal
					#pragma ivdep
					temp[1:LMM] -= added*Conc[subInd*lv3 + i1*lv2 + i2*lv1 + 1:LMM];
					temp[0] -= (added-1)*Conc[subInd*lv3+i1*lv2+i2*lv1];
					temp[LM] -= (added - 1)*Conc[subInd*lv3+i1*lv2+i2*lv1+LM];
					#pragma vector aligned
					#pragma vector nontemporal
					#pragma ivdep
					temp[0:L] *= con;
					#pragma vector aligned
					#pragma vector nontemporal
					#pragma ivdep
					Conc[subInd*lv3+i1*lv2+i2*lv1:L] += temp[0:L]; //sexy

					//lends itself quite well to vectorization
				}
			}
		}
	}
    runDiffusionStep_sw.mark();
}

static void runDecayStep(float* Conc, int L, float mu) {
    runDecayStep_sw.reset();
    // computes the changes in substance concentrations due to decay
	float val = (1 - mu);
	#pragma omp parallel default(shared) if (parallels)
	{
		int i1, i2;
		#pragma omp for
		for (i1 = 0; i1 < L; ++i1) {
			for (i2 = 0; i2 < L; i2++) {
				#pragma vector aligned
				#pragma vector nontemporal
				#pragma ivdep
				Conc[i1*lv2+i2*lv1:L] *= val;
				Conc[lv3+i1*lv2+i2*lv1:L] *= val;
			}
		}
	}
    runDecayStep_sw.mark();
}

static int cellMovementAndDuplication(float* posAll, float* pathTraveled, int* typesAll, int* numberDivisions, float pathThreshold, int divThreshold, int n) {
    cellMovementAndDuplication_sw.reset();
	//int div = min(n / 20, 500); //figure this out later
	int currentNumberCells = n;
	#pragma omp parallel default(shared) if (parallels)
	{
		int i, e;
		__attribute__((aligned(64))) float currenttrNorm[16];
		int newcellnum;
		float currentrNorm2;
		__attribute__((aligned(64))) float currentCellMovement[3*16];
		__attribute__((aligned(64))) float  duplicatedCellOffset[3]; 
		int endv = (n / 16) * 16;
		#pragma omp for
		for (int c = 0; c < n; c += 16) {
			e = min(16, (int)n - c);	 //size of stream
			// random cell movement
			for (i = 0; i < e; ++i) {
				currentCellMovement[i] = RandomFloatPos() - 0.5;
				currentCellMovement[16+i] =  RandomFloatPos() - 0.5;
				currentCellMovement[32 + i] = RandomFloatPos() - 0.5;
			}
			//currentNorm = getNorm(currentCellMovement);
			#pragma vector aligned
			#pragma vector nontemporal
			#pragma ivdep
			currentrNorm[0:e] = 1.0 / sqrtf(currentCellMovement[0:e]* currentCellMovement[0:e] + currentCellMovement[16:e]*currentCellMovement[16:e] + currentCellMovement[32:e]*currentCellMovement[32:e]);
			#pragma vector aligned
			#pragma vector nontemporal
			#pragma ivdep
			posAll[c:e] += 0.1*currentCellMovement[0:e] * currentrNorm[0:e];
			#pragma vector aligned
			#pragma vector nontemporal
			#pragma ivdep
			posAll[finalnum+c:e] += 0.1*currentCellMovement[16:e] * currentrNorm[16:e];
			#pragma vector aligned
			#pragma vector nontemporal
			#pragma ivdep
			posAll[finalnum2+c:e] += 0.1*currentCellMovement[32:e] * currentrNorm[32:e];
			#pragma vector aligned
			#pragma vector nontemporal
			#pragma ivdep
			pathTraveled[c:e] += 0.1;
			for (i = c; i < e; ++i) { //we'll figure this out later
				// cell duplication if conditions fulfilled
				if (numberDivisions[i] < divThreshold) {
					if (pathTraveled[i] > pathThreshold) {
						pathTraveled[i] -= pathThreshold;
						numberDivisions[i] += 1;  // update number of divisions this cell has undergone
						#pragma omp critical 
						{
							newcellnum = currentNumberCells++;   // update number of cells in the simulation (all in one steppp)
						}
						numberDivisions[newcellnum] = numberDivisions[i];   // update number of divisions the duplicated cell has undergone
						typesAll[newcellnum] = -typesAll[i]; // assign type of duplicated cell (opposite to current cell)

						// assign location of duplicated cell
						duplicatedCellOffset[0] = RandomFloatPos() - 0.5;
						duplicatedCellOffset[1] = RandomFloatPos() - 0.5;
						duplicatedCellOffset[2] = RandomFloatPos() - 0.5;
						//currentNorm = getNorm(duplicatedCellOffset);
						currentrNorm2 = 1.0 / getNorm(duplicatedCellOffset);
						#pragma ivdep
						#pragma vector nontemporal
						#pragma vector aligned
						posAll[newcellnum:finalnum2:finalnum] = posAll[i:finalnum2:finalnum];
						#pragma ivdep
						#pragma vector nontemporal
						#pragma vector aligned
						posAll[newcellnum:finalnum2:finalnum] *= 0.05*duplicatedCellOffset[0:3] * currentrNorm2;
					}

				}
			}

		}
	}
    cellMovementAndDuplication_sw.mark();
    return currentNumberCells;
}

static void runDiffusionClusterStep(float* Conc, float* movVec, float* posAll, int* typesAll, int n, int L, float speed) {
	runDiffusionClusterStep_sw.reset();
	// computes movements of all cells based on gradients of the two substances
	float sideLength = 1 / (float)L; // length of a side of a diffusion voxel
	float rsidelength = (float)L;
#pragma omp parallel default(shared)  if(parallels)
	{
		//imma pragma yo momma
		//TODO: dsadsa
		float sideLength = 1 / (float)L; // length of a side of a diffusion voxel
		float rsideLength = (float)L;
		float gradSub1[3];
		float gradSub2[3];

		float normGrad1, normGrad2;
		int i1, i2, i3, xUp, xDown, yUp, yDown, zUp, zDown;

		//for (int c = 0; c < n; ++c)
		int endv = (n / 16) * 16;
		int L2 = L*L;
		int L3 = L*L*L;
		{
			__m512i a, b, c, a0, b0, c0, L1_v, L2_v, L3_v, d, e, f, g, h, i, a_in, b_in, c_in, d_in, e_in, f_in, ;
			__m512 type, t1, t2, t3, Lv, GS10, GS11, GS12, GS20, GS21, GS22, preval1, preval2, norm1, norm2;
			__mmask16 comparemask;
			#pragma omp for
			for (int iter = 0; iter < endv; iter += 16) {
				//spoiler alert, I am jesus
				//important commands:
				//_mm512_stream_ps(void* mem_addr, __m512 a) (for saving shit)
				//_mm512_fmadd_ps(__m512 a, __m512 b, __m512 c) mult a and b, add c
				//_mm512_mul_ps(__m512 a, __m512db) mult a and b
				//_mm512_add_ps(__m512 a,__m512 b) add a and b
				//_mm512_div_round_ps(__m512 a, __m512 b, _MM_FROUND_TO_NEAREST_INT);


				//__nmask16 comparemask;
				a0 = _mm512_setzero_epi32();
				b0 = _mm512_set1_epi32(L - 1);
				c0 = _mm512_set1_epi32(1);
				Lv = _mm512_set1_ps((float)L);
				//if ((c + 31) < endv) { //prefetchhh
				//	_mm_prefetch((char*), _MM_HINT_T0
				//}
				//do le prefetching hereish?
				L1_v = _mm512_set1_epi32(L); //no relation to the cache ;)
				L2_v = _mm512_set1_epi32(L2);
				L3_v = _mm512_set1_epi32(L3);
				//loop begin //perhaps declare all vars below outside loop
				a = _mm512_cvt_roundps_epi32(_mm512_mul_ps(_mm512_load_ps(&posAll[iter]), Lv), _MM_FROUND_TO_NEG_INF); //goood stuff
				b = _mm512_cvt_roundps_epi32(_mm512_mul_ps(_mm512_load_ps(&posAll[finalnum + iter]), Lv), _MM_FROUND_TO_NEG_INF);
				c = _mm512_cvt_roundps_epi32(_mm512_mul_ps(_mm512_load_ps(&posAll[finalnum2 + iter]), Lv), _MM_FROUND_TO_NEG_INF);
				type = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_load_epi32(&typesAll[iter])), speed);
				d = _mm512_maskz_mov_epi32(_mm512_cmp_epi32_masks(a0, _mm512_sub_epi32(a, 1), c0), c0);
				e = _mm512_maskz_mov_epi32(_mm512_cmp_epi32_masks(a0, _mm512_sub_epi32(b, 1), c0), c0);
				f = _mm512_maskz_mov_epi32(_mm512_cmp_epi32_masks(a0, _mm512_sub_epi32(c, 1), c0), c0);//twelve
				g = _mm512_maskz_mov_epi32(_mm512_cmp_epi32_masks(_mm512_add_epi32(a, c0), b0, 1), c0);
				h = _mm512_maskz_mov_epi32(_mm512_cmp_epi32_masks(_mm512_add_epi32(b, c0), b0, 1), c0);//john cena
				i = _mm512_maskz_mov_epi32(_mm512_cmp_epi32_masks(_mm512_add_epi32(c, c0), b0, 1), c0);
				a_in = _mm512_add_epi32(_mm512_mul_epi32(_mm512_sub_epi32(a, d), L2_v), _mm512_add_epi32(_mm512_mul_epi32(L1_v, b), c)); //down
				b_in = _mm512_add_epi32(_mm512_mul_epi32(_mm512_add_epi32(a, g), L2_v), _mm512_add_epi32(_mm512_mul_epi32(L1_v, b), c)); //up
				c_in = _mm512_add_epi32(_mm512_mul_epi32(a, L2_v), _mm512_add_epi32(_mm512_mul_epi32(L1_v, _mm512_sub_epi32(b, e)), c)); //down
				d_in = _mm512_add_epi32(_mm512_mul_epi32(a, L2_v), _mm512_add_epi32(_mm512_mul_epi32(L1_v, _mm512_add_epi32(b, h)), c)); //up
				e_in = _mm512_add_epi32(_mm512_mul_epi32(a, L2_v), _mm512_add_epi32(_mm512_mul_epi32(L1_v, b), _mm512_sub_epi32(c, f))); //down
				f_in = _mm512_add_epi32(_mm512_mul_epi32(a, L2_v), _mm512_add_epi32(_mm512_mul_epi32(L1_v, b), _mm512_add_epi32(c, i))); //up
				_mm512_prefetch_i32gather_ps(_mm512_add_epi32(a_in, L3_v), Conc, 1, _MM_HINT_NTA);  //WTF DOES THE SCALE DOOO
				_mm512_prefetch_i32gather_ps(_mm512_add_epi32(b_in, L3_v), Conc, 1, _MM_HINT_NTA);
				_mm512_prefetch_i32gather_ps(_mm512_add_epi32(c_in, L3_v), Conc, 1, _MM_HINT_NTA);
				_mm512_prefetch_i32gather_ps(_mm512_add_epi32(d_in, L3_v), Conc, 1, _MM_HINT_NTA);
				_mm512_prefetch_i32gather_ps(_mm512_add_epi32(e_in, L3_v), Conc, 1, _MM_HINT_NTA); //faiiiirly certain these last 2 can be pulled using just... like normal vector operations.
				_mm512_prefetch_i32gather_ps(_mm512_add_epi32(f_in, L3_v), Conc, 1, _MM_HINT_NTA);
				t1 = _mm512_div_ps(Lv, _mm512_add_ps(d, g));
				t2 = _mm512_div_ps(Lv, _mm512_add_ps(e, h));
				t3 = _mm512_div_ps(Lv, _mm512_add_ps(f, i));
				GS10 = _mm512_mul_ps(_mm512_sub_ps(_mm512_i32gather_ps(b_in, Conc, 4, _MM_HINT_NTA), _mm512_i32gather_ps(a_in, Conc, 4, _MM_HINT_NTA)), t1);
				GS11 = _mm512_mul_ps(_mm512_sub_ps(_mm512_i32gather_ps(d_in, Conc, 4, _MM_HINT_NTA), _mm512_i32gather_ps(c_in, Conc, 4, _MM_HINT_NTA)), t2);
				GS12 = _mm512_mul_ps(_mm512_sub_ps(_mm512_i32gather_ps(f_in, Conc, 4, _MM_HINT_NTA), _mm512_i32gather_ps(e_in, Conc, 4, _MM_HINT_NTA)), t3);
				preval1 = _mm512_fmadd_ps(GS10, GS10, _mm512_fmadd_ps(GS11, GS11, _mm512_mul_ps(GS12, GS12))); //beautiful
				norm1 = _mm512_rsqrt28_ps(preval2);
				GS20 = _mm512_mul_ps(_mm512_sub_ps(_mm512_i32gather_ps(_mm512_add_epi32(b_in, L3_v), Conc, 4), _mm512_i32gather_ps(_mm512_add_epi32(a_in, L3_v), Conc, 4)), t1);
				GS21 = _mm512_mul_ps(_mm512_sub_ps(_mm512_i32gather_ps(_mm512_add_epi32(d_in, L3_v), Conc, 4), _mm512_i32gather_ps(_mm512_add_epi32(c_in, L3_v), Conc, 4)), t2);
				GS22 = _mm512_mul_ps(_mm512_sub_ps(_mm512_i32gather_ps(_mm512_add_epi32(f_in, L3_v), Conc, 4), _mm512_i32gather_ps(_mm512_add_epi32(e_in, L3_v), Conc, 4)), t3);
				preval2 = _mm512_fmadd_ps(GS20, GS20, _mm512_fmadd_ps(GS21, GS21, _mm512_mul_ps(GS22, GS22)));
				norm2 = _mm512_rsqrt28_ps(preval2);
				comparemask = _mm512_kand(_mm512_cmp_epi32_masks(a0, preval1), _mm512_cmp_epi32_masks(a0, preval2));
				t1 = _mm512_maskz_mul_ps(comparemask, type, _mm512_fmsub_ps(GS10, norm1, _mm512_mul_ps(GS20, norm2))); //type has speed in it
				t2 = _mm512_maskz_mul_ps(comparemask, type, _mm512_fmsub_ps(GS11, norm1, _mm512_mul_ps(GS21, norm2))); //reuse t var
				t3 = _mm512_maskz_mul_ps(comparemask, type, _mm512_fmsub_ps(GS12, norm1, _mm512_mul_ps(GS22, norm2)));
				_mm512_stream_ps(&movVec[iter], t1); //saves
				_mm512_stream_ps(&movVec[finalnum + iter], t2);
				_mm512_stream_ps(&movVec[finalnum2 + iter], t3);
			}
		} //so the context of "c" does not overlap
		#pragma omp for
		for (int c = endv; c < n; ++c) {
			i1 = min((int)floor(posAll[c] * rsideLength), (L - 1));
			i2 = min((int)floor(posAll[finalnum+c] * rsideLength), (L - 1));
			i3 = min((int)floor(posAll[finalnum2+c] * rsideLength), (L - 1));

			xUp = min((i1 + 1), L - 1);
			xDown = max((i1 - 1), 0);
			yUp = min((i2 + 1), L - 1);
			yDown = max((i2 - 1), 0);
			zUp = min((i3 + 1), L - 1);
			zDown = max((i3 - 1), 0);

			gradSub1[0] = (Conc[lv2*xUp+lv1*i2+i3] - Conc[lv2*xDown+lv1*i2+i3]) / (sideLength*(xUp - xDown));
			gradSub1[1] = (Conc[lv2*i1+lv1*yUp+i3] - Conc[lv2*i1+lv1*yDown+i3]) / (sideLength*(yUp - yDown));
			gradSub1[2] = (Conc[lv2*i1+lv1*i2+zUp] - Conc[lv2&i1+lv1*i2+zDown]) / (sideLength*(zUp - zDown));

			gradSub2[0] = (Conc[lv3+xUp*lv2+i2*lv1+i3] - Conc[lv3+lv2*xDown+lv1*i2+i3]) / (sideLength*(xUp - xDown));
			gradSub2[1] = (Conc[lv3+i1*lv2+yUp*lv1+i3] - Conc[lv3+lv2*i1+lv1*yDown+i3]) / (sideLength*(yUp - yDown));
			gradSub2[2] = (Conc[lv3+i1*lv2+i2*lv1+zUp] - Conc[lv3+lv2*i1+lv1*i2+zDown]) / (sideLength*(zUp - zDown));

			normGrad1 = getNorm(gradSub1);
			normGrad2 = getNorm(gradSub2);

			if ((normGrad1>0) && (normGrad2>0)) {
				movVec[c] = typesAll[c] * (gradSub1[0] / normGrad1 - gradSub2[0] / normGrad2)*speed;
				movVec[finalnum+c] = typesAll[c] * (gradSub1[1] / normGrad1 - gradSub2[1] / normGrad2)*speed;
				movVec[finalnum2+c] = typesAll[c] * (gradSub1[2] / normGrad1 - gradSub2[2] / normGrad2)*speed;
			}

			else {
				movVec[c] = 0;
				movVec[finalnum+c] = 0;
				movVec[finalnum2+c] = 0;
			}
		}
	}

	runDiffusionClusterStep_sw.mark();
}

static float getEnergy(float* posAll, int* typesAll, int n, float spatialRange, int targetN) {
    getEnergy_sw.reset();
    // Computes an energy measure of clusteredness within a subvolume. The size of the subvolume
    // is computed by assuming roughly uniform distribution within the whole volume, and selecting
    // a volume comprising approximately targetN cells.


	float * posSubvol = (float*)_mm_malloc(sizeof(float)*3 * n,64);
	int * typesSubvol = (int*)_mm_malloc(sizeof(int) * n, 64);
    float subVolMax = pow(float(targetN)/float(n),1.0/3.0)/2;

    if(quiet < 1)
        printf("subVolMax: %f\n", subVolMax);


    int nrCellsSubVol = 0;

    float intraClusterEnergy = 0.0;
    float extraClusterEnergy = 0.0;
	//parallel_for(blocked_range(0, n), [&](const blocked_range<size_t>& x) {
	#pragma omp parallel default(shared) if (parallels)
	{
		int currsubvol;
		int i1, i2;
		#pragma omp for
		for (i1 = 0; i1 < n; ++i1) {
			if ((fabs(posAll[i1] - 0.5) < subVolMax) && (fabs(posAll[finalnum+i1] - 0.5) < subVolMax) && (fabs(posAll[finalnum2+i1] - 0.5) < subVolMax)) {
				#pragma omp critical  //yay
				{
					currsubvol = nrCellsSubVol++; //iterate after
				}
				#pragma vector aligned
				#pragma vector nontemporal
				#pragma ivdep
				posSubvol[currsubvol:finalnum2:finalnum] = posAll[i1:finalnum2: finalnum]; //#thrash life
				typesSubvol[nrCellsSubVol] = typesAll[i1];
			}
		}
	}
	float nrSmallDist = 0.0;
	float spatialRangeSq = spatialRange*spatialRange;
	#pragma omp parallel default(shared) if (parallels)
	{
		int i1, i2, it, e;
		__attribute__((aligned(64))) float currDist[16];
		#pragma omp for
		for (i1 = 0; i1 < nrCellsSubVol; ++i1) {
			for (i2 = i1 + 1; i2 < nrCellsSubVol; ++i2) {
				e = min(16, (int)nrCellsSubVol);
				currDist[0:e] = sqrtf(getL2DistanceSq(posSubvol[i1] -posSubvol[i2:e], posSubvol[finalnum+i1] - posSubvol[finalnum+i2:e], posSubvol[finalnum2+i1] - posSubvol[finalnum2+i2:e])); //make sure is vectorizing!!!!!!!!!!!!!!!!!!!!!!!!!!!!
				for (it = 0; it < i2 + e; ++it) { //could maybe vectorize this
					if (currDist[it] < spatialRangeSq) {
						++nrSmallDist;//currDist/spatialRange;
						if (typesSubvol[i1] * typesSubvol[i2] > 0) {
							intraClusterEnergy = intraClusterEnergy + fmin(100.0, spatialRange / currDist[it]); 
						}
						else {
							extraClusterEnergy = extraClusterEnergy + fmin(100.0, spatialRange / currDist[it]); //''
						}
					}
				}
			}
		}
	}
	/*
    for (i1 = 0; i1 < nrCellsSubVol; i1++) { //can halve this if we try
        for (i2 = i1+1; i2 < nrCellsSubVol; i2++) {
			currDist = getL2Distance(posSubvol[i1][0], posSubvol[i1][1], posSubvol[i1][2], posSubvol[i2][0], posSubvol[i2][1], posSubvol[i2][2]);
			if (currDist < spatialRange) {
				nrSmallDist = nrSmallDist + 1;//currDist/spatialRange;
				if (typesSubvol[i1] * typesSubvol[i2] > 0) {
					intraClusterEnergy = intraClusterEnergy + fmin(100.0, spatialRange / currDist);
				}
				else {
					extraClusterEnergy = extraClusterEnergy + fmin(100.0, spatialRange / currDist);
				}
			}
        }
    }*/
    float totalEnergy = (extraClusterEnergy-intraClusterEnergy)/(1.0+100.0*nrSmallDist);
    getEnergy_sw.mark();
    return totalEnergy;
}

/*template <typename T>
class llist {
	class item {
		T * object = NULL;
		String key;
		item * next = NULL;
		~item() {
			delete next;
		}
	}
	~llist() {
		delete root;
		root = NULL;
	}
	item * root = NULL;
	T * getelement(string & searchfor, item *& in = root) {
		if (in != NULL) {
			if (in->key == searchfor) {
				linktype * temp = in;
				in = in->next;
				return in->object;
			}
			else {
				linktype * temp = func(in->next);
				if (in == root) root = temp;
				return temp->object;
			}
		}
		else return NULL;
	}
	void addelement(const string & key, const T * object,item *& current = root) {
		if (current == NULL) {
			item * temp = new item();
			temp->object = object;
			temp->key = key;
		}
		else {
			addelement(key, object, current->next);
		}
	}
	void removelelement(const string & key, item *& current = root) {
		if (getelement(key) != NULL);
		item * temp = root;
		root = root->next;
		temp->next = NULL;
		delete temp;
	}
}*/

static bool getCriterion(float* posAll, int* typesAll, int n, float spatialRange, int targetN) {
    getCriterion_sw.reset();
    // Returns 0 if the cell locations within a subvolume of the total system, comprising approximately targetN cells,
    // are arranged as clusters, and 1 otherwise.

    int nrClose=0;      // number of cells that are close (i.e. within a distance of spatialRange)
    int sameTypeClose=0; // number of cells of the same type, and that are close (i.e. within a distance of spatialRange)
    int diffTypeClose=0; // number of cells of opposite types, and that are close (i.e. within a distance of spatialRange)

    float* posSubvol = (float*)_mm_malloc(sizeof(float)*n*3, 64);    // array of all 3 dimensional cell positions in the subcube
	int i;

	__attribute__((aligned(64))) int typesSubvol[n];

    float subVolMax = pow(float(targetN)/float(n),1.0/3.0)/2;

    int nrCellsSubVol = 0;

    // the locations of all cells within the subvolume are copied to array posSubvol
	//parallel_for(blocked_range(0, n), [&](const blocked_range<size_t>& x) {
	
	#pragma omp parallel default(shared) if (parallels)
	{
		int subvolnum;
		int i1, i2;
		#pragma omp for 
		for (i1 = 0; i1 < n; ++i1) {
			if ((fabs(posAll[i1] - 0.5) < subVolMax) && (fabs(posAll[finalnum+i1] - 0.5) < subVolMax) && (fabs(posAll[finalnum2+i1] - 0.5) < subVolMax)) {
				#pragma omp critical
				{
					subvolnum = nrCellsSubVol++;
				}
				#pragma vector aligned
				#pragma vector nontemporal
				#pragma vector ivdef
				posSubvol[subvolnum:finalnum2: finalnum] = posAll[i1:finalnum2:finalnum]; //I'm sorry for thrashing yew ._.
				#pragma vector aligned
				#pragma vector nontemporal
				typesSubvol[subvolnum] = typesAll[i1];
			}
		}
	}

    if(quiet < 1)
        printf("number of cells in subvolume: %d\n", nrCellsSubVol);


    // If there are not enough cells within the subvolume, the correctness criterion is not fulfilled
    if ((((float)(nrCellsSubVol))/(float)targetN) < 0.25) {
        getCriterion_sw.mark();
        if(quiet < 2)
			printf("not enough cells in subvolume: %d\n", nrCellsSubVol);
        return false;
    }

    // If there are too many cells within the subvolume, the correctness criterion is not fulfilled
    if ((((float)(nrCellsSubVol))/(float)targetN) > 4) {
        getCriterion_sw.mark();
        if(quiet < 2)
            printf("too many cells in subvolume: %d\n", nrCellsSubVol);
        return false;
    }
	//could combine these but how many ops would it really save? not many..
	//parallel_for(blocked_range(0, nrCellsSubVol), [&](const blocked_range<size_t>& x) { 
	float spatialRangeSq = spatialRange * spatialRange;

	#pragma omp parallel default(shared) if (parallels)
	{
		__attribute__((aligned(64))) float currDist[16];
		//__declspec(align(64)) float expanded[3][16];
		int i1,it, i2, e,ipe;
		#pragma omp for
		for (i1 = 0; i1 < nrCellsSubVol; ++i1) {
			for (i2 = i1 + 1; i2 < nrCellsSubVol; i2+=16) {
				e = min(16,nrCellsSubVol-i2);
				//ipe = i2 + e; 
				#pragma vector aligned
				#pragma vector nontemporal
				currDist[0:e] = getL2DistanceSq(posSubvol[i1] - posSubvol[i2:e], posSubvol[finalnum+i1] - posSubvol[finalnum+i2:e], posSubvol[finalnum2+i1] - posSubvol[finalnum2+i2:e]);
				for (it = 0; it < e; ++it) {
					if (currDist[it] < spatialRangeSq) {
						#pragma omp critical
						{
							nrClose++;
							if (typesSubvol[i1] * typesSubvol[i2+it] < 0) {
								diffTypeClose++;
							}
							else {
								sameTypeClose++;
							}
						}
					}
				}
			}
		}
	}

    float correctness_coefficient = ((float)diffTypeClose)/(nrClose+1.0);

    // check if there are many cells of opposite types located within a close distance, indicative of bad clustering
    if (correctness_coefficient > 0.1) {
        getCriterion_sw.mark();
        if(quiet < 2)
            printf("cells in subvolume are not well-clustered: %f\n", correctness_coefficient);
        return false;
    }

    // check if clusters are large enough, i.e. whether cells have more than 100 cells of the same type located nearby
    float avgNeighbors = ((float)sameTypeClose/nrCellsSubVol);
    if(quiet < 1)
        printf("average neighbors in subvolume: %f\n", avgNeighbors);
    if (avgNeighbors < 100) {
        getCriterion_sw.mark();
        if(quiet < 2)
            printf("cells in subvolume do not have enough neighbors: %f\n", avgNeighbors);
        return false;
    }


    if(quiet < 1)
        printf("correctness coefficient: %f\n", correctness_coefficient);

    getCriterion_sw.mark();
    return true;
}

static const char usage_str[] = "USAGE:\t%s[-h] [-V] [--<param>=<value>]* <input file> \n";

static void usage(const char *name)
{
    die(usage_str, basename(name));
}

static void help(const char *name)
{
    fprintf(stderr, usage_str, name);
    fprintf(stderr, "DESCRIPTION\n"
            "\t Clustering of Cells in 3D space by movements along substance gradients\n"
            "\t In this simulation, there are two phases. In a first phase, a\n"
            "\t single initial cell moves randomly in 3 dimensional space and\n"
            "\t recursively gives rise to daughter cell by duplication. In the\n"
            "\t second phase, cells move along the gradients of their preferred\n"
            "\t substance. There are two substances in this example, and cells\n"
            "\t produce the same substance as they prefer. The substances\n"
            "\t diffuses and decays in 3D space.\n");
    fprintf(stderr, "PARAMETERS\n"
            "\t <input file> should have <param>=<value> for each of the following:\n"
            "\t speed\n\t    multiplicative factor for speed of gradient-based movement of the cells (float)\n"
            "\t T\n\t    Number of time steps of simulated cell movements (int64_t)\n"
            "\t L\n\t    Defines resolution of diffusion mesh (int64_t)\n"
            "\t D\n\t    Diffusion constant (float)\n"
            "\t mu\n\t    Decay constant (float)\n"
            "\t divThreshold\n\t    number of divisions a cell can maximally undergo (relevant only for the first phase of the simulation) (unsigned)\n"
            "\t finalNumberCells\n\t    Number of cells after cells have recursively duplicated (divided) (int64_t)\n"
            "\t spatialRange\n\t    defines the maximal spatial extend of the clusters. This parameter is only used for computing the energy function and the correctness criterion (float)\n");
    fprintf(stderr, "OPTIONS\n"
            "\t-h,--help\n\t    print this help message\n"
            "\t-v,--version\n\t    print configuration information\n"
            "\t-q,--quiet\n\t    lower output to stdout. Multiples accepted.\n"
            "\t-v,--verbose\n\t    increase output to stdout. Multiples accepted\n"
            "\t--<param>=<value>\n\t    override param/value form input file\n");
}

int main(int argc, char *argv[]) {
	init_sw.reset();
    const option opts[] =
    {
        {"help",            no_argument,       0, 'h'},
        {"version",         no_argument,       0, 'V'},
        {"quiet",           no_argument,       0, 'q'},
        {"verbose",         no_argument,       0, 'v'},
        {0, 0, 0, 0},
    };

    vector<char*> candidate_kvs;

    int opt;
    do
    {
        int in_ind = optind;
        opterr     = 0;
        opt        = getopt_long(argc, argv, "hVqv", opts, 0);
        switch(opt)
        {
        case 0:
            break;
        case '?':
            if(optopt == 0)
            {
                candidate_kvs.push_back(read_kv(argv, in_ind, &optind));
            }
            break;
        case 'h':
            help(argv[0]);
            exit(0);
        case 'V':
            print_sys_config(stderr);
            exit(0);
        case 'q':
            ++quiet;
            break;
        case 'v':
            --quiet;
            break;
        default:
            usage(argv[0]);
        case -1:
            break;
        };
    }
    while(opt != -1);

    if(optind+1 < argc)
        usage(argv[0]);

    fprintf(stderr, "==================================================\n");

    print_sys_config(stderr);

    const cdc_params params = get_params(argv[optind], candidate_kvs, quiet);

    print_params(&params, stderr);

    const float    speed            = params.speed;
    const int  T                = params.T;
    const int  L                = params.L;
    const float    D                = params.D;
    const float    mu               = params.mu;
    const unsigned divThreshold     = params.divThreshold;
    const int  finalNumberCells = params.finalNumberCells;
    const float    spatialRange     = params.spatialRange;
    const float    pathThreshold    = params.pathThreshold;

	finalnum = finalNumberCells;
	finalnum2 = 2 * finalnum;

    int i,c,d,e;
    int i1, i2, i3;

	lv1 = L;
	lv2 = L*L;
	lv3 = L*L*L;

    float energy;   // value that quantifies the quality of the cell clustering output. The smaller this value, the better the clustering.
    float* posAll= (float*)_mm_malloc(sizeof(float)*3*finalNumberCells, 64);   // array of all 3 dimensional cell positions
	float* pathTraveled = (float*)_mm_malloc(sizeof(float)*finalNumberCells, 64);
	int* numberDivisions = (int*)_mm_malloc(sizeof(float)*finalNumberCells, 64);
	int* typesAll = (int*)_mm_malloc(sizeof(float)*finalNumberCells, 64);
	float * Conc = (float*)_mm_malloc(sizeof(float) * 2 * L*L*L, 64); //christ, that barely fits in a signed int (waz dis intentional?)
	float * currMov = (float*)_mm_malloc(sizeof(float) * finalNumberCells * 3, 64); // array of all cell movements in the last time step
	float zeroFloat = 0.0;

	//concref[0] = new float **[7];
	//concref[1] = new float **[7];
    //posAll = new float*[3]; //SWITCH THESE DIMENSIONS LATER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    numberDivisions[0]=0;   // the first cell has initially undergone 0 duplications (= divisions)
    typesAll[0]=1;  // the first cell is of type 1

    bool currCriterion;
	//#pragma omp parallel //let's try this and compare it with the other one
	//{
		//#pragma omp for
		//for (i1 = 0; i1 < finalNumberCells; i1+=16) {
			//e = min(16, (int)finalNumberCells - i1);
			//pathTraveled[0:e] = zeroFloat;
		//}
	//}

    // Initialization of the various arrays

	#pragma omp parallel default(shared)  if (parallels)
	{
		#pragma vector aligned
		#pragma ivdep
		#pragma vector nontemporal	
		#pragma omp for simd
		for (i1 = 0; i1 < 3 * finalNumberCells; ++i1) {
			currMov[i1] = zeroFloat;
			posAll[i1] = (float)0.5;
		}
	}
	#pragma omp parallel default(shared)  if (parallels) 
	{
		#pragma vector aligned
		#pragma ivdep
		#pragma vector nontemporal	
		#pragma omp for simd
		for (int v = 0; v < finalNumberCells; ++v) {
			pathTraveled[v] = zeroFloat; //init touch to keep it close to the place where it will be worked on
			numberDivisions[v] = 0;   //same idea
			typesAll[v] = 1;
		}
	}
    // create 3D concentration matrix
	#pragma vector aligned
	#pragma ivdep
	#pragma vector nontemporal
	#pragma omp parallel for simd
	for (i1 = 0; i1 < 2*lv3; ++i1) {
		
	}

	int halfsies = (int)(0.5*(float)L);
	int sub;
	__attribute__((aligned(64))) int up[3];
	__attribute__((aligned(64))) int down[3];
	/*for (sub = 0; sub < 2; ++sub) {
		for (i1 = 0; i1 < 5; ++i1) {
			concref[sub][i1] = new float*[finalNumberCells];
		}
		concref[sub][i1][0] = &Conc[sub][halfsies][halfsies][halfsies]
		for (i2 = 0; i2 < finalNumberCells;++i2) {
		concred[sub][i1][0] = &Conc[sub]
	}*/
    init_sw.mark();
    fprintf(stderr, "%-35s = %le s\n",  "INITIALIZATION_TIME", init_sw.elapsed);
    compute_sw.reset();
    phase1_sw.reset();

    int64_t n = 1; // initially, there is one single cell

	//do parallel stuff here


	// Phase 1: Cells move randomly and divide until final number of cells is reached
	//fprintf(stderr,"not broken");
	int finalNumberCells2 = 2 * finalNumberCells;
		while (n<finalNumberCells) {
			//fprintf(stderr,"%d\n", (int)n);
			//fprintf(stderr, "not broken1\n");
			fprintf(stderr, "producin'\n");
			produceSubstances(Conc, posAll, typesAll, L, n); // Cells produce substances. Depending on the cell type, one of the two substances is produced.
			//fprintf(stderr,"not broken2\n");
			fprintf(stderr, "diffusin'\n");
			runDiffusionStep(Conc, L, D); // Simulation of substance diffusion
			//fprintf(stderr, "not broken3\rundn");
			fprintf(stderr, "decayin'\n");
			runDecayStep(Conc, L, mu);
			fprintf(stderr, "movin'\n");
			//fprintf(stderr, "not broken4\n");
			n = cellMovementAndDuplication(posAll, pathTraveled, typesAll, numberDivisions, pathThreshold, divThreshold, n);
			//fprintf(stderr, "not broken5\n");
			#if parallels
			int e = 16;
			#pragma omp parallel if (parallels)
			{
				#pragma omp for
				for (c=0; c<n; c+=16) {
					if (n - c < 16) { e = (int)n - c; }
					// boundary conditions
					for (d=0; d<3; d++) {
						if (posAll[d*finalnum+c:e]<0) { posAll[d*finalnum+c:e] = 0; }
						if (posAll[d*finalnum+c:e]>1) { posAll[d*finalnum+c:e] = 1; }
					}
				}
			}
			#else
			int tmp;
			for (d = 0; d<3; d++) {
				tmp = finalnum*d;
				#pragma vector aligned
				#pragma ivdep
				#pragma vector nontemporal}
				if (posAll[tmp:n]<0) { posAll[tmp:n] = 0; }
				if (posAll[tmp:n]>0.9999) { posAll[tmp:n] = 0.9999; } //<3
			}
			#endif
		}
		phase1_sw.mark();
		phase2_sw.reset();
		halfway = true;
		fprintf(stderr, "%-35s = %le s\n",  "PHASE1_TIME", phase1_sw.elapsed);
		return 0;
		/*fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "produceSubstances_TIME", produceSubstances_sw.elapsed, produceSubstances_sw.elapsed*100.0f / compute_sw.elapsed);
		fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "runDiffusionStep_TIME", runDiffusionStep_sw.elapsed, runDiffusionStep_sw.elapsed*100.0f / compute_sw.elapsed);
		fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "runDecayStep_TIME", runDecayStep_sw.elapsed, runDecayStep_sw.elapsed*100.0f / compute_sw.elapsed);
		fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "cellMovementAndDuplication_TIME", cellMovementAndDuplication_sw.elapsed, cellMovementAndDuplication_sw.elapsed*100.0f / compute_sw.elapsed);
		fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "runDiffusionClusterStep_TIME", runDiffusionClusterStep_sw.elapsed, runDiffusionClusterStep_sw.elapsed*100.0f / compute_sw.elapsed);
		fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "getEnergy_TIME", getEnergy_sw.elapsed, getEnergy_sw.elapsed*100.0f / compute_sw.elapsed);
		fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "getCriterion_TIME", getCriterion_sw.elapsed, getCriterion_sw.elapsed*100.0f / compute_sw.elapsed);
		fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "TOTAL_COMPUTE_TIME", compute_sw.elapsed, compute_sw.elapsed*100.0f / compute_sw.elapsed);

		fprintf(stderr, "==================================================\n");

		return 0;*/
		// Phase 2: Cells move along the substance gradients and cluster

		for (i = 0; i < T; i++) {

			if ((i % 10) == 0) {
				if (quiet < 1) {
					printf("step %d\n", i);
				}
				else if (quiet < 2) {
					printf("\rstep %d", i);
					fflush(stdout);
				}
			}

			if (quiet == 1)
				printf("\n");

			if (i == 0) {
				fprintf(stderr, "getenergy\n");
				energy = getEnergy(posAll, typesAll, n, spatialRange, 10000);
				fprintf(stderr, "getcriterion");
				currCriterion = getCriterion(posAll, typesAll, n, spatialRange, 10000);
				fprintf(stderr, "%-35s = %d\n", "INITIAL_CRITERION", currCriterion);
				fprintf(stderr, "%-35s = %le\n", "INITIAL_ENERGY", energy);
			}

			if (i == (T - 1)) {
				fprintf(stderr, "getenergy\n");
				energy = getEnergy(posAll, typesAll, n, spatialRange, 10000);
				fprintf(stderr, "getcriterion");
				currCriterion = getCriterion(posAll, typesAll, n, spatialRange, 10000);
				fprintf(stderr, "%-35s = %d\n", "FINAL_CRITERION", currCriterion);
				fprintf(stderr, "%-35s = %le\n", "FINAL_ENERGY", energy);

			}

			//perhaps combine these four into one thing:
			fprintf(stderr, "produce\n");
			produceSubstances(Conc, posAll, typesAll, L, n);
			fprintf(stderr, "diffuse\n");
			runDiffusionStep(Conc, L, D);
			fprintf(stderr, "decay\n");
			runDecayStep(Conc, L, mu);
			fprintf(stderr, "cluster\n");
			runDiffusionClusterStep(Conc,currMov, posAll, typesAll, n, L, speed);

			//parallel_for(blocked_range(0, n), [&](const blocked_range<size_t>& x) {
			#pragma omp parallel default(shared)  if (parallels && false)
			{
				int d,alignv;
				for (d = 0; d < 3; ++d) {
					alignv = finalNumberCells*d;
					#pragma vector aligned
					#pragma vector nontemporal
					#pragma ivdep
					#pragma omp for simd
					for (int c = 0; c < n; ++c) {
						posAll[alignv + c] = posAll[alignv + c] + currMov[alignv + c];
						if (posAll[alignv+c] < 0) {
							posAll[alignv+c] = 0;
						}
						if (posAll[alignv+c] > 0.9999) {
							posAll[alignv+c] = 1;
						}
					}
				}
			}
		}
		phase2_sw.mark();
		compute_sw.mark();

    fprintf(stderr, "%-35s = %le s\n",  "PHASE2_TIME", phase2_sw.elapsed);


    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "produceSubstances_TIME",          produceSubstances_sw.elapsed, produceSubstances_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "runDiffusionStep_TIME",           runDiffusionStep_sw.elapsed, runDiffusionStep_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "runDecayStep_TIME",               runDecayStep_sw.elapsed, runDecayStep_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "cellMovementAndDuplication_TIME", cellMovementAndDuplication_sw.elapsed, cellMovementAndDuplication_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "runDiffusionClusterStep_TIME",    runDiffusionClusterStep_sw.elapsed, runDiffusionClusterStep_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "getEnergy_TIME",                  getEnergy_sw.elapsed, getEnergy_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "getCriterion_TIME",               getCriterion_sw.elapsed, getCriterion_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "TOTAL_COMPUTE_TIME",              compute_sw.elapsed, compute_sw.elapsed*100.0f/compute_sw.elapsed);

    fprintf(stderr, "==================================================\n");

    return 0;
}
