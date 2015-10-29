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

#include <cilk/cilk.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
//#include <cmath>
#include <getopt.h>
#include "util.hpp"
//#include <tbb/tbb.h>
#include <algorithm>
//#include <mkl.h>
#include <omp.h>
#include <cmath>
//used floating point mode relaxed

using namespace std;
//using namespace tbb;
static int quiet = 0;

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
    float distArray[3];
    distArray[0] = pos2x-pos1x;
    distArray[1] = pos2y-pos1y;
    distArray[2] = pos2z-pos1z;
    float l2Norm = getNorm(distArray);
    return l2Norm;
}

__attribute__((vector)) inline float getL2DistanceSq(float dx,float dy,float dz) {
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

static void produceSubstances(float**** Conc, float** posAll, int* typesAll, int L, int n) {
	produceSubstances_sw.reset();

	// increases the concentration of substances at the location of the cells
	//float sideLength = 1 / (float)L; // length of a side of a diffusion voxel
	float rsideLength = (float)L;
	int div = min(n / 20, 500); //figure this out later
	#pragma omp parallel default(shared)
	{
		int c,c2,e, i1, i2, i3;
		#pragma omp for
		for (c = 0; c < n; ++c) {
			i1 = std::min((int)(posAll[0][c] * rsideLength), (L - 1));
			i2 = std::min((int)(posAll[1][c] * rsideLength), (L - 1));
			i3 = std::min((int)(posAll[2][c] * rsideLength), (L - 1));
			if (typesAll[c] == 1) {
				Conc[0][i1][i2][i3] += 0.1;
				if (Conc[0][i1][i2][i3] > 1) {
					Conc[0][i1][i2][i3] = 1;
				}
			}
			else {
				Conc[1][i1][i2][i3] += 0.1;
				if (Conc[1][i1][i2][i3] > 1) {
					Conc[1][i1][i2][i3] = 1;
				}
			}
		}
	}
    produceSubstances_sw.mark();
}

static void runDiffusionStep(float **** Conc, float **** tempConc, int L, float D) {
    runDiffusionStep_sw.reset();
    // computes the changes in substance concentrations due to diffusion
    //float tempConc[2][L][L][L]; //holy jesus pls no
	#pragma omp parallel default(shared)
	{
		int i1, i2, i3, subInd, e;
		#pragma omp for
		for (int i1 = 0; i1 < L; ++i1) {
			for (int i2 = 0; i2 < L; ++i2) {
				tempConc[0][i1][i2][0:L] = Conc[0][i1][i2][0:L];
				tempConc[1][i1][i2][0:L] = Conc[1][i1][i2][0:L];
			}
		}
		int xUp, xDown, yUp, yDown, zUp, zDown;
		int up[3][16];
		int down[3][16];
		
		#pragma omp for      //                  TRY TO PUT PARALLEL FOR ON MIDDLE "FOR" LOOP SO WE CAN  USE MORE THREADING AT ONCE!!!
		for (i1 =0; i1 < L; ++i1) {
			//e = min(i1 + 15, (int)n) - i1 + 1; //size of 
			for (i2 = 0; i2 < L; ++i2) {
				//for (i3 = 0; i3 < L; ++i3) {
					//xUp = (i1 + 1);
					//xDown = (i1 - 1);
					//yUp = (i2 + 1);
					//yDown = (i2 - 1);
					//zUp = (i3 + 1);
					//zDown = (i3 - 1);
					
					for (subInd = 0; subInd < 2; subInd++) {
						if ((i1 + 1) < L) {
							Conc[subInd][i1][i2][0:L] += (tempConc[subInd][i1+1][i2][0:L] - tempConc[subInd][i1][i2][0:L])*D / 6;
						}
						if ((i1 - 1) >= 0) {
							Conc[subInd][i1][i2][0:L] += (tempConc[subInd][i1 - 1][i2][0:L] - tempConc[subInd][i1][i2][0:L])*D / 6;
						}
						if ((i2 + 1) < L) {
							Conc[subInd][i1][i2][0:L] += (tempConc[subInd][i1][i2 + 1][0:L] - tempConc[subInd][i1][i2][0:L])*D / 6;
						}
						if ((i2 - 1) >= 0) {
							Conc[subInd][i1][i2][0:L] += (tempConc[subInd][i1][i2 - 1][0:L] - tempConc[subInd][i1][i2][0:L])*D / 6;
						}
						Conc[subInd][i1][i2][0:(L - 1)] += (tempConc[subInd][i1][i2][1:(L - 1)] - tempConc[subInd][i1][i2][0:(L - 1)])*D / 6;
						Conc[subInd][i1][i2][1:(L - 1)] += (tempConc[subInd][i1][i2][0:(L - 1)] - tempConc[subInd][i1][i2][1:(L - 1)])*D / 6;
						//lends itself quite well to vectorization
					}
				//}
			}
		}
	}
    runDiffusionStep_sw.mark();
}

static void runDecayStep(float**** Conc, int L, float mu) {
    runDecayStep_sw.reset();
    // computes the changes in substance concentrations due to decay
	float val = (1 - mu);
	#pragma omp parallel default(shared)
	{
		int i1, i2;
		#pragma omp for
		for (i1 = 0; i1 < L; ++i1) {
			for (i2 = 0; i2 < L; i2++) {
				Conc[0][i1][i2][0:L] = Conc[0][i1][i2][0:L] * val;
				Conc[1][i1][i2][0:L] = Conc[1][i1][i2][0:L] * val;
			}
		}
	}
    runDecayStep_sw.mark();
}

static int cellMovementAndDuplication(float** posAll, float* pathTraveled, int* typesAll, int* numberDivisions, float pathThreshold, int divThreshold, int n) {
    cellMovementAndDuplication_sw.reset();
	int div = min(n / 20, 500); //figure this out later
	int currentNumberCells = n;
	#pragma omp parallel default(shared)
	{
		int c, i, e;
		int newcellnum;
		float currentrNorm[16];
		float currentrNorm2;
		float currentCellMovement[3][16];
		float duplicatedCellOffset[3];
		#pragma omp for
		for (c = 0; c < n; c += 16) {
			e = min(16, (int)n - c);	 //size of stream
			// random cell movement
			for (i = 0; i < e; ++i) {
				currentCellMovement[0][i] = RandomFloatPos() - 0.5;
				currentCellMovement[1][i] = RandomFloatPos() - 0.5;
				currentCellMovement[2][i] = RandomFloatPos() - 0.5;
			}
			//currentNorm = getNorm(currentCellMovement);
			currentrNorm[0:e] = 1.0 / sqrtf(currentCellMovement[0][0:e]* currentCellMovement[0][0:e] + currentCellMovement[1][0:e]*currentCellMovement[1][0:e] + currentCellMovement[2][0:e]*currentCellMovement[2][0:e]);

			posAll[0][c:e] += 0.1*currentCellMovement[0][0:e] * currentrNorm[0:e];
			posAll[1][c:e] += 0.1*currentCellMovement[1][0:e] * currentrNorm[0:e];
			posAll[2][c:e] += 0.1*currentCellMovement[2][0:e] * currentrNorm[0:e];
			pathTraveled[c:e] += 0.1;
			for (i = c; i < (e+c); ++i) { //we'll figure this out later
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
						posAll[0][newcellnum] = posAll[0][i] + 0.05*duplicatedCellOffset[0] * currentrNorm2;
						posAll[1][newcellnum] = posAll[0][i] + 0.05*duplicatedCellOffset[1] * currentrNorm2;
						posAll[2][newcellnum] = posAll[0][i] + 0.05*duplicatedCellOffset[2] * currentrNorm2;

					}

				}
			}
		}
	}
    cellMovementAndDuplication_sw.mark();
    return currentNumberCells;
}


static void runDiffusionClusterStep(float**** Conc, float** movVec, float** posAll, int* typesAll, int n, int L, float speed) {
	runDiffusionClusterStep_sw.reset();
	// computes movements of all cells based on gradients of the two substances

	float sideLength = 1 / (float)L; // length of a side of a diffusion voxel
	float rsidelength = (float)L;
	//float gradsub1[3];
	//float gradsub2[3];

	//float normGrad1, normGrad2;
	//int c, i1, i2, i3, xUp, xDown, yUp, yDown, zUp, zDown;

	//(blocked_range(0, n), [&](const blocked_range<size_t>& x) {

	#pragma omp parallel default(shared)
	{
		//int c, i1, i2, i3, xUp, xDown, yUp, yDown, zUp, zDown;
		//int[][] i[3][16];
		int i[3];
		int up[3][16];
		int down[3][16];
		int normGrad1[16]; //just reuse so we don't need to allocate more
		int normGrad2[16];
		float gradsub1[3][16];
		float gradsub2[3][16];
		int e,c;
		int i1, i2, i3,it;
		int cit;
		for (c = 0; c < n; c += 16) {
			e = min(16, (int)n - c);
			for (it = 0; it < e; ++it) {
				cit = c + it;
				i[0] = std::min((int)(posAll[0][cit] * rsidelength), (L - 1));
				i[1] = std::min((int)(posAll[1][cit] * rsidelength), (L - 1));
				i[2] = std::min((int)(posAll[2][cit] * rsidelength), (L - 1));
				up[0][it] = std::min((i1 + 1), L - 1);
				down[0][it] = std::max((i1 - 1), 0);
				up[1][it] = std::min((i2 + 1), L - 1);
				down[1][it] = std::max((i2 - 1), 0);
				up[2][it] = std::min((i3 + 1), L - 1);
				down[2][it] = std::max((i3 - 1), 0);
				gradsub1[0][cit] = Conc[0][up[0][it]][i[1]][i[2]] - Conc[0][down[0][it]][i[1]][i[2]]; //redo wiff ze mappingz
				gradsub1[1][cit] = Conc[0][i[0]][up[1][it]][i[2]] - Conc[0][i[0]][down[1][it]][i[2]];
				gradsub1[2][cit] = Conc[0][i[0]][i[1]][up[2][it]] - Conc[0][i[0]][i[1]][down[2][it]];
				gradsub1[0][cit] = Conc[1][up[0][it]][i[1]][i[2]] - Conc[1][down[0][it]][i[1]][i[2]];
				gradsub1[1][cit] = Conc[1][i[0]][up[1][it]][i[2]] - Conc[1][i[0]][down[1][it]][i[2]];
				gradsub1[2][cit] = Conc[1][i[0]][i[1]][up[2][it]] - Conc[1][i[0]][i[1]][down[2][it]];
			}
			for (it = 0; it < 3; ++it) {
				gradsub1[it][0:e] /= (sideLength*(up[it][0:e] - down[it][0:e]));
				gradsub2[it][0:e] /= (sideLength*(up[it][0:e] - down[it][0:e]));
			}
			//i1 = std::min((int)floor(posAll[c][0] * rsidelength), (L - 1));
			//i2 = std::min((int)floor(posAll[c][1] * rsidelength), (L - 1));
			//i3 = std::min((int)floor(posAll[c][2] * rsidelength), (L - 1));
			/*
			up[0][c:e] = (L - 1) - (i[0][c:e] < (L - 2))*(L - 2 - i[0][c:e]);
			up[1][c:e] = (L - 1) - (i[1][c:e] < (L - 2))*(L - 2 - i[1][c:e]);
			up[2][c:e] = (L - 1) - (i[2][c:e] < (L - 2))*(L - 2 - i[2][c:e]);
			down[c:e] = ((i[0][c:e] - 1) > 0)*(i[0][c:e] - 1);
			down[c:e] = ((i[1][c:e] - 1) > 0)*(i[1][c:e] - 1);
			down[c:e] = ((i[2][c:e] - 1) > 0)*(i[2][c:e] - 1);
			

			gradsub1[0][c:e] = (Conc[0][up[0][c:e]][i[1][c:e]][i[2][c:e]] - Conc[0][down[0][c:e]][i[1][c:e]][i[2][c:e]]) / (sideLength*(up[0][c:e] - down[0][c:e]));
			gradsub1[1][c:e] = (Conc[0][i[0][c:e]][up[1][c:e]][i[2][c:e]] - Conc[0][i[0][c:e]][down[1][c:e]][i[2][c:e]]) / (sideLength*(up[1][c:e] - down[1][c:e]));
			gradsub1[2][c:e] = (Conc[0][i[0][c:e]][i[1][c:e]][up[2][c:e]] - Conc[0][i[0][c:e]][i[1][c:e]][down[2][c:e]]) / (sideLength*(up[2][c:e] - down[2][c:e]));

			gradsub2[0][c:e] = (Conc[1][up[0][c:e]][i[1][c:e]][i[2][c:e]] - Conc[1][down[0][c:e]][i[1][c:e]][i[2][c:e]]) / (sideLength*(up[0][c:e] - down[0][c:e]));
			gradsub2[1][c:e] = (Conc[1][i[0][c:e]][up[1][c:e]][i[2][c:e]] - Conc[1][i[0][c:e]][down[1][c:e]][i[2][c:e]]) / (sideLength*(up[1][c:e] - down[1][c:e]));
			gradsub2[2][c:e] = (Conc[1][i[0][c:e]][i[1][c:e]][up[2][c:e]] - Conc[1][i[0][c:e]][i[1][c:e]][down[2][c:e]]) / (sideLength*(up[2][c:e] - down[2][c:e]));
			*/
			//normGrad1 = getNorm(gradsub1);
			//normGrad2 = getNorm(gradsub2);
			normGrad1[c:e] = sqrt(gradsub1[0][c:e] * gradsub1[0][c:e] + gradsub1[1][c:e] * gradsub1[1][c:e] + gradsub1[2][c:e] * gradsub1[2][c:e]);
			normGrad2[c:e] = sqrt(gradsub2[0][c:e] * gradsub2[0][c:e] + gradsub2[1][c:e] * gradsub2[1][c:e] + gradsub2[2][c:e] * gradsub2[2][c:e]);

			movVec[0][c:e] = typesAll[c:e] * (gradsub1[0][0:e] / normGrad1[0:e] - gradsub2[0][0:e] / normGrad2[0:e])*speed*(normGrad1[0:e] > 0)* (normGrad2[0:e] > 0);
			movVec[1][c:e] = typesAll[c:e] * (gradsub1[1][0:e] / normGrad1[0:e] - gradsub2[1][0:e] / normGrad2[0:e])*speed*(normGrad1[0:e] > 0)* (normGrad2[0:e] > 0);
			movVec[2][c:e] = typesAll[c:e] * (gradsub1[2][0:e] / normGrad1[0:e] - gradsub2[2][0:e] / normGrad2[0:e])*speed*(normGrad1[0:e] > 0)* (normGrad2[0:e] > 0);
		}
	}
	runDiffusionClusterStep_sw.mark();
}

/*
static void runClusterStep(float**** Conc, float** movVec, float** posAll, int* typesAll, int n, int L, float speed) {
    runDiffusionClusterStep_sw.reset();
    // computes movements of all cells based on gradients of the two substances

    float sideLength = 1/(float)L; // length of a side of a diffusion voxel

    float gradsub1[3];
    float gradsub2[3];

    float normGrad1, normGrad2;
    int c, i1, i2, i3, xUp, xDown, yUp, yDown, zUp, zDown;
	parallel_for(blocked_range(0, n), 1000, [&](const blocked_range<size_t>& x) {
		for (c = x.begin(); c < x.end(); ++c) {
			i1[c:e] = (L - 1) - (int)(posAll[c][0]/sideLength);
			i1 = std::min((int)floor(posAll[c][0] / sideLength), (L - 1));
			i2 = std::min((int)floor(posAll[c][1] / sideLength), (L - 1));
			i3 = std::min((int)floor(posAll[c][2] / sideLength), (L - 1));
			


			//split this into 3 different cases, one for i1 = 0, one for i1 = L-1
			xUp = std::min((i1 + 1), L - 1);
			xDown = std::max((i1 - 1), 0);
			yUp = std::min((i2 + 1), L - 1);
			yDown = std::max((i2 - 1), 0);
			zUp = std::min((i3 + 1), L - 1);
			zDown = std::max((i3 - 1), 0);

			gradsub1[0] = (Conc[0][xUp][i2][i3] - Conc[0][xDown][i2][i3]) / (sideLength*(xUp - xDown));
			gradsub1[1] = (Conc[0][i1][yUp][i3] - Conc[0][i1][yDown][i3]) / (sideLength*(yUp - yDown));
			gradsub1[2] = (Conc[0][i1][i2][zUp] - Conc[0][i1][i2][zDown]) / (sideLength*(zUp - zDown));

			gradsub2[0] = (Conc[1][xUp][i2][i3] - Conc[1][xDown][i2][i3]) / (sideLength*(xUp - xDown));
			gradsub2[1] = (Conc[1][i1][yUp][i3] - Conc[1][i1][yDown][i3]) / (sideLength*(yUp - yDown));
			gradsub2[2] = (Conc[1][i1][i2][zUp] - Conc[1][i1][i2][zDown]) / (sideLength*(zUp - zDown));

			//normGrad1 = getNorm(gradsub1);
			//normGrad2 = getNorm(gradsub2);
			rnormGrad1 = 1/getNorm(gradsub1);
			rnormGrad2 = 1/getNorm(gradsub2);

			if ((normGrad1 > 0) && (normGrad2 > 0)) {
				movVec[c][0:3] = typesAll[c] * (gradsub1[0:3] / normGrad1 - gradsub2[0:3] / normGrad2)*speed;
			}

			else {
				movVec[c][0:3] = 0;
			}
		}
    }
    runDiffusionClusterStep_sw.mark();
}
*/
static float getEnergy(float** posAll, int* typesAll, int n, float spatialRange, int targetN) {
    getEnergy_sw.reset();
    // Computes an energy measure of clusteredness within a subvolume. The size of the subvolume
    // is computed by assuming roughly uniform distribution within the whole volume, and selecting
    // a volume comprising approximately targetN cells.


   // float** posSubvol=0;    // array of all 3 dimensional cell positions
	float** posSubvol = new float*[3];
	for (int i = 0; i < 3; ++i) {
		posSubvol[i] = new float[n];
	}
    int typesSubvol[n];

    float subVolMax = pow(float(targetN)/float(n),1.0/3.0)/2;

    if(quiet < 1)
        printf("subVolMax: %f\n", subVolMax);


    int nrCellsSubVol = 0;

    float intraClusterEnergy = 0.0;
    float extraClusterEnergy = 0.0;
	//parallel_for(blocked_range(0, n), [&](const blocked_range<size_t>& x) {
	#pragma omp parallel default(shared)
	{
		int currsubvol;
		int i1, i2;
		#pragma omp for
		for (i1 = 0; i1 < n; ++i1) {
			if ((fabs(posAll[0][i1] - 0.5) < subVolMax) && (fabs(posAll[1][i1] - 0.5) < subVolMax) && (fabs(posAll[2][i1] - 0.5) < subVolMax)) {
				#pragma omp critical //yay
				{
					currsubvol = nrCellsSubVol++; //iterate after
				}
				posSubvol[0][currsubvol] = posAll[0][i1];
				posSubvol[1][currsubvol] = posAll[1][i1];
				posSubvol[2][currsubvol] = posAll[2][i1];
				typesSubvol[nrCellsSubVol] = typesAll[i1];
			}
		}
	}
	float nrSmallDist = 0.0;
	float spatialRangeSq = spatialRange*spatialRange;
	#pragma omp parallel default(shared)
	{
	//parallel_for(blocked_range(0, nrCellsSubVol), [&](const blocked_range<size_t>& x) {
		//float currDist;
		int i1, i2, it, e;
		float currDist[16];
		float expanded[3][16];
		#pragma omp for
		for (i1 = 0; i1 < nrCellsSubVol; ++i1) {
			for (it = 0; it < 3; ++it) { //save a few operations
				expanded[it][0:16] = posSubvol[it][i1];
			}
			for (i2 = i1 + 1; i2 < nrCellsSubVol; ++i2) {
				e = min(16, (int)nrCellsSubVol);
				currDist[0:e] = sqrtf(getL2DistanceSq(expanded[0][0:e]-posSubvol[0][i2:e], expanded[1][0:e] - posSubvol[1][i2:e], expanded[2][0:e] - posSubvol[2][i2:e])); //make sure is vectorizing!!!!!!!!!!!!!!!!!!!!!!!!!!!!
				for (it = 0; it < i2 + e; ++it) { //could maybe vectorize this
					if (currDist[it] < spatialRangeSq) {
						++nrSmallDist;//currDist/spatialRange;
						if (typesSubvol[i1] * typesSubvol[i2] > 0) {
							intraClusterEnergy = intraClusterEnergy + fmin(100.0, spatialRange / currDist[it]); //only perform costly sqrt when necessary
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

static bool getCriterion(float** posAll, int* typesAll, int n, float spatialRange, int targetN) {
    getCriterion_sw.reset();
    // Returns 0 if the cell locations within a subvolume of the total system, comprising approximately targetN cells,
    // are arranged as clusters, and 1 otherwise.

    int nrClose=0;      // number of cells that are close (i.e. within a distance of spatialRange)
    int sameTypeClose=0; // number of cells of the same type, and that are close (i.e. within a distance of spatialRange)
    int diffTypeClose=0; // number of cells of opposite types, and that are close (i.e. within a distance of spatialRange)

    float** posSubvol = new float*[3];    // array of all 3 dimensional cell positions in the subcube
	int i;
	for (i = 0; i < 3; ++i) {
		posSubvol[i] = new float[n];
	}
    int typesSubvol[n];

    float subVolMax = pow(float(targetN)/float(n),1.0/3.0)/2;

    int nrCellsSubVol = 0;

    // the locations of all cells within the subvolume are copied to array posSubvol
	//parallel_for(blocked_range(0, n), [&](const blocked_range<size_t>& x) {
	#pragma omp parallel default(shared)
	{
		int subvolnum;
		int i1, i2;
		#pragma omp for
		for (i1 = 0; i1 < n; ++i1) {
			if ((fabs(posAll[0][i1] - 0.5) < subVolMax) && (fabs(posAll[1][i1] - 0.5) < subVolMax) && (fabs(posAll[2][i1] - 0.5) < subVolMax)) {
				subvolnum = nrCellsSubVol++;
				posSubvol[0][subvolnum] = posAll[0][i1];
				posSubvol[1][subvolnum] = posAll[1][i1];
				posSubvol[2][subvolnum] = posAll[2][i1];
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

	#pragma omp parallel default(shared)
	{

		__declspec(align(64)) float currDist[16];
		__declspec(align(64)) float expanded[3][16];
		int i1,it, i2, e,ipe;
		#pragma omp for
		for (i1 = 0; i1 < nrCellsSubVol; ++i1) {
			for (it = 0; it < 3; ++it) { //save a few operations
				expanded[it][0:16] = posSubvol[it][i1];
			}
			for (i2 = i1 + 1; i2 < nrCellsSubVol; i2+=16) {
				e = min(16,nrCellsSubVol-i2);
				//ipe = i2 + e; 
				currDist[0:e] = getL2DistanceSq(expanded[0][0:e] - posSubvol[0][i2:e], expanded[1][0:e] - posSubvol[1][i2:e], expanded[2][0:e] - posSubvol[2][i2:e]);
				//getL2Distance(posSubvol[0][i1], posSubvol[1][i1], posSubvol[2][i1], posSubvol[0][i2], posSubvol[1][i2], posSubvol[2][i2]);
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
    const int64_t  T                = params.T;
    const int64_t  L                = params.L;
    const float    D                = params.D;
    const float    mu               = params.mu;
    const unsigned divThreshold     = params.divThreshold;
    const int64_t  finalNumberCells = params.finalNumberCells;
    const float    spatialRange     = params.spatialRange;
    const float    pathThreshold    = params.pathThreshold;

    int i,c,d,e;
    int i1, i2, i3, i4;

    float energy;   // value that quantifies the quality of the cell clustering output. The smaller this value, the better the clustering.

    float** posAll= new float*[3];   // array of all 3 dimensional cell positions
	//float **** concref = new float ***[2]; //0 for center, 1 for north, 2 for south, 3 for east, 4 for west
	//concref[0] = new float **[7];
	//concref[1] = new float **[7];
    //posAll = new float*[3]; //SWITCH THESE DIMENSIONS LATER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    float** currMov=0;  // array of all 3 dimensional cell movements at the last time point
    currMov = new float*[3]; // array of all cell movements in the last time step
    float zeroFloat = 0.0;

    __declspec(align(64)) float pathTraveled[finalNumberCells];   // array keeping track of length of path traveled until cell divides
	__declspec(align(64)) int numberDivisions[finalNumberCells];  //array keeping track of number of division a cell has undergone
	__declspec(align(64)) int typesAll[finalNumberCells];     // array specifying cell type (+1 or -1)

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
	pathTraveled[0:finalNumberCells] = zeroFloat;

    // Initialization of the various arrays
	for (i1 = 0; i1 < 3; ++i1) {
		currMov[i1] = (float*)_mm_malloc(sizeof(float*)*finalNumberCells, 64);
		posAll[i1] = (float*)_mm_malloc(sizeof(float*)*finalNumberCells, 64);
		currMov[i1][0:finalNumberCells] = zeroFloat;
		posAll[i1][0:finalNumberCells] = 0.5;
    }

    // create 3D concentration matrix
    float**** Conc; //we also need to rearrange the dimensions on this
	float**** tempConc;
    Conc = new float***[2]; //HOW DID YOU MAKE A MISTAKE LIKE THAT!?!??!??!?!?!?!
	tempConc = new float***[2];
	int i22, i33;
	for (i1 = 0; i1 < 2; ++i1) {
		Conc[i1] = new float**[L];
		tempConc[i1] = new float**[L];
		//#pragma omp parallel default(shared)
		//{
		//int i22, i33;
			//#pragma omp for
			for (i22 = 0; i22 < L; ++i22) {
				Conc[i1][i22] = new float*[L];
				tempConc[i1][i22] = new float*[L];
				for (i33 = 0; i33 < L; ++i33) {
					Conc[i1][i22][i33] = (float*)_mm_malloc(sizeof(float)*L, 64);
					tempConc[i1][i22][i33] = (float*)_mm_malloc(sizeof(float)*L, 64);
					//for (i4 = 0; i4 < L; i4++) {
					//tempConc[i1][i2][i3][0:L] = zeroFloat;
					Conc[i1][i22][i33][0:L] = zeroFloat;
					//}
				}
			}
		//}
	}

	int halfsies = (int)(0.5*(float)L);
	int sub;
	__declspec(align(64)) int up[3];
	__declspec(align(64)) int down[3];
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
		while (n<finalNumberCells) {
			//fprintf(stderr,"%d\n", (int)n);
			//fprintf(stderr, "not broken1\n");
			produceSubstances(Conc, posAll, typesAll, L, n); // Cells produce substances. Depending on the cell type, one of the two substances is produced.
			//fprintf(stderr,"not broken2\n");
			runDiffusionStep(Conc,tempConc, L, D); // Simulation of substance diffusion
			//fprintf(stderr, "not broken3\n");
			runDecayStep(Conc, L, mu);
			//fprintf(stderr, "not broken4\n");
			n = cellMovementAndDuplication(posAll, pathTraveled, typesAll, numberDivisions, pathThreshold, divThreshold, n);
			//fprintf(stderr, "not broken5\n");
			//#pragma omp parallel
			//{
				//#pragma omp for
				//for (c=0; c<n; c+=16) {
					// boundary conditions
					//e = min(16,(int)n-c);	
					for (d=0; d<3; d++) {
						if (posAll[d][0:n]<0) { posAll[d][0:n] = 0; }
						if (posAll[d][0:n]>1) { posAll[d][0:n] = 1; }
					}
				//}
			//}
		}
		phase1_sw.mark();
		phase2_sw.reset();
		fprintf(stderr, "%-35s = %le s\n",  "PHASE1_TIME", phase1_sw.elapsed);

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
				energy = getEnergy(posAll, typesAll, n, spatialRange, 10000);
				currCriterion = getCriterion(posAll, typesAll, n, spatialRange, 10000);
				fprintf(stderr, "%-35s = %d\n", "INITIAL_CRITERION", currCriterion);
				fprintf(stderr, "%-35s = %le\n", "INITIAL_ENERGY", energy);
			}

			if (i == (T - 1)) {
				energy = getEnergy(posAll, typesAll, n, spatialRange, 10000);
				currCriterion = getCriterion(posAll, typesAll, n, spatialRange, 10000);
				fprintf(stderr, "%-35s = %d\n", "FINAL_CRITERION", currCriterion);
				fprintf(stderr, "%-35s = %le\n", "FINAL_ENERGY", energy);

			}

			//perhaps combine these four into one thing:

			produceSubstances(Conc, posAll, typesAll, L, n);
			runDiffusionStep(Conc, tempConc, L, D);
			runDecayStep(Conc, L, mu);
			runDiffusionClusterStep(Conc, currMov, posAll, typesAll, n, L, speed);
			//parallel_for(blocked_range(0, n), [&](const blocked_range<size_t>& x) {
			#pragma omp parallel default(shared)
			{
				int e;
				#pragma omp for
				for (c = 0; c < n; c += 16) {
					e = min(16, (int)n - c);
					posAll[0][c:e] = posAll[0][c:e] + currMov[0][c:e];
					posAll[1][c:e] = posAll[1][c:e] + currMov[1][c:e];
					posAll[2][c:e] = posAll[2][c:e] + currMov[2][c:e];
					// boundary conditions: cells can not move out of the cube [0,1]^3
					for (d = 0; d < 3; d++) {
						if (posAll[d][c:e] < 0) { posAll[d][c:e] = 0; }
						if (posAll[d][c:e] > 1) { posAll[d][c:e] = 1; }
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
