#pragma once
#include "util/NumType.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

class ScaleAccumulator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Mat22f H;
  Vec2f b;
  size_t num;

  inline void initialize() {
    H.setZero();
    b.setZero();
    memset(SSEData, 0, sizeof(float) * 4 * 3);
    memset(SSEData1k, 0, sizeof(float) * 4 * 3);
    memset(SSEData1m, 0, sizeof(float) * 4 * 3);
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish() {
    H.setZero();
    shiftUp(true);
    assert(numIn1 == 0);
    assert(numIn1k == 0);

    int idx = 0;
    for (int r = 0; r < 2; r++)
      for (int c = r; c < 2; c++) {
        float d = SSEData1m[idx + 0] + SSEData1m[idx + 1] + SSEData1m[idx + 2] +
                  SSEData1m[idx + 3];
        H(r, c) = H(c, r) = d;
        idx += 4;
      }
    assert(idx == 4 * 3);
  }

  inline void updateSSE_oneed(const __m128 J0, const __m128 J1,
                              const __m128 w) {
    float *pt = SSEData;

    __m128 J0w = _mm_mul_ps(J0, w);
    _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J0)));
    pt += 4;
    _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J1)));
    pt += 4;

    __m128 J1w = _mm_mul_ps(J1, w);
    _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J1)));
    pt += 4;

    num += 4;
    numIn1++;
    shiftUp(false);
  }

private:
  EIGEN_ALIGN16 float SSEData[4 * 3];
  EIGEN_ALIGN16 float SSEData1k[4 * 3];
  EIGEN_ALIGN16 float SSEData1m[4 * 3];
  float numIn1, numIn1k, numIn1m;

  void shiftUp(bool force) {
    if (numIn1 > 1000 || force) {
      for (int i = 0; i < 3; i++)
        _mm_store_ps(SSEData1k + 4 * i,
                     _mm_add_ps(_mm_load_ps(SSEData + 4 * i),
                                _mm_load_ps(SSEData1k + 4 * i)));
      numIn1k += numIn1;
      numIn1 = 0;
      memset(SSEData, 0, sizeof(float) * 4 * 3);
    }

    if (numIn1k > 1000 || force) {
      for (int i = 0; i < 3; i++)
        _mm_store_ps(SSEData1m + 4 * i,
                     _mm_add_ps(_mm_load_ps(SSEData1k + 4 * i),
                                _mm_load_ps(SSEData1m + 4 * i)));
      numIn1m += numIn1k;
      numIn1k = 0;
      memset(SSEData1k, 0, sizeof(float) * 4 * 3);
    }
  }
};

} // namespace dso
