// Copyright (C) <2020> <Jiawei Mo, Junaed Sattar>

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// This file is modified from <https://github.com/JakobEngel/dso>

#pragma once
#include "util/NumType.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

class ScaleAccumulator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Mat22f hessian_;
  Vec2f b_;
  size_t num_;

  inline void initialize() {
    hessian_.setZero();
    b_.setZero();
    memset(sse_data_, 0, sizeof(float) * 4 * 3);
    memset(sse_data_1k_, 0, sizeof(float) * 4 * 3);
    memset(sse_data_1m_, 0, sizeof(float) * 4 * 3);
    num_ = num_in_1_ = num_in_1k_ = num_in_1m_ = 0;
  }

  inline void finish() {
    hessian_.setZero();
    shiftUp(true);
    assert(num_in_1_ == 0);
    assert(num_in_1k_ == 0);

    int idx = 0;
    for (int r = 0; r < 2; r++)
      for (int c = r; c < 2; c++) {
        float d = sse_data_1m_[idx + 0] + sse_data_1m_[idx + 1] +
                  sse_data_1m_[idx + 2] + sse_data_1m_[idx + 3];
        hessian_(r, c) = hessian_(c, r) = d;
        idx += 4;
      }
    assert(idx == 4 * 3);
  }

  inline void updateSSE_oneed(const __m128 J0, const __m128 J1,
                              const __m128 w) {
    float *pt = sse_data_;

    __m128 J0w = _mm_mul_ps(J0, w);
    _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J0)));
    pt += 4;
    _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J1)));
    pt += 4;

    __m128 J1w = _mm_mul_ps(J1, w);
    _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J1)));
    pt += 4;

    num_ += 4;
    num_in_1_++;
    shiftUp(false);
  }

private:
  EIGEN_ALIGN16 float sse_data_[4 * 3];
  EIGEN_ALIGN16 float sse_data_1k_[4 * 3];
  EIGEN_ALIGN16 float sse_data_1m_[4 * 3];
  float num_in_1_, num_in_1k_, num_in_1m_;

  void shiftUp(bool force) {
    if (num_in_1_ > 1000 || force) {
      for (int i = 0; i < 3; i++)
        _mm_store_ps(sse_data_1k_ + 4 * i,
                     _mm_add_ps(_mm_load_ps(sse_data_ + 4 * i),
                                _mm_load_ps(sse_data_1k_ + 4 * i)));
      num_in_1k_ += num_in_1_;
      num_in_1_ = 0;
      memset(sse_data_, 0, sizeof(float) * 4 * 3);
    }

    if (num_in_1k_ > 1000 || force) {
      for (int i = 0; i < 3; i++)
        _mm_store_ps(sse_data_1m_ + 4 * i,
                     _mm_add_ps(_mm_load_ps(sse_data_1k_ + 4 * i),
                                _mm_load_ps(sse_data_1m_ + 4 * i)));
      num_in_1m_ += num_in_1k_;
      num_in_1k_ = 0;
      memset(sse_data_1k_, 0, sizeof(float) * 4 * 3);
    }
  }
};

} // namespace dso
