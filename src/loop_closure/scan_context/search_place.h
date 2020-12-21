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

#pragma once
#include <Eigen/Core>

#include <vector>

#define FLANN_NN 3
#define LOOP_MARGIN 200
#define RINGKEY_THRES 0.1

inline void search_ringkey(const flann::Matrix<float> &ringkey,
                           flann::Index<flann::L2<float>> *ringkeys,
                           std::vector<int> &candidates) {
  // query ringkey
  if (ringkeys->size() > FLANN_NN) {
    flann::Matrix<int> idces(new int[FLANN_NN], 1, FLANN_NN);
    flann::Matrix<float> dists(new float[FLANN_NN], 1, FLANN_NN);
    ringkeys->knnSearch(ringkey, idces, dists, FLANN_NN,
                        flann::SearchParams(128));
    for (int i = 0; i < FLANN_NN; i++) {
      if (dists[0][i] < RINGKEY_THRES && idces[0][i] > 0) {
        candidates.emplace_back(idces[0][i] - 1);
      }
    }
  }

  // store ringkey in waiting queue of size LOOP_MARGIN
  int r_cols = ringkey.cols;
  static int queue_idx = 0;
  static flann::Matrix<float> ringkey_queue(new float[LOOP_MARGIN * r_cols],
                                            LOOP_MARGIN, r_cols);
  if (queue_idx >= LOOP_MARGIN) {
    flann::Matrix<float> ringkey_to_add(new float[r_cols], 1, r_cols);
    for (int j = 0; j < r_cols; j++) {
      ringkey_to_add[0][j] = ringkey_queue[queue_idx % LOOP_MARGIN][j];
    }
    ringkeys->addPoints(ringkey_to_add);
  }
  for (int j = 0; j < r_cols; j++) {
    ringkey_queue[queue_idx % LOOP_MARGIN][j] = ringkey[0][j];
  }
  queue_idx++;
}

inline void search_sc(SigType &signature,
                      const std::vector<dso::LoopFrame *> &loop_frames,
                      const std::vector<int> &candidates, int sc_width,
                      int &res_idx, float &res_diff) {
  res_idx = candidates[0];
  res_diff = 1.1;
  for (auto &candidate : candidates) {
    // Compute difference by Euclidean distance
    float cur_prod = 0;
    size_t m(0), n(0);
    SigType &candi_sig = loop_frames[candidate]->signature;
    // siganture is a sparse vector <index, value>
    while (m < signature.size() && n < candi_sig.size()) {
      if (signature[m].first == candi_sig[n].first) {
        cur_prod += signature[m++].second * candi_sig[n++].second;
      } else {
        signature[m].first < candi_sig[n].first ? m++ : n++;
      }
    }

    float cur_diff = (1 - cur_prod / sc_width) / 2.0;
    if (res_diff > cur_diff) {
      res_idx = candidate;
      res_diff = cur_diff;
    }
  }
}