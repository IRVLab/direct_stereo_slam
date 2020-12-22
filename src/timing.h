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

#ifndef TIMING_H
#define TIMING_H

#include <chrono>

inline double duration(const std::chrono::steady_clock::time_point &t0,
                       const std::chrono::steady_clock::time_point &t1) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
      .count();
}

template <typename T> inline T average(const std::vector<T> &vec) {
  T sum = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    sum += vec[i];
  }
  return sum / vec.size();
}

#endif