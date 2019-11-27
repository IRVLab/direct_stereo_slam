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

#include "util/settings.h"
#include <stdio.h>

//#include <GL/glx.h>
//#include <GL/gl.h>
//#include <GL/glu.h>

#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "KeyFrameDisplay.h"
#include "util/FrameShell.h"
#include <pangolin/pangolin.h>

namespace dso {
namespace IOWrap {

KeyFrameDisplay::KeyFrameDisplay() {
  original_input_sparse_ = 0;
  num_sparse_buffer_size_ = 0;
  num_sparse_points_ = 0;

  id_ = 0;
  active_ = true;
  tfm_c_w_ = SE3();

  need_refresh_ = true;

  my_scaled_th_ = 1e10;
  my_abs_th_ = 1e10;
  my_displayMode_ = 1;
  my_min_rel_bs_ = 0;
  my_sparsify_factor_ = 1;

  num_gl_buffer_points_ = 0;
  buffer_valid_ = false;
}
void KeyFrameDisplay::setFromF(FrameShell *frame, CalibHessian *HCalib) {
  id_ = frame->id;
  fx_ = HCalib->fxl();
  fy_ = HCalib->fyl();
  cx_ = HCalib->cxl();
  cy_ = HCalib->cyl();
  width_ = wG[0];
  height_ = hG[0];
  fxi_ = 1 / fx_;
  fyi_ = 1 / fy_;
  cxi_ = -cx_ / fx_;
  cyi_ = -cy_ / fy_;
  tfm_c_w_ = frame->camToWorld;
  need_refresh_ = true;
}

void KeyFrameDisplay::setFromKF(FrameHessian *fh, CalibHessian *HCalib) {
  setFromF(fh->shell, HCalib);

  // add all traces, inlier and outlier points.
  int npoints = fh->immaturePoints.size() + fh->pointHessians.size() +
                fh->pointHessiansMarginalized.size() +
                fh->pointHessiansOut.size();

  if (num_sparse_buffer_size_ < npoints) {
    if (original_input_sparse_ != 0)
      delete original_input_sparse_;
    num_sparse_buffer_size_ = npoints + 100;
    original_input_sparse_ =
        new InputPointSparse<MAX_RES_PER_POINT>[num_sparse_buffer_size_];
  }

  InputPointSparse<MAX_RES_PER_POINT> *pc = original_input_sparse_;
  num_sparse_points_ = 0;
  for (ImmaturePoint *p : fh->immaturePoints) {
    for (int i = 0; i < patternNum; i++)
      pc[num_sparse_points_].color[i] = p->color[i];

    pc[num_sparse_points_].u = p->u;
    pc[num_sparse_points_].v = p->v;
    pc[num_sparse_points_].idpeth = (p->idepth_max + p->idepth_min) * 0.5f;
    pc[num_sparse_points_].idepth_hessian = 1000;
    pc[num_sparse_points_].relObsBaseline = 0;
    pc[num_sparse_points_].numGoodRes = 1;
    pc[num_sparse_points_].status = 0;
    num_sparse_points_++;
  }

  for (PointHessian *p : fh->pointHessians) {
    for (int i = 0; i < patternNum; i++)
      pc[num_sparse_points_].color[i] = p->color[i];
    pc[num_sparse_points_].u = p->u;
    pc[num_sparse_points_].v = p->v;
    pc[num_sparse_points_].idpeth = p->idepth_scaled;
    pc[num_sparse_points_].relObsBaseline = p->maxRelBaseline;
    pc[num_sparse_points_].idepth_hessian = p->idepth_hessian;
    pc[num_sparse_points_].numGoodRes = 0;
    pc[num_sparse_points_].status = 1;

    num_sparse_points_++;
  }

  for (PointHessian *p : fh->pointHessiansMarginalized) {
    for (int i = 0; i < patternNum; i++)
      pc[num_sparse_points_].color[i] = p->color[i];
    pc[num_sparse_points_].u = p->u;
    pc[num_sparse_points_].v = p->v;
    pc[num_sparse_points_].idpeth = p->idepth_scaled;
    pc[num_sparse_points_].relObsBaseline = p->maxRelBaseline;
    pc[num_sparse_points_].idepth_hessian = p->idepth_hessian;
    pc[num_sparse_points_].numGoodRes = 0;
    pc[num_sparse_points_].status = 2;
    num_sparse_points_++;
  }

  for (PointHessian *p : fh->pointHessiansOut) {
    for (int i = 0; i < patternNum; i++)
      pc[num_sparse_points_].color[i] = p->color[i];
    pc[num_sparse_points_].u = p->u;
    pc[num_sparse_points_].v = p->v;
    pc[num_sparse_points_].idpeth = p->idepth_scaled;
    pc[num_sparse_points_].relObsBaseline = p->maxRelBaseline;
    pc[num_sparse_points_].idepth_hessian = p->idepth_hessian;
    pc[num_sparse_points_].numGoodRes = 0;
    pc[num_sparse_points_].status = 3;
    num_sparse_points_++;
  }
  assert(num_sparse_points_ <= npoints);

  tfm_c_w_ = fh->PRE_camToWorld;
  need_refresh_ = true;
}

KeyFrameDisplay::~KeyFrameDisplay() {
  if (original_input_sparse_ != 0)
    delete[] original_input_sparse_;
}

bool KeyFrameDisplay::refreshPC(bool canRefresh, float scaledTH, float absTH,
                                int mode, float minBS, int sparsity) {
  if (canRefresh) {
    need_refresh_ = need_refresh_ || my_scaled_th_ != scaledTH ||
                    my_abs_th_ != absTH || my_displayMode_ != mode ||
                    my_min_rel_bs_ != minBS || my_sparsify_factor_ != sparsity;
  }

  if (!need_refresh_)
    return false;
  need_refresh_ = false;

  my_scaled_th_ = scaledTH;
  my_abs_th_ = absTH;
  my_displayMode_ = mode;
  my_min_rel_bs_ = minBS;
  my_sparsify_factor_ = sparsity;

  // if there are no vertices, done!
  if (num_sparse_points_ == 0)
    return false;

  // make data
  Vec3f *tmpVertexBuffer = new Vec3f[num_sparse_points_ * patternNum];
  Vec3b *tmpColorBuffer = new Vec3b[num_sparse_points_ * patternNum];
  int vertexBufferNumPoints = 0;

  for (int i = 0; i < num_sparse_points_; i++) {
    /* display modes:
     * my_displayMode_==0 - all pts, color-coded
     * my_displayMode_==1 - normal points
     * my_displayMode_==2 - active only
     * my_displayMode_==3 - nothing
     */

    if (my_displayMode_ == 1 && original_input_sparse_[i].status != 1 &&
        original_input_sparse_[i].status != 2)
      continue;
    if (my_displayMode_ == 2 && original_input_sparse_[i].status != 1)
      continue;
    if (my_displayMode_ > 2)
      continue;

    if (original_input_sparse_[i].idpeth < 0)
      continue;

    float depth = 1.0f / original_input_sparse_[i].idpeth;
    float depth4 = depth * depth;
    depth4 *= depth4;
    float var = (1.0f / (original_input_sparse_[i].idepth_hessian + 0.01));

    if (var * depth4 > my_scaled_th_)
      continue;

    if (var > my_abs_th_)
      continue;

    if (original_input_sparse_[i].relObsBaseline < my_min_rel_bs_)
      continue;

    for (int pnt = 0; pnt < patternNum; pnt++) {

      if (my_sparsify_factor_ > 1 && rand() % my_sparsify_factor_ != 0)
        continue;
      int dx = patternP[pnt][0];
      int dy = patternP[pnt][1];

      tmpVertexBuffer[vertexBufferNumPoints][0] =
          ((original_input_sparse_[i].u + dx) * fxi_ + cxi_) * depth;
      tmpVertexBuffer[vertexBufferNumPoints][1] =
          ((original_input_sparse_[i].v + dy) * fyi_ + cyi_) * depth;
      tmpVertexBuffer[vertexBufferNumPoints][2] =
          depth * (1 + 2 * fxi_ * (rand() / (float)RAND_MAX - 0.5f));

      if (my_displayMode_ == 0) {
        if (original_input_sparse_[i].status == 0) {
          tmpColorBuffer[vertexBufferNumPoints][0] = 0;
          tmpColorBuffer[vertexBufferNumPoints][1] = 255;
          tmpColorBuffer[vertexBufferNumPoints][2] = 255;
        } else if (original_input_sparse_[i].status == 1) {
          tmpColorBuffer[vertexBufferNumPoints][0] = 0;
          tmpColorBuffer[vertexBufferNumPoints][1] = 255;
          tmpColorBuffer[vertexBufferNumPoints][2] = 0;
        } else if (original_input_sparse_[i].status == 2) {
          tmpColorBuffer[vertexBufferNumPoints][0] = 0;
          tmpColorBuffer[vertexBufferNumPoints][1] = 0;
          tmpColorBuffer[vertexBufferNumPoints][2] = 255;
        } else if (original_input_sparse_[i].status == 3) {
          tmpColorBuffer[vertexBufferNumPoints][0] = 255;
          tmpColorBuffer[vertexBufferNumPoints][1] = 0;
          tmpColorBuffer[vertexBufferNumPoints][2] = 0;
        } else {
          tmpColorBuffer[vertexBufferNumPoints][0] = 255;
          tmpColorBuffer[vertexBufferNumPoints][1] = 255;
          tmpColorBuffer[vertexBufferNumPoints][2] = 255;
        }

      } else {
        tmpColorBuffer[vertexBufferNumPoints][0] =
            original_input_sparse_[i].color[pnt];
        tmpColorBuffer[vertexBufferNumPoints][1] =
            original_input_sparse_[i].color[pnt];
        tmpColorBuffer[vertexBufferNumPoints][2] =
            original_input_sparse_[i].color[pnt];
      }
      vertexBufferNumPoints++;

      assert(vertexBufferNumPoints <= num_sparse_points_ * patternNum);
    }
  }

  if (vertexBufferNumPoints == 0) {
    delete[] tmpColorBuffer;
    delete[] tmpVertexBuffer;
    return true;
  }

  num_gl_buffer_good_points_ = vertexBufferNumPoints;
  if (num_gl_buffer_good_points_ > num_gl_buffer_points_) {
    num_gl_buffer_points_ = vertexBufferNumPoints * 1.3;
    vertex_buffer_.Reinitialise(pangolin::GlArrayBuffer, num_gl_buffer_points_,
                                GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    color_buffer_.Reinitialise(pangolin::GlArrayBuffer, num_gl_buffer_points_,
                               GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
  }
  vertex_buffer_.Upload(tmpVertexBuffer,
                        sizeof(float) * 3 * num_gl_buffer_good_points_, 0);
  color_buffer_.Upload(tmpColorBuffer,
                       sizeof(unsigned char) * 3 * num_gl_buffer_good_points_,
                       0);
  buffer_valid_ = true;
  delete[] tmpColorBuffer;
  delete[] tmpVertexBuffer;

  return true;
}

void KeyFrameDisplay::drawCam(float lineWidth, float *color, float sizeFactor) {
  if (width_ == 0)
    return;

  float sz = sizeFactor;

  glPushMatrix();

  Sophus::Matrix4f m = tfm_c_w_.matrix().cast<float>();
  glMultMatrixf((GLfloat *)m.data());

  if (color == 0) {
    glColor3f(1, 0, 0);
  } else
    glColor3f(color[0], color[1], color[2]);

  glLineWidth(lineWidth);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (0 - cx_) / fx_, sz * (0 - cy_) / fy_, sz);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (0 - cx_) / fx_, sz * (height_ - 1 - cy_) / fy_, sz);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (width_ - 1 - cx_) / fx_, sz * (height_ - 1 - cy_) / fy_, sz);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (width_ - 1 - cx_) / fx_, sz * (0 - cy_) / fy_, sz);

  glVertex3f(sz * (width_ - 1 - cx_) / fx_, sz * (0 - cy_) / fy_, sz);
  glVertex3f(sz * (width_ - 1 - cx_) / fx_, sz * (height_ - 1 - cy_) / fy_, sz);

  glVertex3f(sz * (width_ - 1 - cx_) / fx_, sz * (height_ - 1 - cy_) / fy_, sz);
  glVertex3f(sz * (0 - cx_) / fx_, sz * (height_ - 1 - cy_) / fy_, sz);

  glVertex3f(sz * (0 - cx_) / fx_, sz * (height_ - 1 - cy_) / fy_, sz);
  glVertex3f(sz * (0 - cx_) / fx_, sz * (0 - cy_) / fy_, sz);

  glVertex3f(sz * (0 - cx_) / fx_, sz * (0 - cy_) / fy_, sz);
  glVertex3f(sz * (width_ - 1 - cx_) / fx_, sz * (0 - cy_) / fy_, sz);

  glEnd();
  glPopMatrix();
}

void KeyFrameDisplay::drawPC(float pointSize) {

  if (!buffer_valid_ || num_gl_buffer_good_points_ == 0)
    return;

  glDisable(GL_LIGHTING);

  glPushMatrix();

  Sophus::Matrix4f m = tfm_c_w_.matrix().cast<float>();
  glMultMatrixf((GLfloat *)m.data());

  glPointSize(pointSize);

  color_buffer_.Bind();
  glColorPointer(color_buffer_.count_per_element, color_buffer_.datatype, 0, 0);
  glEnableClientState(GL_COLOR_ARRAY);

  vertex_buffer_.Bind();
  glVertexPointer(vertex_buffer_.count_per_element, vertex_buffer_.datatype, 0,
                  0);
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, num_gl_buffer_good_points_);
  glDisableClientState(GL_VERTEX_ARRAY);
  vertex_buffer_.Unbind();

  glDisableClientState(GL_COLOR_ARRAY);
  color_buffer_.Unbind();

  glPopMatrix();
}

} // namespace IOWrap
} // namespace dso
