#ifndef __RIGID_TRACKING_PROBLEM__
#define __RIGID_TRACKING_PROBLEM__

/**************************
 * Solve the rigid motion *
 * tracking problem       *
 *************************/
#include "common.hpp"
#include "expr.hpp"
#include "numpy_utils.hpp"
#include "optimizer.hpp"
#include "grid.hpp"
#include "grid_numpy_utils.hpp"
#include <boost/bind.hpp>
#include <math.h>

struct RigidObservationZeroCrossingCost : public CostFunc {
  /***
   * Logic : this cost is to solve for a transformation [x,y,theta],
   *         which must be applied to the last TSDF so that it becomes consistent
   *         with the current observation.
   *
   *   		 The current observation is a list of 2D grid points, where we expect a surface
   *   		 i.e., the zero-crossings should lie at these points.
   *
   *   		 So while iterating, we apply the inverse transform to the observation points
   *   		 to bring them in the frame of the last TSDF, get the TSDF values there and
   *   		 iterate. Upon solving, we apply the new found transformation [x,y,theta] to
   *   		 the old TSDF to get the new updated TSDF.
   *
   *
   *   		 ***CURRENTLY THIS COST IMPLEMENTS solving only for [x,y]***
   *
   ***/
	const GridParams m_gp;   // TSDF grid parameters
	RowVector2d m_grid_center;
	const DoubleField m_phi; // last iterate values of TSDF
	MatrixX2d m_zero_points;  // points where phi should be zero : these come from camera observations
  Var m_dx, m_dy, m_dth; // optimization variables



  RigidObservationZeroCrossingCost(const DoubleField phi,
                                       const Var &dx,
                                       const Var &dy,
                                       const Var &dth)
    : m_phi(phi), m_gp(phi.grid_params()), m_dx(dx), m_dy(dy), m_dth(dth)
  {
    m_grid_center << (m_gp.xmax - m_gp.xmin) / 2., (m_gp.ymax - m_gp.ymin) / 2.;
  }

  string name() const { return "rigid_obs_zc"; }
  int num_residuals() const { return m_zero_points.rows(); }
  bool is_linear() const { return false; }


  void set_zero_points(const MatrixX2d& zero_points) {
    m_zero_points = zero_points;
  }

  void py_set_zero_points(py::object py_zero_points) {
    util::fromNdarray(py_zero_points, m_zero_points);
  }


  void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    /**
     * returns a vector of distance of the observations points
     * in the transformed sdf.
     *
     * Note : if the transformed point lies outside the tracked grid,
     *        the residual returned is zero.
     **/
    assert(m_zero_points.rows() > 0);
    MatrixX2d tf_zero_points = transform_zero_points(x);

    for (int i = 0; i < tf_zero_points.rows(); ++i) {
      if (is_in_grid(tf_zero_points.row(i))) {
          out(i) = m_phi.eval_xy(tf_zero_points(i,0), tf_zero_points(i,1));
      } else {
          out(i) = 0;
      }
    }
  }

  void linearize(const VectorXd& x, CostFuncLinearization& lin) {
    /**
     * Returns a linearization for each of the residual terms.
     **/
    assert(m_zero_points.rows() > 0);
    MatrixX2d tf_zero_points = transform_zero_points(x);
    VectorXd grad_th = grad_theta(x);

    for (int i = 0; i < m_zero_points.rows(); ++i) {
      if (is_in_grid(tf_zero_points.row(i))) {
        DoubleField::ExprVec J_xy = m_phi.grad_xy(tf_zero_points(i,0), tf_zero_points(i,1));
        double f0   = m_phi.eval_xy(tf_zero_points(i,0), tf_zero_points(i,1));
        lin.set_by_expr(i, AffExpr(f0)
                           + J_xy.x/m_gp.eps_x*(m_dx - x(0))
                           + J_xy.y/m_gp.eps_y*(m_dy- x(1))
                           + grad_th(i)*(m_dth - x(2)));
      } else {
        lin.set_by_expr(i, AffExpr(0));
      }
    }
  }

private:

  MatrixX2d transform_zero_points(const VectorXd& tf_x) const {
    /**
     * Rotate and translate observation points.
     */
    double delta_th = tf_x(2);
    RowVector2d delta_xy = tf_x.head(2);

    MatrixX2d transformed_zero_points;
    rotate_points(delta_th, transformed_zero_points);
    transformed_zero_points.rowwise() += delta_xy;

    return transformed_zero_points;
  }

  bool is_in_grid(const Vector2d &pt) const {
    return
        m_gp.xmin <= pt.x() &&
        m_gp.xmax >= pt.x() &&
        m_gp.ymin <= pt.y() &&
        m_gp.ymax >= pt.y();
  }

  void rotate_points(double theta, MatrixX2d &out_pts) const {
    /**Rotate the obseration points by THETA and store in OUT_PTS.
     * Center of rotation is at the center of the grid.*/
    Matrix2d rot;
    rot << cos(theta), sin(theta),
          -sin(theta), cos(theta);

    out_pts = m_zero_points;
    out_pts.rowwise() -= m_grid_center;
    out_pts = out_pts*rot.transpose();
    out_pts.rowwise() += m_grid_center;
  }

  VectorXd grad_theta(const VectorXd& x) const {
    /**
    * Calculates the gradient wrt rotation angle theta at the observation pts.
    * Center of rotation is at the center of the grid.
    **/
    static const double delta = 1e-5;
    VectorXd p_x = x; p_x(2) += delta;
    VectorXd n_x = x; n_x(2) -= delta;

    MatrixX2d p_dth_pts = transform_zero_points(p_x);
    MatrixX2d n_dth_pts = transform_zero_points(n_x);

    VectorXd g_th(num_residuals());
    for (int i=0; i < m_zero_points.rows(); ++i) {
      g_th(i) = ( m_phi.eval_xy(p_dth_pts(i,0), p_dth_pts(i,1)) - m_phi.eval_xy(n_dth_pts(i,0), n_dth_pts(i,1)) ) / (2.*delta);
    }
    return g_th;
  }

};
typedef boost::shared_ptr<RigidObservationZeroCrossingCost> RigidObservationZeroCrossingCostPtr;

#endif
