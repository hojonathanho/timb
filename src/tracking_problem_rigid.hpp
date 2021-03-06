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
   ***/
	const GridParams m_gp;   // TSDF grid parameters
	RowVector2d m_grid_center;
	const DoubleField m_phi; // last iterate values of TSDF
	const DoubleField m_weight; // last iterate weights of TSDF
	const double m_eps_weight;

	MatrixX2d m_zero_points;  // points where phi should be zero : these come from camera observations
  Var m_dx, m_dy, m_dth; // optimization variables



  RigidObservationZeroCrossingCost(const DoubleField phi,
                                       const DoubleField weight,
                                       const Var &dx,
                                       const Var &dy,
                                       const Var &dth)
    : m_phi(phi), m_weight(weight), m_gp(phi.grid_params()),
      m_dx(dx), m_dy(dy), m_dth(dth),
      m_eps_weight(1e-2)
  {
    m_grid_center << (m_gp.xmax + m_gp.xmin) / 2., (m_gp.ymax + m_gp.ymin) / 2.;
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
      if (is_in_grid(tf_zero_points.row(i)) and m_weight.eval_xy(tf_zero_points(i,0), tf_zero_points(i,1)) > m_eps_weight) {
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
      if (is_in_grid(tf_zero_points.row(i)) and m_weight.eval_xy(tf_zero_points(i,0), tf_zero_points(i,1)) > m_eps_weight) {
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
    /** Rotate and translate observation points.*/
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
    rot << cos(theta), -sin(theta),
           sin(theta), cos(theta);

    out_pts = m_zero_points;
    out_pts.rowwise() -= m_grid_center;
    out_pts = out_pts*rot.transpose();
    out_pts.rowwise() += m_grid_center;
  }

  VectorXd grad_theta(const VectorXd& x) const {
    /**
    * Calculates the gradient wrt rotation angle theta at the observation pts.
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


struct DisplacementCost : public CostFunc {
  Var m_dx, m_dy, m_dth; // optimization variables
  const double w_x, w_y, w_th;
  DisplacementCost(const Var &dx, const Var &dy,const Var &dth)
    : m_dx(dx), m_dy(dy), m_dth(dth), w_x(1), w_y(1), w_th(1) {
  }

  string name() const { return "dx_norm_c"; }
  int num_residuals() const { return 3;}
  bool is_linear() const { return true; }


  void eval(const VectorXd& x, Eigen::Ref<VectorXd> out) {
    out(0) = w_x*x(0);
    out(1) = w_y*x(1);
    out(2) = w_th*x(2);
  }

  void linearize(const VectorXd& x, CostFuncLinearization& lin) {
    lin.set_by_expr(0, w_x*m_dx);
    lin.set_by_expr(1, w_y*m_dy);
    lin.set_by_expr(2, w_th*m_dth);
  }
};
typedef boost::shared_ptr<DisplacementCost> DisplacementCostPtr;


DoubleField apply_rigid_transform(const DoubleField &phi, double dx, double dy, double dth) {
  /** returns a new sdf after applying a rigid transform [dX,dY, dTH] on the old sdf PHI. */
  Matrix2d rot;
  rot << cos(dth), -sin(dth),
         sin(dth), cos(dth);

  RowVector2d grid_center;

  grid_center << (phi.grid_params().xmax + phi.grid_params().xmin) / 2., (phi.grid_params().ymax + phi.grid_params().ymin) / 2.;

  DoubleField new_phi(phi.grid_params());
  for (int i=0; i < phi.grid_params().nx; i++)
    for (int j=0; j < phi.grid_params().ny; j++) {
      std::pair<double, double> xy = phi.grid_params().to_xy(i,j);
      Vector2d vec_xy; vec_xy << xy.first, xy.second;
      vec_xy -= grid_center;
      vec_xy = rot*vec_xy;
      vec_xy += grid_center;
      vec_xy += Vector2d(dx,dy);
      new_phi(i,j) = phi.eval_xy(vec_xy.x(), vec_xy.y());
    }
  return new_phi;
}

#endif
