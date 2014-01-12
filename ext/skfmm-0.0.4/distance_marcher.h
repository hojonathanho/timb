//distance_marcher.h
#pragma once
#include "base_marcher.h"

class distanceMarcher : public baseMarcher
{
public:
  distanceMarcher(double *phi,      double *dx, long *flag, long *ignore_mask,
                  double *distance, int ndim,   int *shape,
                  bool self_test,   int order) :
    baseMarcher(phi, dx, flag, distance, ndim, shape, self_test, order), ignore_mask_(ignore_mask) { }
  virtual ~distanceMarcher() { }

protected:
  virtual double           solveQuadratic(int i, const double &a,
                                          const double &b, double &c);

  virtual void             initalizeFrozen();
  virtual double           updatePointOrderOne(int i);
  virtual double           updatePointOrderTwo(int i);

  long *ignore_mask_;
};
