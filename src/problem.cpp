OptResult optimize(const VectorXd& start_x) {
  OptResult result;
  VectorXd curr_x = start_x;
  double trust_region_size = m_params.init_trust_region_size;

  vector<QuadExprPtr> quad_costs(m_costs.size(), NULL);
  QuadExprPtr trust_region_cost(new QuadExpr);
  for (int iter = 0; ; ++iter) {

    quad_costs.clear();
    for (int i = 0; i < m_costs.size(); ++i) {
      quad_costs[i] = m_costs[i].convex();
    }




    if (iter > m_params.max_iter) {
      blah
    }


  }
}






void optimize() {
  vector<string> cost_names = getCostNames(prob_->getCosts());
  vector<ConstraintPtr> constraints = prob_->getConstraints();
  vector<string> cnt_names = getCntNames(constraints);

  DblVec& x_ = results_.x; // just so I don't have to rewrite code
  if (x_.size() == 0) PRINT_AND_THROW("you forgot to initialize!");
  if (!prob_) PRINT_AND_THROW("you forgot to set the optimization problem");    
  
  x_ = prob_->getClosestFeasiblePoint(x_);

  assert(x_.size() == prob_->getVars().size());
  assert(prob_->getCosts().size() > 0 || constraints.size() > 0);

  OptStatus retval = INVALID;

  for (int merit_increases=0; merit_increases < max_merit_coeff_increases_; ++merit_increases) { /* merit adjustment loop */
    for (int iter=1; ; ++iter) { /* sqp loop */
      callCallbacks(x_);

      LOG_DEBUG("current iterate: %s", CSTR(x_));
      LOG_INFO("iteration %i", iter);

      // speedup: if you just evaluated the cost when doing the line search, use that
      if (results_.cost_vals.empty() && results_.cnt_viols.empty()) { //only happens on the first iteration
        results_.cnt_viols = evaluateConstraintViols(constraints, x_);
        results_.cost_vals = evaluateCosts(prob_->getCosts(), x_);
        assert(results_.n_func_evals == 0);
        ++results_.n_func_evals;
      }

      // DblVec new_cnt_viols = evaluateConstraintViols(constraints, x_);
      // DblVec new_cost_vals = evaluateCosts(prob_->getCosts(), x_);
      // cout << "costs" << endl;
      // for (int i=0; i < new_cnt_viols.size(); ++i) {
      //   cout << cnt_names[i] << " " << new_cnt_viols[i] - results_.cnt_viols[i] << endl;
      // }
      // for (int i=0; i < new_cost_vals.size(); ++i) {
      //   cout << cost_names[i] << " " << new_cost_vals[i] - results_.cost_vals[i] << endl;
      // }


      vector<ConvexObjectivePtr> cost_models = convexifyCosts(prob_->getCosts(),x_, model_.get());
      vector<ConvexConstraintsPtr> cnt_models = convexifyConstraints(constraints, x_, model_.get());
      vector<ConvexObjectivePtr> cnt_cost_models = cntsToCosts(cnt_models, merit_error_coeff_, model_.get());
      model_->update();
      BOOST_FOREACH(ConvexObjectivePtr& cost, cost_models)cost->addConstraintsToModel();
      BOOST_FOREACH(ConvexObjectivePtr& cost, cnt_cost_models)cost->addConstraintsToModel();
      model_->update();
      QuadExpr objective;
      BOOST_FOREACH(ConvexObjectivePtr& co, cost_models)exprInc(objective, co->quad_);
      BOOST_FOREACH(ConvexObjectivePtr& co, cnt_cost_models){
        exprInc(objective, co->quad_);
      }
//    objective = cleanupExpr(objective);
      model_->setObjective(objective);

//    if (logging::filter() >= IPI_LEVEL_DEBUG) {
//      DblVec model_cost_vals;
//      BOOST_FOREACH(ConvexObjectivePtr& cost, cost_models) {
//        model_cost_vals.push_back(cost->value(x));
//      }
//      LOG_DEBUG("model costs %s should equalcosts  %s", printer(model_cost_vals), printer(cost_vals));
//    }

      while (trust_box_size_ >= min_trust_box_size_) {

        setTrustBoxConstraints(x_);
        CvxOptStatus status = model_->optimize();
        ++results_.n_qp_solves;
        if (status != CVX_SOLVED) {
          LOG_ERROR("convex solver failed! set TRAJOPT_LOG_THRESH=DEBUG to see solver output. saving model to /tmp/fail.lp");
          model_->writeToFile("/tmp/fail.lp");
          retval = OPT_FAILED;
          goto cleanup;
        }
        DblVec model_var_vals = model_->getVarValues(model_->getVars());

        DblVec model_cost_vals = evaluateModelCosts(cost_models, model_var_vals);
        DblVec model_cnt_viols = evaluateModelCntViols(cnt_models, model_var_vals);

        // the n variables of the OptProb happen to be the first n variables in the Model
        DblVec new_x(model_var_vals.begin(), model_var_vals.begin() + x_.size());

        if (GetLogLevel() >= util::LevelDebug) {
          DblVec cnt_costs1 = evaluateModelCosts(cnt_cost_models, model_var_vals);
          DblVec cnt_costs2 = model_cnt_viols;
          for (int i=0; i < cnt_costs2.size(); ++i) cnt_costs2[i] *= merit_error_coeff_;
          LOG_DEBUG("SHOULD BE ALMOST THE SAME: %s ?= %s", CSTR(cnt_costs1), CSTR(cnt_costs2) );
          // not exactly the same because cnt_costs1 is based on aux variables, but they might not be at EXACTLY the right value
        }

        DblVec new_cost_vals = evaluateCosts(prob_->getCosts(), new_x);
        DblVec new_cnt_viols = evaluateConstraintViols(constraints, new_x);
        ++results_.n_func_evals;

        double old_merit = vecSum(results_.cost_vals) + merit_error_coeff_ * vecSum(results_.cnt_viols);
        double model_merit = vecSum(model_cost_vals) + merit_error_coeff_ * vecSum(model_cnt_viols);
        double new_merit = vecSum(new_cost_vals) + merit_error_coeff_ * vecSum(new_cnt_viols);
        double approx_merit_improve = old_merit - model_merit;
        double exact_merit_improve = old_merit - new_merit;
        double merit_improve_ratio = exact_merit_improve / approx_merit_improve;

        if (util::GetLogLevel() >= util::LevelInfo) {
          LOG_INFO(" ");
          printCostInfo(results_.cost_vals, model_cost_vals, new_cost_vals,
                        results_.cnt_viols, model_cnt_viols, new_cnt_viols, cost_names,
                        cnt_names, merit_error_coeff_);
          printf("%15s | %10.3e | %10.3e | %10.3e | %10.3e\n", "TOTAL", old_merit, approx_merit_improve, exact_merit_improve, merit_improve_ratio);
        }

        if (approx_merit_improve < -1e-5) {
          LOG_ERROR("approximate merit function got worse (%.3e). (convexification is probably wrong to zeroth order)", approx_merit_improve);
        }
        if (approx_merit_improve < min_approx_improve_) {
          LOG_INFO("converged because improvement was small (%.3e < %.3e)", approx_merit_improve, min_approx_improve_);
          retval = OPT_CONVERGED;
          goto penaltyadjustment;
        }
        if (approx_merit_improve / old_merit < min_approx_improve_frac_) {
          LOG_INFO(
              "converged because improvement ratio was small (%.3e < %.3e)",
              approx_merit_improve/old_merit, min_approx_improve_frac_);
          retval = OPT_CONVERGED;
          goto penaltyadjustment;
        } 
        else if (exact_merit_improve < 0 || merit_improve_ratio < improve_ratio_threshold_) {
          adjustTrustRegion(trust_shrink_ratio_);
          LOG_INFO("shrunk trust region. new box size: %.4f",
              trust_box_size_);
        } else {
          x_ = new_x;
          results_.cost_vals = new_cost_vals;
          results_.cnt_viols = new_cnt_viols;
          adjustTrustRegion(trust_expand_ratio_);
          LOG_INFO("expanded trust region. new box size: %.4f",trust_box_size_);
          break;
        }
      }

      if (trust_box_size_ < min_trust_box_size_) {
        LOG_INFO("converged because trust region is tiny");
        retval = OPT_CONVERGED;
        goto penaltyadjustment;
      } else if (iter >= max_iter_) {
        LOG_INFO("iteration limit");
        retval = OPT_SCO_ITERATION_LIMIT;
        goto cleanup;
      }
    }

    penaltyadjustment:
    if (results_.cnt_viols.empty() || vecMax(results_.cnt_viols) < cnt_tolerance_) {
      if (results_.cnt_viols.size() > 0) LOG_INFO("woo-hoo! constraints are satisfied (to tolerance %.2e)", cnt_tolerance_);
      goto cleanup;
    }
    else {
      LOG_INFO("not all constraints are satisfied. increasing penalties");
      merit_error_coeff_ *= merit_coeff_increase_ratio_;
      trust_box_size_ = fmax(trust_box_size_, min_trust_box_size_ / trust_shrink_ratio_ * 1.5);
    }



  }
  retval = OPT_PENALTY_ITERATION_LIMIT;
  LOG_INFO("optimization couldn't satisfy all constraints");


  cleanup:
  assert(retval != INVALID && "should never happen");
  results_.status = retval;
  results_.total_cost = vecSum(results_.cost_vals);
  LOG_INFO("\n==================\n%s==================", CSTR(results_));
  callCallbacks(x_);

  return retval;
}
