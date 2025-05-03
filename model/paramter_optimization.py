from utils.logger import Logger
from utils import config

from bayes_opt import BayesianOptimization
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error

logger = Logger().get_logger()

X_train = None
Y_train = None
X_test = None
Y_test = None


def clac_score(Y_Train, pred):
    return -root_mean_squared_error(Y_Train, pred)
    # return -mean_absolute_error(Y_Train,pred)


def xgb_model(n_estimators, max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree, reg_alpha, reg_lambda):
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "gamma": gamma,
        "min_child_weight": min_child_weight,
        "max_delta_step": max_delta_step,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
    }
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = clac_score(Y_test, pred)
    return score


def lgb_model(num_leaves, n_estimators, max_bin, bagging_fraction, bagging_freq, feature_fraction, min_data_in_leaf, min_sum_hessian_in_leaf):
    params = {
        "num_leaves": int(num_leaves),
        "n_estimators": int(n_estimators),
        "max_bin": int(max_bin),
        "bagging_fraction": bagging_fraction,
        "bagging_freq": int(bagging_freq),
        "feature_fraction": feature_fraction,
        "min_data_in_leaf": int(min_data_in_leaf),
        "min_sum_hessian_in_leaf": min_sum_hessian_in_leaf,
        "verbosity": -1,
        "metric": "l1",
        "device": "gpu",
    }
    model = lgb.LGBMRegressor(**params, random_state=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = clac_score(Y_test, pred)
    return score


def gb_model(n_estimators, max_depth, min_samples_leaf, min_samples_split):
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": "sqrt",
    }
    model = GradientBoostingRegressor(**params, random_state=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = clac_score(Y_test, pred)
    return score


def lasso_model(alpha, max_iter):
    params = {
        "alpha": alpha,
        "max_iter": int(max_iter),
    }
    model = make_pipeline(RobustScaler(), Lasso(**params, random_state=42))
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = clac_score(Y_test, pred)
    return score


def rf_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf):
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": min_weight_fraction_leaf,
        "max_features": "sqrt",
    }
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = clac_score(Y_test, pred)
    return score


def enet_model(alpha, l1_ratio, max_iter):
    params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "max_iter": int(max_iter),
    }
    model = make_pipeline(RobustScaler(), ElasticNet(**params, random_state=42))
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = clac_score(Y_test, pred)
    return score


def krr_model(alpha, degree, coef0):
    params = {
        "alpha": alpha,
        "degree": int(degree),
        "coef0": coef0,
        "kernel": "polynomial",
    }
    model = KernelRidge(**params)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = clac_score(Y_test, pred)
    return score


def cb_model(iterations, learning_rate, l2_leaf_reg, bagging_temperature, random_strength, depth, min_data_in_leaf):
    params = {
        "iterations": int(iterations),
        "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg,
        "bagging_temperature": bagging_temperature,
        "random_strength": random_strength,
        "depth": int(depth),
        "min_data_in_leaf": int(min_data_in_leaf),
        "loss_function": "MAPE",
        "custom_metric": "MAPE",
        "eval_metric": "MAPE",
        "task_type": "GPU",
        "verbose": False,
    }
    model = cb.CatBoostRegressor(**params, random_seed=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = clac_score(Y_test, pred)
    return score


def rr_model(alpha, tol):
    params = {
        "alpha": alpha,
        "tol": tol,
        "solver": "lbfgs",
        "positive": True,
    }
    model = Ridge(**params, random_state=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = clac_score(Y_test, pred)
    return score


def gp_model(alpha):
    params = {
        "alpha": alpha,
    }
    model = GaussianProcessRegressor(**params, random_state=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    score = clac_score(Y_test, pred)
    return score


def get_best_params(X1, Y1, X2, Y2):
    try:
        global X_train, Y_train, X_test, Y_test
        X_train = X1
        Y_train = Y1
        X_test = X2
        Y_test = Y2

        opt_model_dict = config.training["paramter_opt_model"]
        model_dict = {}

        # xgboost
        if opt_model_dict["XGBoost"]:
            xgb_opt = BayesianOptimization(
                xgb_model,
                {
                    "n_estimators": (500, 5000),
                    "max_depth": (2, 12),
                    "gamma": (0.001, 10.0),
                    "min_child_weight": (0, 20),
                    "max_delta_step": (0, 10),
                    "subsample": (0.4, 1.0),
                    "colsample_bytree": (0.4, 1.0),
                    "reg_alpha": (0.0, 1.5),
                    "reg_lambda": (0.0, 1.5),
                },
            )
            model_dict["XGBoost"] = xgb_opt

        # lgbm
        if opt_model_dict["LightGBM"]:
            lgb_opt = BayesianOptimization(
                lgb_model,
                {
                    "num_leaves": (5, 60),
                    "n_estimators": (500, 1500),
                    "max_bin": (5, 255),
                    "bagging_fraction": (0.01, 1.0),
                    "bagging_freq": (0, 10),
                    "feature_fraction": (0.01, 1.0),
                    "min_data_in_leaf": (1, 100),
                    "min_sum_hessian_in_leaf": (0.0, 50.0),
                },
            )
            model_dict["LightGBM"] = lgb_opt

        # gb
        if opt_model_dict["GradientBoost"]:
            gb_opt = BayesianOptimization(
                gb_model,
                {
                    "n_estimators": (500, 5000),
                    "max_depth": (1, 100),
                    "min_samples_leaf": (0.001, 0.5),
                    "min_samples_split": (0.0001, 1),
                },
            )
            model_dict["GradientBoost"] = gb_opt

        # lasso
        if opt_model_dict["Lasso"]:
            lasso_opt = BayesianOptimization(
                lasso_model,
                {
                    "alpha": (0.0, 50.0),
                    "max_iter": (800, 2000),
                },
            )
            model_dict["Lasso"] = lasso_opt

        # rf
        if opt_model_dict["RandomForest"]:
            rf_opt = BayesianOptimization(
                rf_model,
                {
                    "n_estimators": (500, 5000),
                    "max_depth": (1, 100),
                    "min_samples_split": (0.0001, 1),
                    "min_samples_leaf": (0.001, 0.5),
                    "min_weight_fraction_leaf": (0.0, 0.5),
                },
            )
            model_dict["RandomForest"] = rf_opt

        # enet
        if opt_model_dict["ElasticNet"]:
            enet_opt = BayesianOptimization(
                enet_model,
                {
                    "alpha": (0, 50),
                    "l1_ratio": (0, 1),
                    "max_iter": (800, 2000),
                },
            )
            model_dict["ElasticNet"] = enet_opt

        # krr
        if opt_model_dict["KernelRidge"]:
            krr_opt = BayesianOptimization(
                krr_model,
                {
                    "alpha": (0, 50),
                    "degree": (1, 20),
                    "coef0": (1, 20),
                },
            )
            model_dict["KernelRidge"] = krr_opt

        # cb
        if opt_model_dict["CatBoost"]:
            cb_opt = BayesianOptimization(
                cb_model,
                {
                    "iterations": (500, 5000),
                    "learning_rate": (0.001, 0.3),
                    "l2_leaf_reg": (0, 50),
                    "bagging_temperature": (0, 50),
                    "random_strength": (0, 50),
                    "depth": (1, 12),
                    "min_data_in_leaf": (1, 100),
                },
            )
            model_dict["CatBoost"] = cb_opt

        # rr
        if opt_model_dict["Ridge"]:
            rr_opt = BayesianOptimization(
                rr_model,
                {
                    "alpha": (0, 50),
                    "tol": (0.00001, 0.5),
                },
            )
            model_dict["Ridge"] = rr_opt

        model_dict_keys = list(model_dict.keys())
        best_params = {}
        for key in model_dict_keys:
            model = model_dict[key]
            model._verbose = 2
            model.maximize(init_points=config.training["paramter_init_points"], n_iter=config.training["paramter_n_iter"])
            best_params[key] = model.max

        return best_params

    except Exception as e:
        logger.error(f"Error get_best_params: {str(e)}")
