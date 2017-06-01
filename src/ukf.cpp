#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  // initial state vector
  x_ = VectorXd(5);
  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.57;
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;
  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  // State dimension
  n_x_ = x_.size();

  n_aug_ = n_x_ + 2;

  n_sig_ = 2 * n_aug_ + 1;

  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(n_sig_);
  
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0,std_radrd_ * std_radrd_;
  
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_,0,
              0,std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

/**
 *  Convert coordinates from polar to cartesian
 */
VectorXd UKF::InitializeRadarMeasurement(MeasurementPackage measurement_pack) {
  float rho = measurement_pack.raw_measurements_[0];
  float phi = measurement_pack.raw_measurements_[1];
  float rho_dot = measurement_pack.raw_measurements_[2];
  float px = rho * cos(phi);
  float py = rho * sin(phi);
  float vx = rho_dot * cos(phi);
  float vy = rho_dot * sin(phi);
  float v  = sqrt(vx * vx + vy * vy);
  x_ << px, py, v, 0, 0;
  return x_;
}

/**
 *  Convert coordinates from polar to cartesian
 */
VectorXd UKF::InitializeLaserMeasurement(MeasurementPackage measurement_pack) {
  x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0;
  
  if (fabs(x_(0)) < 0.001 and fabs(x_(1)) < 0.001){
    x_(0) = 0.001;
    x_(1) = 0.001;
  }
  
  return x_;
}

/**
 *  Initialize weights
 */
void UKF::InitializeWeights() {
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  
  for (int i = 1; i < weights_.size(); i++) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
}

void UKF::InitializeMeasurements(MeasurementPackage measurement_pack) {
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;
  
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    x_ << InitializeRadarMeasurement(measurement_pack);
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    x_ << InitializeLaserMeasurement(measurement_pack);
  }
  
  InitializeWeights();
  
  time_us_ = measurement_pack.timestamp_;
  is_initialized_ = true;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {

  if (!is_initialized_) {
    InitializeMeasurements(measurement_pack);
    return;
  }
  
  double dt = (measurement_pack.timestamp_ - time_us_);
  dt /= 1000000.0;
  time_us_ = measurement_pack.timestamp_;
  
  Prediction(dt);

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(measurement_pack);
  }
  if (measurement_pack.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(measurement_pack);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  
  MatrixXd Xsig_aug = AugmentedSigmaPoints(n_x_, n_aug_, std_a_, std_yawdd_, x_, P_);
  
  for (int i = 0; i< n_sig_; i++){
    Xsig_pred_ = PredictSigmaPoints(i, n_x_, n_aug_, delta_t, Xsig_aug);
  }
  

  PredictMeanAndCovariance(Xsig_pred_);
  
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  
  for (int i = 0; i < n_sig_; i++) {
    Zsig = SigmaPointsRadarSpace(Zsig, i);
  }
  
  UpdateState(meas_package, Zsig, n_z);
}

MatrixXd UKF::SigmaPointsRadarSpace(MatrixXd Zsig, int i) {
  double p_x = Xsig_pred_(0,i);
  double p_y = Xsig_pred_(1,i);
  double v  = Xsig_pred_(2,i);
  double yaw = Xsig_pred_(3,i);
  double v1 = cos(yaw)*v;
  double v2 = sin(yaw)*v;
  
  Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);
  Zsig(1,i) = atan2(p_y,p_x);
  Zsig(2,i) = (p_x*v1 + p_y*v2 ) / Zsig(0,i);
  
  return Zsig;
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;

  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);
  UpdateState(meas_package, Zsig, n_z);
}

void UKF::UpdateState(MeasurementPackage meas_package, MatrixXd Zsig, int n_z){

  VectorXd z_pred = VectorXd(n_z);
  z_pred  = Zsig * weights_;
  
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  
  for (int i = 0; i < n_sig_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(&(z_diff(1)));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z, n_z);
  
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    R = R_radar_;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
    R = R_lidar_;
  }
  
  S = S + R;
  
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0.0);
  
  for (int i = 0; i < n_sig_; i++) {

    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
      NormalizeAngle(&(z_diff(1)));
    }

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    NormalizeAngle(&(x_diff(3)));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  VectorXd z = meas_package.raw_measurements_;

  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;
  
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    NormalizeAngle(&(z_diff(1)));
  }

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    NIS_radar_ = z.transpose() * S.inverse() * z;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
    NIS_laser_ = z.transpose() * S.inverse() * z;
  }
}


/**
 * Augment Sigma Points
 */
MatrixXd UKF::AugmentedSigmaPoints(int n_x, int n_aug, double std_a,
                                   double std_yawdd, const VectorXd& x,
                                   const MatrixXd& P) {
  
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;
  
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  Xsig_aug.col(0) = x_aug;
  
  MatrixXd L = P_aug.llt().matrixL();
  
  double sqrt_lambda_n_aug = sqrt(lambda_+n_aug_);
  VectorXd sqrt_lambda_n_aug_L;
  
  for(int i = 0; i < n_aug_; i++) {
    sqrt_lambda_n_aug_L = sqrt_lambda_n_aug * L.col(i);
    Xsig_aug.col(i+1) = x_aug + sqrt_lambda_n_aug_L;
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt_lambda_n_aug_L;
  }
  
  return Xsig_aug;
}


/**
 * Predict Sigma Points
 */
MatrixXd UKF::PredictSigmaPoints(int i, int n_x, int n_aug, double delta_t,
                                 const MatrixXd& Xsig_aug) {
  double delta_t2 = delta_t * delta_t;
  double p_x = Xsig_aug(0,i);
  double p_y = Xsig_aug(1,i);
  double v = Xsig_aug(2,i);
  double yaw = Xsig_aug(3,i);
  double yawd = Xsig_aug(4,i);
  double nu_a = Xsig_aug(5,i);
  double nu_yawdd = Xsig_aug(6,i);
  double sin_yaw = sin(yaw);
  double cos_yaw = cos(yaw);
  double arg = yaw + yawd * delta_t;
  
  double px_p, py_p;
  
  if (fabs(yawd) > 0.001) {
    double v_yawd = v/yawd;
    px_p = p_x + v_yawd * (sin(arg) - sin_yaw);
    py_p = p_y + v_yawd * (cos_yaw - cos(arg) );
  }
  else {
    double v_delta_t = v * delta_t;
    px_p = p_x + v_delta_t * cos_yaw;
    py_p = p_y + v_delta_t * sin_yaw;
  }
  
  double v_p = v;
  double yaw_p = arg;
  double yawd_p = yawd;
  
  px_p += 0.5 * nu_a * delta_t2 * cos_yaw;
  py_p += 0.5 * nu_a * delta_t2 * sin_yaw;
  v_p += nu_a * delta_t;
  yaw_p += 0.5 * nu_yawdd*delta_t2;
  yawd_p += nu_yawdd * delta_t;
  
  Xsig_pred_(0,i) = px_p;
  Xsig_pred_(1,i) = py_p;
  Xsig_pred_(2,i) = v_p;
  Xsig_pred_(3,i) = yaw_p;
  Xsig_pred_(4,i) = yawd_p;
  
  return Xsig_pred_;
}


void UKF::PredictMeanAndCovariance(MatrixXd Xsig_pred_) {
  x_ = Xsig_pred_ * weights_;
  P_.fill(0.0);
  
  for (int i = 0; i < n_sig_; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(&(x_diff(3)));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 *  Normalize angle between PI and -PI
 */
void UKF::NormalizeAngle(double *ang) {
  while (*ang > M_PI) *ang -= 2. * M_PI;
  while (*ang < -M_PI) *ang += 2. * M_PI;
}
