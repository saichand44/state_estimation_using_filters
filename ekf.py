import numpy as np
import matplotlib.pyplot as plt

class EKF:
    '''
    Extended Kalman FIlter for state estimation
    '''
    def __init__(self, observations):
        self.observations = observations

        self.num_data = len(self.observations)
        self.mu_k = np.array([[0.0, 0.0, 0.0, 0.0]]).reshape(4, 1)  # mean @ {t-1}: x, y, yaw, linear_velocity
        self.sigma_k = 1e-8 * np.eye(4)               # covariance of state
        self.R = np.diag([1e-8, 1e-8, 1e-8, 1e-8])    # process noise covariance
        self.Q = np.diag([1e-8, 1e-8, 1e-8])          # measurement noise covariance

        self.state_mean = np.zeros((self.num_data, 4, 1))
        self.state_cov = np.zeros((self.num_data, 4, 4))
        self.state_mean[0, :] = self.mu_k
        self.state_cov[0, :, :] = self.sigma_k
    
    def get_control(self, timestamp):
        '''
        Use the pose at time t+1 and t to calculate what control the robot could have taken
        at time t at state (x,y,th)_{t} to come to the next state (x,y,th)_{t+1}. Assumption
        is that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t+1.

        NOTE: Here the observations (x, y, yaw) are w.r.t a static coordinate system. Hence
        we can directly take the difference of the states to get the contol input. If the 
        observations are w.r.t relative coordinate system (ex. lidar), we need to take into
        account the rotation when computing the differences

        Returns
        -------
        u : numpy array
            Control input at the timestamp {t} of the format (linear_velocity, angular_velocity)
        '''
        self.next_timestamp = timestamp+1

        # compute the control taken at {t}
        self.dt = self.next_timestamp - timestamp
        delta_pose = self.observations[timestamp+1] - self.observations[timestamp]
        linear_vel = np.sqrt(delta_pose[0]**2 + delta_pose[1]**2) / self.dt
        anglular_vel = delta_pose[2] / self.dt

        u = np.array([[linear_vel, anglular_vel]])

        return u.reshape(2,1)

    def dynamics_model(self, mu_k, uk):
        '''
        Motion model:
        x_{k+1} = x_{k} + dt*linear_velocity*cos(yaw_{k})
        y_{k+1} = y_{k} + dt*linear_velocity*sin(yaw_{k})
        yaw_{k+1} = yaw_{k} + dt*angular_velocity
        linear_velocity_{k+1} = linear_velocity_{k}

        Returns
        -------
        mu_k1 : numpy array
                Mean of the state estimation at {k+1} using the mean, control at time {k}
        '''
        # compute the control input
        dt = self.dt

        # propagate the dynamics: X_{k+1} = A X_{k} + B U_{k}
        A = np.eye(4)
        B = np.array([[dt*np.cos(mu_k[2][0]),  0.0],
                      [dt*np.sin(mu_k[2][0]),  0.0],
                      [               0.0,   dt],
                      [               0.0,  0.0]])

        mu_k1 = A @ mu_k + B @ uk

        return mu_k1
    
    def prediction(self, mu_k, uk, sigma_k):
        '''
        Estimate the state using the dynamics

        Returns
        -------
        mu_k1 : numpy array
                Mean of the state estimation at {k+1} using the mean, control at time {k}
        sigma_k1 : numpy array
                   Covariance of the state estimation at {k+1} using the covariance, 
                   state jacobian at time {k}, process noise covariance
        '''

        # compute the jacobian based on the motion model
        dt = self.dt
        state_jacobian = np.array([[1.0, 0.0, -dt*uk[0][0]*np.sin(mu_k[2][0]), dt*np.cos(mu_k[2][0])],
                                   [0.0, 1.0,  dt*uk[0][0]*np.cos(mu_k[2][0]), dt*np.sin(mu_k[2][0])],
                                   [0.0, 0.0,                             1.0,                   0.0],
                                   [0.0, 0.0,                             0.0,                   1.0]])

        # estimate the mean, covariance at {k+1} given {k}
        self.mu_k1 = self.dynamics_model(mu_k, uk)
        self.sigma_k1 = state_jacobian @ sigma_k @ state_jacobian.T + self.R

        # return the mean and covariance at {k+1} given {k}
        return self.mu_k1, self.sigma_k1
    
    def measurement_model(self):
        '''
        Estimate the measurement (x, y, yaw)
        z_{k+1} = C * x_{k+1}  i.e. extract (x, y, yaw) from the state. Since in this case, we just
        need to extract the (x, y, yaw), we just need linear observation model. For other nature of
        observations, non-linear observation model might be needed.

        Returns
        -------
        z_k1_hat : numpy array
                   Estimate of observation at {k+1} using the mean estimated at time {k+1}

        '''
        self.C = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])
        
        z_k1_hat = self.C @ self.mu_k1

        return z_k1_hat
    
    def update(self, timestamp):
        '''
        Update the mean and covaraince using the observation at time {k+1}

        Returns
        -------
        mu_k1 : numpy array
                Mean of the state estimation at {k+1} using the state estimate, true observation at {k+1}
        sigma_k1 : numpy array
                   Covariance of the state estimation at {k+1} using the state estimate, 
                   true observation at {k+1}
        '''
        z_k1_true = self.observations[timestamp]
        z_k1_hat = self.measurement_model()
        innovation = z_k1_true.reshape(-1,1) - z_k1_hat

        # compute the kalman gain
        K = self.sigma_k1 @ self.C.T @ np.linalg.inv(self.C @ self.sigma_k1 @ self.C.T + self.Q)

        # update mean and covariance
        self.mu_k1 = self.mu_k1 + K @ innovation
        self.sigma_k1 = (np.eye(4) - K @ self.C) @ self.sigma_k1

        return self.mu_k1, self.sigma_k1

    def run_ekf(self):
        '''
        Run the EKF implementation
        '''
        mu_k, sigma_k = self.mu_k, self.sigma_k   # initial estimate of mean, covariance

        for i in range(self.num_data-1): #self.num_data-1
            uk = self.get_control(i)                # get the control input
            self.prediction(mu_k, uk, sigma_k)      # estimate the state
            mu_k, sigma_k = self.update(i+1)        # update the filter using observation

            self.state_mean[i+1, :, :] = mu_k
            self.state_cov[i+1, :, :] = sigma_k
        
        return self.state_mean, self.state_cov

if __name__ == '__main__':
    # load the data
    dataset = 0
    odom_data = np.load('dataset/' + 'poses_world_{:02d}.npz'.format(dataset))
    poses_world = odom_data['poses_world']

    ekf = EKF(poses_world)
    mean, cov = ekf.run_ekf()

    # plot the ground truths and ekf
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    # Plot the x coordinate
    ax1.plot(poses_world[:, 0], label='Ground Truth')
    ax1.plot(mean[:, 0].flatten(), label='EKF')
    ax1.set_xlabel('No. of observations')
    ax1.set_ylabel('x')
    ax1.legend()

    # Plot the y coordinate
    ax2.plot(poses_world[:, 1], label='Ground Truth')
    ax2.plot(mean[:, 1].flatten(), label='EKF')
    ax2.set_xlabel('No. of observations')
    ax2.set_ylabel('y')
    ax2.legend()

    # Plot the yaw angle
    ax3.plot(poses_world[:, 2], label='Ground Truth')
    ax3.plot(mean[:, 2].flatten(), label='EKF')
    ax3.set_xlabel('No. of observations')
    ax3.set_ylabel('yaw angle (radians)')
    ax3.legend()

    # plot the entire odometry (x, y, yaw angle)
    length = 0.1
    for i, ((x_true, y_true, yaw_true), (x_ekf, y_ekf, yaw_ekf)) in enumerate(zip(poses_world, mean[:, :3]),1):

        # true poses
        dx_true = length * np.cos(yaw_true)  # Calculate arrow delta x
        dy_true = length * np.sin(yaw_true)  # Calculate arrow delta y
        ax4.quiver(x_true, y_true, dx_true, dy_true, angles='xy', scale_units='xy', 
                   scale=0.1, width=0.005, headwidth=5, color='red')

        # ekf poses
        dx_ekf = length * np.cos(yaw_ekf)  # Calculate arrow delta x
        dy_ekf = length * np.sin(yaw_ekf)  # Calculate arrow delta y
        ax4.quiver(x_ekf, y_ekf, dx_ekf, dy_ekf, angles='xy', scale_units='xy', 
                   scale=0.1, width=0.005, headwidth=5, color='blue')

    ax4.set_title('Odometry Plot')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.legend(['Ground Truth', 'EKF'])

    plt.show()