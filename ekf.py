import numpy as np

# load the data
dataset = 1
odom_data = np.load('dataset/' + 'poses_world_{:02d}.npz'.format(dataset))
poses_world = odom_data['poses_world']

class EKF:
    '''
    Extended Kalman FIlter for state estimation
    '''
    def __init__(self, observations):
        self.observations = observations

        self.mu_k = np.array([0.0, 0.0, 0.0, 0.0]) # mean @ {t-1}: x, y, yaw, linear_velocity
        self.sigma_k = 1e-8 * np.eye(4)            # covariance of state
        self.R = np.diag([1e-8, 1e-8, 1e-8, 1e-8])    # process noise covariance
        self.Q = np.diag([1e-8, 1e-8, 1e-8])          # measurement noise covariance
    
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
        anglular_vel = delta_pose[3] / self.dt
        
        u = np.array([linear_vel, anglular_vel])

        return u

    def dynamics_model(self, timestamp):
        '''
        Motion model:
        x_{k+1} = x_{k} + dt*linear_velocity*cos(yaw_{k})
        y_{k+1} = y_{k} + dt*linear_velocity*sin(yaw_{k})
        yaw_{k+1} = yaw_{k} + dt*angular_velocity
        linear_velocity_{k+1} = linear_velocity_{k}
        '''
        # compute the control input
        dt = self.dt
        self.u = self.get_control(timestamp)

        # propagate the dynamics: X_{k+1} = A X_{k} + B U_{k} + noise
        A = np.eye(4)
        B = np.array([[dt*np.cos(self.mu_k[2]),  0.0],
                      [dt*np.sin(self.mu_k[2]),  0.0],
                      [                    0.0,   dt],
                      [                    0.0,  0.0]])
        
        self.mu_k1 = A @ self.mu_k + B @ self.u
    
    def prediction(self):
        '''
        Estimate the state using the dynamics
        '''

        # compute the jacobian based on the motion model
        dt = self.dt
        state_jacobian = np.array([[1.0, 0.0, -dt*self.u[0]*np.sin(self.mu_k[2]), dt*np.cos(self.mu_k[2])],
                                   [0.0, 1.0,  dt*self.u[0]*np.cos(self.mu_k[2]), dt*np.sin(self.mu_k[2])],
                                   [0.0, 0.0,                                1.0,                     0.0],
                                   [0.0, 0.0,                                0.0,                     1.0]])

        # estimate the covariance at {k+1} given {k}
        self.sigma_k1 = state_jacobian @ self.sigma_k @ state_jacobian.T + self.R

        # return the mean and covariance at {k+1} given {k}
        return self.mu_k1, self.sigma_k1
    
    def measurement_model(self):
        '''
        Estimate the measurement (x, y, yaw)
        z_{k+1} = C * x_{k+1}  i.e. extract (x, y, yaw) from the state. Since in this case, we just
        need to extract the (x, y, yaw), we just need linear observation model. For other nature of
        observations, non-linear observation model might be needed.
        '''
        self.C = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [1.0, 0.0, 1.0, 0.0]])
        
        z_k1_hat = self.C @ self.mu_k1

        return z_k1_hat
    
    def update(self, timestamp):
        '''
        Update the mean and covaraince using the observation at time {k+1}
        '''
        z_k1_true = self.observations[timestamp]
        z_k1_hat = self.measurement_model()
        innovation = z_k1_true - z_k1_hat

        # compute the kalman gain
        K = self.sigma_k1 @ self.C.T @ np.linalg.inv(self.C @ self.sigma_k1 @ self.C.T + self.Q)

        # update mean and covariance
        self.mu_k1 = self.mu_k1 + K @ innovation
        self.sigma_k1 = (np.eye(4) - K @ self.C) @ self.sigma_k1

        return self.mu_k1, self.sigma_k
