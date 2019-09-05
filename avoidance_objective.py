import numpy as np
import numpy.linalg as LA
import objective_functions as of

class CollisionAvoidance(of.RunningCostABC):
    """
    Cost to penalize the x-y distance between vehicles
    """

    def __init__(self,avoid_states):
        """Initialize the running cost

        Args:
            state_cost (QuadraticForm): quadratic penatly on state
            control_cost (QuadraticForm): quadratic penalty on exogenous (control) input
        """
        super(CollisionAvoidance,self).__init__()
        self.eps = 0.01
        self.AVstates = avoid_states
        self.dimZ = 4

    def f(self,state,control,timestep):
        """Running cost for optimal control problem

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        # Penalize the x-y distance between vehicles
        avoiding_state_cost = 1/np.power(self.eps+LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),2)
        #avoiding_state_cost = np.exp(-np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),2))
        return(avoiding_state_cost)

    def deriv_statestate(self,state,control,timestep):
        """Second derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        avoiding_state_cost = np.zeros((self.dimZ,self.dimZ))
        avoiding_state_cost[0:2,0:2] = 8*np.outer(state[0:2]-self.AVstates[0:2,timestep],state[0:2]-self.AVstates[0:2,timestep])/(self.eps+np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),6))\
            -2*np.eye(2)/(self.eps+np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),4))
        #avoiding_state_cost[0:2,0:2] = np.exp(-np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),2))\
        #    * (4*np.outer(state[0:2]-self.AVstates[0:2,timestep],state[0:2]-self.AVstates[0:2,timestep]) - 2*np.eye(2))
        #avoiding_state_cost = np.zeros((self.dimZ,self.dimZ))
        return(avoiding_state_cost)

    def deriv_state(self,state,control,timestep):
        """First derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        avoiding_state_cost = np.zeros((self.dimZ,1))
        avoiding_state_cost[0:2] = -2*np.matrix(state[0:2]-self.AVstates[0:2,timestep]).T/(self.eps+np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),4))
        #avoiding_state_cost[0:2] = np.exp(-np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),2))*-2*np.matrix(state[0:2]-self.AVstates[0:2,timestep]).T
        #print("avoidance: ",avoiding_state_cost)
        #no_reverse_cost = -2/np.power(np.abs(state[3]),3)
        #avoiding_state_cost = np.zeros((self.dimZ,1))
        return(avoiding_state_cost)

    def deriv_controlcontrol(self,state,control,timestep):
        """Second derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(np.zeros((control.shape[0],control.shape[0])))

    def deriv_control(self,state,control,timestep):
        """First derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(np.matrix(np.zeros(control.shape)).T)

class CollisionAvoidance3(of.RunningCostABC):
    """
    Cost to penalize the x-y distance between vehicles
    """

    def __init__(self,avoid_states):
        """Initialize the running cost

        Args:
            state_cost (QuadraticForm): quadratic penatly on state
            control_cost (QuadraticForm): quadratic penalty on exogenous (control) input
        """
        super(CollisionAvoidance3,self).__init__()
        self.eps = 0.01
        self.radius = 1.8
        self.AVstates = avoid_states
        self.dimZ = 3

    def f(self,state,control,timestep):
        """Running cost for optimal control problem

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        # Penalize the x-y distance between vehicles
        avoiding_state_cost = 1/np.power(self.eps+max(0,-self.radius+LA.norm(state[0:2]-self.AVstates[0:2,timestep],2)),2)
        #avoiding_state_cost = np.exp(-np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),2))
        return(avoiding_state_cost)

    def deriv_statestate(self,state,control,timestep):
        """Second derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        avoiding_state_cost = np.zeros((self.dimZ,self.dimZ))
        avoiding_state_cost[0:2,0:2] = 8*np.outer(state[0:2]-self.AVstates[0:2,timestep],state[0:2]-self.AVstates[0:2,timestep])/(self.eps-self.radius+np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),6))\
            -2*np.eye(2)/(self.eps-self.radius+np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),4))
        #avoiding_state_cost[0:2,0:2] = np.exp(-np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),2))\
        #    * (4*np.outer(state[0:2]-self.AVstates[0:2,timestep],state[0:2]-self.AVstates[0:2,timestep]) - 2*np.eye(2))
        #avoiding_state_cost = np.zeros((self.dimZ,self.dimZ))
        return(avoiding_state_cost)

    def deriv_state(self,state,control,timestep):
        """First derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        avoiding_state_cost = np.zeros((self.dimZ,1))
        avoiding_state_cost[0:2] = -2*np.matrix(state[0:2]-self.AVstates[0:2,timestep]).T/(self.eps-self.radius+np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),4))
        #avoiding_state_cost[0:2] = np.exp(-np.power(LA.norm(state[0:2]-self.AVstates[0:2,timestep],2),2))*-2*np.matrix(state[0:2]-self.AVstates[0:2,timestep]).T
        #print("avoidance: ",avoiding_state_cost)
        #no_reverse_cost = -2/np.power(np.abs(state[3]),3)
        #avoiding_state_cost = np.zeros((self.dimZ,1))
        return(avoiding_state_cost)

    def deriv_controlcontrol(self,state,control,timestep):
        """Second derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(np.zeros((control.shape[0],control.shape[0])))

    def deriv_control(self,state,control,timestep):
        """First derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(np.matrix(np.zeros(control.shape)).T)
