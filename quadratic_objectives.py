import numpy as np
import numpy.linalg as LA
import objective_functions as of

class QuadraticStateCost(of.RunningCostABC):
    def __init__(self,state_cost):
        """Initialize the running cost

        Args:
            state_cost (QuadraticForm): quadratic penatly on state
            control_cost (QuadraticForm): quadratic penalty on exogenous (control) input
        """
        self.state_cost = state_cost
        super(QuadraticStateCost,self).__init__()

    def f(self,state,control,timestep):
        """Running cost for optimal control problem

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        #running_control_cost = crr + np.dot(np.transpose(control),rr) + np.dot(np.dot(np.transpose(control),RR),control)
        #running_state_cost = cqq + np.dot(np.transpose(state),qq) + np.dot(np.dot(np.transpose(state),QQ),state)
        return(self.state_cost.evaluate(state))

    def deriv_statestate(self,state,control,timestep):
        """Second derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        #return(2*QQ)
        return(self.state_cost.hessian())

    def deriv_state(self,state,control,timestep):
        """First derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        #return(qq + 2*QQ*np.matrix(state).T)
        return(self.state_cost.derivative(state))

    def deriv_controlcontrol(self,state,control,timestep):
        """Second derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        #return(2*RR)
        dimU = control.shape[0]
        return(np.zeros((dimU,dimU)))

    def deriv_control(self,state,control,timestep):
        """First derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        #return(rr + 2*RR*np.matrix(control).T)
        dimU = control.shape[0]
        return(np.zeros((dimU,1)))

class QuadraticControlCost(of.RunningCostABC):
    def __init__(self,control_cost):
        """Initialize the running cost

        Args:
            state_cost (QuadraticForm): quadratic penatly on state
            control_cost (QuadraticForm): quadratic penalty on exogenous (control) input
        """
        self.control_cost = control_cost
        super(QuadraticControlCost,self).__init__()

    def f(self,state,control,timestep):
        """Running cost for optimal control problem

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        #running_control_cost = crr + np.dot(np.transpose(control),rr) + np.dot(np.dot(np.transpose(control),RR),control)
        #running_state_cost = cqq + np.dot(np.transpose(state),qq) + np.dot(np.dot(np.transpose(state),QQ),state)
        return(self.control_cost.evaluate(control))

    def deriv_statestate(self,state,control,timestep):
        """Second derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        #return(2*QQ)
        dimZ = state.shape[0]
        return(np.zeros((dimZ,dimZ)))

    def deriv_state(self,state,control,timestep):
        """First derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        #return(qq + 2*QQ*np.matrix(state).T)
        dimZ = state.shape[0]
        return(np.zeros((dimZ,1)))

    def deriv_controlcontrol(self,state,control,timestep):
        """Second derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        #return(2*RR)
        return(self.control_cost.hessian())

    def deriv_control(self,state,control,timestep):
        """First derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        #return(rr + 2*RR*np.matrix(control).T)
        return(self.control_cost.derivative(control))
