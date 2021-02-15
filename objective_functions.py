from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.linalg as LA

class RunningCostABC(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def f(self,state,control,timestep):
        """Running cost for optimal control problem

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(0)

    @abstractmethod
    def deriv_statestate(self,state,control,timestep):
        """Second derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)

        Returns:
            Hessian (np.matrix): second derivative with dimension z by z, where z is the dimension of the state
                vector

        This method is used to produce a second-order Taylor approximation of the cost around a guess trajectory.
        """
        return(np.matrix([[0]]))

    @abstractmethod
    def deriv_state(self,state,control,timestep):
        """First derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(np.array([0]))

    @abstractmethod
    def deriv_controlcontrol(self,state,control,timestep):
        """Second derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)

        Returns:
            Hessian (np.matrix): second derivative with dimension z by z, where z is the dimension of the control
                vector

        This method is used to produce a second-order Taylor approximation of the cost around a guess trajectory.

        """
        return(np.matrix([[0]]))

    @abstractmethod
    def deriv_control(self,state,control,timestep):
        """First derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(np.array([0]))

    def __add__(self, other):
        return SummedRunningCost(self,other)

    def __mul__(self, scalar):
        return ScaledRunningCost(self,scalar)

    def __init__(self):
        pass
        #super().__init__()

class SummedRunningCost(RunningCostABC):

    def __init__(self,costA,costB):
        self.costA = costA
        self.costB = costB
        super(SummedRunningCost,self).__init__()

    def f(self,state,control,timestep):
        """Running cost for optimal control problem

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(self.costA.f(state,control,timestep) + self.costB.f(state,control,timestep))

    def deriv_statestate(self,state,control,timestep):
        """Second derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)

        Returns:
            Hessian (np.matrix): second derivative with dimension z by z, where z is the dimension of the state
                vector

        This method is used to produce a second-order Taylor approximation of the cost around a guess trajectory.
        """
        return(self.costA.deriv_statestate(state,control,timestep)
                + self.costB.deriv_statestate(state,control,timestep))

    def deriv_state(self,state,control,timestep):
        """First derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(self.costA.deriv_state(state,control,timestep)
                + self.costB.deriv_state(state,control,timestep))

    def deriv_controlcontrol(self,state,control,timestep):
        """Second derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)

        Returns:
            Hessian (np.matrix): second derivative with dimension z by z, where z is the dimension of the control
                vector

        This method is used to produce a second-order Taylor approximation of the cost around a guess trajectory.

        """
        return(self.costA.deriv_controlcontrol(state,control,timestep)
                + self.costB.deriv_controlcontrol(state,control,timestep))

    def deriv_control(self,state,control,timestep):
        """First derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(self.costA.deriv_control(state,control,timestep)
                + self.costB.deriv_control(state,control,timestep))

class ScaledRunningCost(RunningCostABC):
    def __init__(self,cost0,scalar):
        self.cost0 = cost0
        self.scalar = scalar
        super(ScaledRunningCost,self).__init__()

    def f(self,state,control,timestep):
        """Running cost for optimal control problem

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(self.scalar*self.cost0.f(state,control,timestep))

    def deriv_statestate(self,state,control,timestep):
        """Second derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)

        Returns:
            Hessian (np.matrix): second derivative with dimension z by z, where z is the dimension of the state
                vector

        This method is used to produce a second-order Taylor approximation of the cost around a guess trajectory.
        """
        return(self.scalar*self.cost0.deriv_statestate(state,control,timestep))

    def deriv_state(self,state,control,timestep):
        """First derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(self.scalar*self.cost0.deriv_state(state,control,timestep))

    def deriv_controlcontrol(self,state,control,timestep):
        """Second derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)

        Returns:
            Hessian (np.matrix): second derivative with dimension z by z, where z is the dimension of the control
                vector

        This method is used to produce a second-order Taylor approximation of the cost around a guess trajectory.

        """
        return(self.scalar*self.cost0.deriv_controlcontrol(state,control,timestep))

    def deriv_control(self,state,control,timestep):
        """First derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(self.scalar*self.cost0.deriv_control(state,control,timestep))

class TimeSequencedCosts(RunningCostABC):
    def __init__(self,costs):
        self.costs = costs
        super(TimeSequencedCosts,self).__init__()

    def f(self,state,control,timestep):
        """Running cost for optimal control problem

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(self.costs[timestep].f(state,control,timestep))

    def deriv_statestate(self,state,control,timestep):
        """Second derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)

        Returns:
            Hessian (np.matrix): second derivative with dimension z by z, where z is the dimension of the state
                vector

        This method is used to produce a second-order Taylor approximation of the cost around a guess trajectory.
        """
        return(self.costs[timestep].deriv_statestate(state,control,timestep))

    def deriv_state(self,state,control,timestep):
        """First derivative of running cost with respect to state

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(self.costs[timestep].deriv_state(state,control,timestep))

    def deriv_controlcontrol(self,state,control,timestep):
        """Second derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)

        Returns:
            Hessian (np.matrix): second derivative with dimension z by z, where z is the dimension of the control
                vector

        This method is used to produce a second-order Taylor approximation of the cost around a guess trajectory.

        """
        return(self.costs[timestep].deriv_controlcontrol(state,control,timestep))

    def deriv_control(self,state,control,timestep):
        """First derivative of running cost with respect to exogenous (control) input

        Args:
            state (np.array): state at which to evaluate the cost function
            control (np.array): exogenous (control) input at which to evaluate the cost function
            timestep (int): current timestep (for time-varying costs)
        """
        return(self.costs[timestep].deriv_control(state,control,timestep))

#======================================================================#
#============================ TERMINAL COSTS ==========================#
#======================================================================#
# The terminal cost is only used to judge the final state the rollout termintes
# in. When interpreting the finite horizon rollout as approximating a larger
# (infinite) horizon we're embedded in, the terminal cost approximates the
# cost-to-go for where we pruned the game-tree.
#   Note that the terminal cost only judges state, since no more control inputs
# are executed after the final state. We piggyback on running-cost objects since
# they are a richer representation that we can chop terms out of. This piggybacking
# is done in interest of #KeepingItDRY ("Don't Repeat Yourself").

class TerminalCost(object):
    """Terminal cost class built off of more featured running cost class.

    This class provides the interface needed for terminal costs and enforces the
    non-input and time dependence of terminal costs by repressing those
    derivative functions and passing zero inputs in for input.
    """

    def __init__(self,truncatedRunCost):
        self.cost = truncatedRunCost

    def f(self,state):
        """Terminal cost for optimal control problem

        Args:
            state (np.array): state at which to evaluate the terminal cost
        """
        return(self.cost.f(state,0,0))

    def deriv_statestate(self,state):
        """Second derivative of terminal cost with respect to state

        Args:
            state (np.array): state at which to evaluate the terminal cost

        Returns:
            Hessian (np.matrix): second derivative with dimension z by z, where z is the dimension of the state
                vector

        This method is used to produce a second-order Taylor approximation of the cost around a guess trajectory.
        """
        return(self.cost.deriv_statestate(state,0,0))

    def deriv_state(self,state):
        """First derivative of terminal cost with respect to state

        Args:
            state (np.array): state at which to evaluate the terminal cost
        """
        return(self.cost.deriv_state(state,0,0))

    def __add__(self, other):
        return TerminalCost(SummedRunningCost(self.cost,other.cost))

    def __mul__(self, scalar):
        return TerminalCost(ScaledRunningCost(self.cost,scalar))

class costToGoLS(object):
    def __init__(self,levelset):
        self.levelset = levelset

    def f(self,state):
        """Terminal cost for optimal control problem

        Args:
            state (np.array): state at which to evaluate the terminal cost
        """
        return(self.levelset.value(state).reshape(1,1) )

    def deriv_statestate(self,state):
        """Second derivative of terminal cost with respect to state

        Args:
            state (np.array): state at which to evaluate the terminal cost

        Returns:
            Hessian (np.matrix): second derivative with dimension z by z, where z is the dimension of the state
                vector

        This method is used to produce a second-order Taylor approximation of the cost around a guess trajectory.
        """
        return(self.levelset.hessian(state) )

    def deriv_state(self,state):
        """First derivative of terminal cost with respect to state

        Args:
            state (np.array): state at which to evaluate the terminal cost
        """
        return(self.levelset.gradient(state).reshape(self.levelset.dim,1) )
