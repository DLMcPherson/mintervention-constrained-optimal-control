from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.linalg as LA
import level_sets

class DynamicsABC(object):
    __metaclass__ = ABCMeta
    """
    Abstract class that defines the interface required by the reactor simulator

    This is the dynamic equation for a dynamical differential equation of the form:

    $  \dot{x(t)} = f(x(t) ,u(t) )  $

    In application, this differential equation will be approximated via a discrete-time difference equation:

    $ x_{t+1} = x_t + f(x_t, u_t)*\delta $

    where $\delta$ is the sampling period of this time discretization.
    """

    @abstractmethod
    def f(self,state,control):
        """Dynamic drift function of state over time

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            derivative (np.array): the time derivative of state over time according to these dynamics
        """
        return(0)

    @abstractmethod
    def deriv_x(self,state,control):
        """Derivative of dynamics with respect to state

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            A (np.array): partial derivative of dynamics with respect to state
        """
        return([0])

    @abstractmethod
    def deriv_u(self,state,control):
        """Derivative of dynamics with respect to control

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            B (np.array): partial derivative of dynamics with respect to state
        """
        return([0])

    def __init__(self,dimZ,dimU):
        self.dimZ = dimZ
        self.dimU = dimU

# Dubins car Dynamics
class DubinsCarDynamics(DynamicsABC):

    """
    Encapsulates the extended unicycle/Dubins dynamics used to model car motion.

    Operates on a four-dimensional state space (z) with numbered states encoding:

      0. Horizontal position (x)
      1. Vertical position (y)
      2. Heading measured from the x-axis in radians (theta)
      3. Forward speed (v)

    with the form:

    dx/dt = v cos(theta)
    dy/dt = v sin(theta)
    d\theta / dt = u1
    dv/dt = u0
    """

    def f(self,state,control):
        """Dynamic drift function of state over time

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            derivative (np.array): the time derivative of state over time according to these dynamics
        """
        derivative = np.array([state[3]*np.cos(state[2]),state[3]*np.sin(state[2]),0,0]) + np.array([0,0,control[1],control[0]])
        return(derivative)

    def deriv_x(self,state,control):
        """Derivative of dynamics with respect to state

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            A (np.array): partial derivative of dynamics with respect to state
        """
        A = np.zeros((4,4))
        A[0,2] = state[3] * -np.sin(state[2])
        A[1,2] = state[3] *  np.cos(state[2])
        A[0,3] = np.cos(state[2])
        A[1,3] = np.sin(state[2])
        return(A)

    def deriv_u(self,state,control):
        """Derivative of dynamics with respect to control

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            B (np.array): partial derivative of dynamics with respect to state
        """
        B = np.array([[0,0],[0,0],[0,1],[1,0]])
        return(B)

    def __init__(self):
        super(DubinsCarDynamics,self).__init__(4,2)

# Dubins car Dynamics
class DubinsCarBrakelessDynamics(DynamicsABC):

    """
    Encapsulates the unicycle/Dubins dynamics used to model car motion.

    Operates on a three-dimensional state space (z) with numbered states encoding:

      1. Horizontal position (x)
      2. Vertical position (y)
      3. Heading measured from the x-axis in radians (theta)

    with the form:

    dx/dt = v cos(theta)
    dy/dt = v sin(theta)
    d\theta / dt = u
    """

    def f(self,state,control):
        """Dynamic drift function of state over time

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            derivative (np.array): the time derivative of state over time according to these dynamics
        """
        Az = np.array([self.speed*np.cos(state[2]),
                        self.speed*np.sin(state[2]),
                        0])
        Bu = np.array([0,0,control[0]])
        derivative = Az + Bu
        return(derivative)

    def deriv_x(self,state,control):
        """Derivative of dynamics with respect to state

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            A (np.array): partial derivative of dynamics with respect to state
        """
        A = np.zeros((3,3))
        A[0,2] = self.speed * -np.sin(state[2])
        A[1,2] = self.speed *  np.cos(state[2])
        return(A)

    def deriv_u(self,state,control):
        """Derivative of dynamics with respect to control

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            B (np.array): partial derivative of dynamics with respect to state
        """
        B = np.array([[0],[0],[1]])
        return(B)

    def __init__(self):
        self.speed = 3
        super(DubinsCarBrakelessDynamics,self).__init__(3,1)

# Double integrator dynamics
class DoubleDoubleIntegrator(DynamicsABC):

    """
    Encodes two decoupled copies of double integrator dynamics

    Operates on a four-dimensional state space (z) with numbered states encoding:

      1. Horizontal position (x)
      2. Horizontal velocity (v)
      3. Vertical position (y)
      4. Vertical velocity (w)

    with the form:

    dx/dt = v
    dv/dt = u_0
    dy/dt = w
    dw/dt = u_1
    """

    def f(self,state,control):
        """Dynamic drift function of state over time

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            derivative (np.array): the time derivative of state over time according to these dynamics
        """
        Az = np.array([state[1],0,state[3],0])
        Bu = np.array([0,control[0],0,control[1]])
        derivative = Az + Bu
        return(derivative)

    def deriv_x(self,state,control):
        """Derivative of dynamics with respect to state

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            A (np.array): partial derivative of dynamics with respect to state
        """
        A = np.zeros((4,4))
        A[0,1] = 1
        A[2,3] = 1
        return(A)

    def deriv_u(self,state,control):
        """Derivative of dynamics with respect to control

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            B (np.array): partial derivative of dynamics with respect to state
        """
        B = np.array([[0,0],[1,0],[0,0],[0,1]])
        return(B)

    def __init__(self):
        super(DoubleDoubleIntegrator,self).__init__(4,2)

class InterventionDynamicsWrapper(DynamicsABC):
    """
    Wraps another dynamics object to add perpetual safety controlling on a safe set.

    Implements the dynamical differential equation:

    $  \dot{x(t)} = f(x(t) , u*(t) )  $

    """

    def __init__(self, _wrappedDynamic,_safeSet):
        self.wrappedDynamic = _wrappedDynamic
        self.safeSet = _safeSet
        self.uMax = 1.0
        super(InterventionDynamicsWrapper,self).__init__(_wrappedDynamic.dimZ,_wrappedDynamic.dimU)

    def f(self,state,control):
        """Dynamic drift function of state over time

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            derivative (np.array): the time derivative of state over time according to these dynamics
        """
        optControl = np.zeros(control.shape)
        gradV = self.safeSet.gradient(state)
        # We assume that the wrapped dynamic is control affine, so that the closed form optimal control
        # is to bang-bang within the control bounds dictated by the sign of the control coefficient multiplied by the gradient
        controlCoefficient = self.wrappedDynamic.deriv_u(state,control)
        hamiltonianControlCoefficient = np.dot(gradV,controlCoefficient)
        optControl = np.sign(hamiltonianControlCoefficient)*self.uMax
        xdot = self.wrappedDynamic.f(state,optControl)
        return(xdot)

    def deriv_x(self,state,control):
        """Derivative of dynamics with respect to state

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            A (np.array): partial derivative of dynamics with respect to state
        """
        return(self.wrappedDynamic.deriv_x(state,control)) # TODO: add controller dependent term

    def deriv_u(self,state,control):
        """Derivative of dynamics with respect to control

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            B (np.array): partial derivative of dynamics with respect to state
        """
        return(0)

class MinimumInterventionDynamicsWrapper(DynamicsABC):
    """
    Wraps another dynamics object to add safety intervention on a safe set.

    Implements the dynamical differential equation:

    $  \dot{x(t)} = f(x(t) , S(u(t),x(t)) )  $

    where S(u,x) shifts between freedom of control outside the safe set to
    enacting the optimally safe action $u_{safe}$ inside the safe set described
    by V(x):

    $ S(u, x) = H(V(x)) u + (1 - H(V(x))) u_{safe}

    H() could be the indicator or (for numerics) a more infinitely differentiable
    function such as the logistic sigmoid or arctangent. It just needs to be between 0 and
    1 everywhere and equal to zero for all points inside the safe set.

    """

    def __init__(self, _wrappedDynamic,_safeSet):
        self.wrappedDynamic = _wrappedDynamic
        self.safeSet = _safeSet
        self.uMax = 1.0
        super(MinimumInterventionDynamicsWrapper,self).__init__(_wrappedDynamic.dimZ,_wrappedDynamic.dimU)

    def f(self,state,control):
        """Dynamic drift function of state over time

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            derivative (np.array): the time derivative of state over time according to these dynamics
        """
        # Calculate the safe action
        safeControl = np.zeros(control.shape)
        gradV = self.safeSet.gradient(state)
        # We assume that the wrapped dynamic is control affine, so that the closed form optimal control
        # is to bang-bang within the control bounds dictated by the sign of the control coefficient multiplied by the gradient
        controlCoefficient = self.wrappedDynamic.deriv_u(state,control)
        hamiltonianControlCoefficient = np.dot(gradV,controlCoefficient)
        safeControl = np.sign(hamiltonianControlCoefficient)*self.uMax
        #safeControl.reshape((1,))

        interpolation = self.logistic(self.safeSet.value(state))
        interpolatedControl = interpolation * control + (1-interpolation) * safeControl
        xdot = self.wrappedDynamic.f(state,interpolatedControl)
        return(xdot)

    def deriv_x(self,state,control):
        """Derivative of dynamics with respect to state

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            A (np.array): partial derivative of dynamics with respect to state
        """
        # Calculate the safe action
        safeControl = np.zeros(control.shape)
        gradV = self.safeSet.gradient(state)
        # We assume that the wrapped dynamic is control affine, so that the closed form optimal control
        # is to bang-bang within the control bounds dictated by the sign of the control coefficient multiplied by the gradient
        controlCoefficient = self.wrappedDynamic.deriv_u(state,control)
        hamiltonianControlCoefficient = np.dot(gradV,controlCoefficient)
        safeControl = np.sign(hamiltonianControlCoefficient)*self.uMax

        interpolation = self.logistic(self.safeSet.value(state))

        dustar = 0 # TODO: add this (discontinuous) term that acknowledges how the safe action changes with state

        dSdX = ( np.outer((control - safeControl),gradV) * self.deriv_logistic(self.safeSet.value(state))
            + (1-interpolation) * dustar);

        return(self.wrappedDynamic.deriv_x(state,control)
                + np.dot(self.wrappedDynamic.deriv_u(state,control),dSdX));

    def deriv_u(self,state,control):
        """Derivative of dynamics with respect to control

        Args:
            state (np.array): state at which to evaluate the dynamics
            control (np.array): exogenous (control) input at which to evaluate the dynamics

        Returns:
            B (np.array): partial derivative of dynamics with respect to state
        """
        return(self.wrappedDynamic.deriv_u(state,control)*self.logistic(self.safeSet.value(state)))

    def logistic(self,value):
        tightness = 40
        knee_position = -5
        interpolation =  1/(np.exp(-tightness*value-knee_position)+1)
        return(interpolation)

    def deriv_logistic(self,value):
        tightness = 40
        knee_position = -5
        exponential = np.exp(-tightness*value-knee_position)
        interpolation =  tightness*exponential/np.power((exponential+1),2)
        return(interpolation)
