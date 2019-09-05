# Test for Discrete Time Linear-Quadratic Regulator module
# Should generate a straight line motion with quadratic slowing down to origin

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import dlqr

colors = [['#cf4c34','#078752','#6333ed','#d6ca54','#0ca0ad','#2ea62a','#c96b0d','#d947bb'],
          ['#4C1C13','#073824','#000000','#000000','#000000','#000000','#000000','#000000'],
          ['#FF745A','#0BD480','#EEEEEE','#EEEEEE','#EEEEEE','#EEEEEE','#EEEEEE','#EEEEEE']]

Asource = np.zeros((3,3))
Asource[0,2] = -1/np.sqrt(2)
Asource[1,2] = -1/np.sqrt(2)
Bsource = np.array([[0],[0],[1]])

# Dubins car Dynamics without turning
def dynamics(state,control):
    """Dynamic drift function of state over time

    Args:
        state (np.array): state at which to evaluate the dynamics
        control (np.array): exogenous (control) input at which to evaluate the dynamics

    Returns:
        derivative (np.array): the time derivative of state over time according to these dynamics
    """
    #derivative = np.array([state[2]* -1/np.sqrt(2),state[2]* -1/np.sqrt(2),0]) + np.array([0,0,control[0]])
    derivative = np.dot(Asource,state) + np.dot(Bsource,control)
    return(derivative)

def dynamics_deriv_x(state):
    """Derivative of dynamics with respect to state

    Args:
        state (np.array): state at which to evaluate the dynamics
        control (np.array): exogenous (control) input at which to evaluate the dynamics

    Returns:
        A (np.array): partial derivative of dynamics with respect to state
    """
    return(Asource)

def dynamics_deriv_u(control):
    """Derivative of dynamics with respect to control

    Args:
        state (np.array): state at which to evaluate the dynamics
        control (np.array): exogenous (control) input at which to evaluate the dynamics

    Returns:
        A (np.array): partial derivative of dynamics with respect to state
    """
    return(Bsource)

# Set problem parameters
N = 400 # Time Horizon
dimZ = 3 # Number of states
dimU = 1 # Number of controls
# Initialize state
z0 = np.array([10,10,15*np.sqrt(2)/4])
timestep_length = 0.01

# Optimization objective definition
Pfinal = np.eye(dimZ)*100
Pfinal[2,2] = 1
pfinal = np.zeros((dimZ,1))
cp = 10

# These arrays implicitly describe a quadratic form: $ x^T QQ x + x^T qq + cqq $
QQ = np.eye(dimZ)
QQ[2,2] = 0
QQ = np.matrix(QQ * timestep_length)
qq = np.zeros((dimZ,1))
cqq = 0
# These arrays implicitly describe a quadratic form: $ x^T RR x + x^T rr + crr $
RR = np.matrix([[0.1]])
#R = np.matrix([[0]])
RR = np.matrix(RR * timestep_length)
rr = np.zeros((dimU,1))
crr = 0

def runcost(state,control,timestep):
    """Running cost for optimal control problem

    Args:
        state (np.array): state at which to evaluate the dynamics
        control (np.array): exogenous (control) input at which to evaluate the dynamics
        timestep (int): current timestep (for time-varying costs)
    """
    running_control_cost = crr + np.dot(np.transpose(control),rr) + np.dot(np.dot(np.transpose(control),RR),control)
    running_state_cost = cqq + np.dot(np.transpose(state),qq) + np.dot(np.dot(np.transpose(state),QQ),state)
    return(running_control_cost + running_state_cost)

def runcost_deriv_statestate(state,control,timestep):
    """Second derivative of running cost with respect to state

    Args:
        state (np.array): state at which to evaluate the dynamics
        control (np.array): exogenous (control) input at which to evaluate the dynamics
        timestep (int): current timestep (for time-varying costs)
    """
    return(2*QQ)

def runcost_deriv_state(state,control,timestep):
    """First derivative of running cost with respect to state

    Args:
        state (np.array): state at which to evaluate the dynamics
        control (np.array): exogenous (control) input at which to evaluate the dynamics
        timestep (int): current timestep (for time-varying costs)
    """
    return(qq + 2*QQ*np.matrix(state).T)

def runcost_deriv_controlcontrol(state,control,timestep):
    """Second derivative of running cost with respect to the exogenous (control) input

    Args:
        state (np.array): state at which to evaluate the dynamics
        control (np.array): exogenous (control) input at which to evaluate the dynamics
        timestep (int): current timestep (for time-varying costs)
    """
    return(2*RR)

def runcost_deriv_control(state,control,timestep):
    """First derivative of running cost with respect to the exogenous (control) input

    Args:
        state (np.array): state at which to evaluate the dynamics
        control (np.array): exogenous (control) input at which to evaluate the dynamics
        timestep (int): current timestep (for time-varying costs)
    """
    return(rr + 2*RR*np.matrix(control).T)

# Sanity check on LQR (our implementation) for double integrator car
QN = dlqr.QuadraticForm(dimZ,QQ,qq,cqq)
RN = dlqr.QuadraticForm(dimZ,RR,rr,crr)
PN = [dlqr.QuadraticForm(dimZ,Pfinal,pfinal,cp) for n in np.arange(0,N+1)]
KN = np.zeros((dimU,dimZ,N)) # Allocate the gain schedule array
kN = np.zeros((dimU,1,N)) # Allocate the offset schedule array
lamb = 0
step_size = 1
for i in ((N-1)-np.arange(0,N)):
    #print(i,PN[i+1].Z)
    KN[:,:,i], kN[:,:,i], PN[i], eigvals = dlqr.onestep_dlqr_affine(np.eye(dimZ)+timestep_length*Asource,timestep_length*Bsource,QN,RN,PN[i+1])

cost = 0
control = np.zeros((dimU,N+1))
states = np.zeros((dimZ,N+1))
states[:,0] = z0
for i in np.arange(0,N):
    control[:,i] = step_size*(np.dot(KN[:,:,i],states[:,i]) + 1*kN[:,0,i])
    states[:,i+1] = states[:,i] + timestep_length * dynamics(states[:,i],control[:,i])
    cost = cost + runcost(states[:,i],control[:,i],i)

plt.plot(states[0,:],states[1,:],color=colors[0][1])
plt.plot(states[0,np.arange(0,20)*20],states[1,np.arange(0,20)*20],'ro',color=colors[1][1])
print(cost)
states[:,N]
